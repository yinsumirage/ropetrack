"""Causal sequence primitives for temporal rope refinement."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import operator
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ropetrack.refine.alpha_student import (
    STUDENT_FEATURE_DIM,
    features_from_cache,
    normalize_features,
    student_from_payload,
)
from ropetrack.refine.actions import ACTION_SPACES, alpha_dim


@dataclass(frozen=True)
class SequenceSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeFrame:
    episode_id: str | None
    phase: str
    episode_offset: int
    segment_id: str


class TemporalRopeAlphaStudent(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        max_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_alpha = float(max_alpha)
        if min(self.in_dim, self.out_dim, self.hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if not math.isfinite(self.max_alpha) or self.max_alpha <= 0.0:
            raise ValueError("max_alpha must be positive and finite")

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        windows: torch.Tensor,
        lengths: torch.Tensor,
        base_alpha: torch.Tensor,
    ) -> torch.Tensor:
        if windows.ndim != 3 or windows.shape[2] != self.in_dim:
            raise ValueError(
                f"windows must be [batch, history, {self.in_dim}], got {tuple(windows.shape)}"
            )
        batch_size, history_length, _ = windows.shape
        lengths = torch.as_tensor(lengths)
        if lengths.ndim != 1 or len(lengths) != batch_size:
            raise ValueError(f"lengths must be [{batch_size}], got {tuple(lengths.shape)}")
        if lengths.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("lengths must contain integers")
        if not torch.is_tensor(base_alpha) or not torch.is_floating_point(base_alpha):
            raise ValueError("base_alpha must be a floating-point tensor")
        if base_alpha.shape != (batch_size, self.out_dim):
            raise ValueError(
                f"base_alpha must be {(batch_size, self.out_dim)}, got {tuple(base_alpha.shape)}"
            )
        if bool(((lengths <= 0) | (lengths > history_length)).any().item()):
            raise ValueError(f"lengths must be positive and at most {history_length}")

        encoded = self.encoder(windows)
        packed = nn.utils.rnn.pack_padded_sequence(
            encoded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        delta = self.head(hidden[-1])
        epsilon = max(1e-6, torch.finfo(delta.dtype).eps)
        unit = (base_alpha.to(dtype=delta.dtype) / self.max_alpha).clamp(
            -1.0 + epsilon, 1.0 - epsilon
        )
        return self.max_alpha * torch.tanh(torch.atanh(unit) + delta)


def _validate_temporal_checkpoint_config(
    config: dict,
    framewise_config: dict,
    model: TemporalRopeAlphaStudent | None = None,
) -> tuple[int, int, int, float]:
    dimensions = {}
    for name in ("in_dim", "out_dim", "hidden_dim"):
        if name not in config:
            raise ValueError(f"temporal checkpoint missing {name}")
        value = config[name]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        dimensions[name] = int(value)

    if "max_alpha" not in config:
        raise ValueError("temporal checkpoint missing max_alpha")
    try:
        max_alpha = float(config["max_alpha"])
    except (TypeError, ValueError) as exc:
        raise ValueError("max_alpha must be positive and finite") from exc
    if not math.isfinite(max_alpha) or max_alpha <= 0.0:
        raise ValueError("max_alpha must be positive and finite")

    if model is not None:
        for name, value in dimensions.items():
            if value != getattr(model, name):
                raise ValueError(f"config {name}={value} does not match model {getattr(model, name)}")
        if max_alpha != model.max_alpha:
            raise ValueError(
                f"config max_alpha={max_alpha} does not match model {model.max_alpha}"
            )

    in_dim = dimensions["in_dim"]
    for name in ("temporal_feature_mean", "temporal_feature_std"):
        if name not in config:
            raise ValueError(f"temporal checkpoint missing {name}")
        values = np.asarray(config[name], dtype=np.float64)
        if values.shape != (in_dim,):
            raise ValueError(f"{name} must be [{in_dim}], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        if name == "temporal_feature_std" and np.any(values <= 0.0):
            raise ValueError("temporal_feature_std must be strictly positive")

    if "out_dim" not in framewise_config:
        raise ValueError("framewise checkpoint missing out_dim")
    try:
        framewise_out_dim = operator.index(framewise_config["out_dim"])
    except TypeError as exc:
        raise ValueError("framewise out_dim must be a positive integer") from exc
    if framewise_out_dim <= 0 or framewise_out_dim != dimensions["out_dim"]:
        raise ValueError(
            f"framewise out_dim={framewise_out_dim} does not match temporal out_dim={dimensions['out_dim']}"
        )

    try:
        framewise_max_alpha = float(framewise_config.get("max_alpha", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError("framewise max_alpha must be positive and finite") from exc
    if not math.isfinite(framewise_max_alpha) or framewise_max_alpha <= 0.0:
        raise ValueError("framewise max_alpha must be positive and finite")
    if framewise_max_alpha != max_alpha:
        raise ValueError(
            f"framewise max_alpha={framewise_max_alpha} does not match temporal max_alpha={max_alpha}"
        )

    return in_dim, dimensions["out_dim"], dimensions["hidden_dim"], max_alpha


def save_temporal_checkpoint(
    path: Path,
    model: TemporalRopeAlphaStudent,
    config: dict,
    framewise_payload: dict,
) -> None:
    _validate_temporal_checkpoint_config(config, framewise_payload["config"], model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {
                key: value.detach().clone() for key, value in model.state_dict().items()
            },
            "config": copy.deepcopy(config),
            "framewise": {
                "model_state": {
                    key: value.detach().clone()
                    for key, value in framewise_payload["model_state"].items()
                },
                "config": copy.deepcopy(framewise_payload["config"]),
            },
        },
        path,
    )


def load_temporal_checkpoint(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    in_dim, out_dim, hidden_dim, max_alpha = _validate_temporal_checkpoint_config(
        config, payload["framewise"]["config"]
    )
    model = TemporalRopeAlphaStudent(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        max_alpha=max_alpha,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    framewise, framewise_config = student_from_payload(payload["framewise"], device)
    framewise.requires_grad_(False)
    return model, config, framewise, framewise_config


def sequence_frame(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).replace("\\", "/").split("/")
    if len(parts) < 2 or not parts[-2] or not parts[-1].isdigit():
        raise ValueError(f"invalid temporal sample_id: {sample_id}")
    return parts[-2], int(parts[-1])


def deterministic_sequence_split(
    sample_ids, val_fraction: float, seed: int
) -> SequenceSplit:
    fraction = float(val_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be finite and strictly between 0 and 1")

    sequences = np.asarray([sequence_frame(sample_id)[0] for sample_id in sample_ids])
    names = sorted(
        set(sequences.tolist()),
        key=lambda name: (
            hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
            name,
        ),
    )
    n_val = max(1, math.ceil(len(names) * fraction))
    if n_val >= len(names):
        raise ValueError("sequence split must leave at least one training sequence")

    val_sequences = tuple(names[:n_val])
    train_sequences = tuple(names[n_val:])
    return SequenceSplit(
        train_idx=np.flatnonzero(np.isin(sequences, train_sequences)),
        val_idx=np.flatnonzero(np.isin(sequences, val_sequences)),
        train_sequences=train_sequences,
        val_sequences=val_sequences,
    )


def _positive_int(name: str, value: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _contiguous_segments(sample_ids, raw_frame_step: int):
    ids = np.asarray(sample_ids)
    if ids.ndim != 1:
        raise ValueError(f"sample_ids must be one-dimensional, got {ids.shape}")

    parsed = [sequence_frame(sample_id) for sample_id in ids]
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate temporal (sequence, frame) rows")

    grouped: dict[str, list[tuple[int, int]]] = {}
    for row, (sequence, frame) in enumerate(parsed):
        grouped.setdefault(sequence, []).append((frame, row))

    segments: list[list[tuple[int, int]]] = []
    for sequence in sorted(grouped):
        rows = sorted(grouped[sequence])
        start = 0
        for index in range(1, len(rows)):
            if rows[index][0] - rows[index - 1][0] != raw_frame_step:
                segments.append(rows[start:index])
                start = index
        segments.append(rows[start:])
    return ids, segments


def episode_schedule(
    sample_ids,
    context: int,
    masked: int,
    recovery: int,
    raw_frame_step: int,
) -> list[EpisodeFrame]:
    """Assign complete causal occlusion cycles inside each contiguous segment."""
    context = _positive_int("context", context)
    masked = _positive_int("masked", masked)
    recovery = _positive_int("recovery", recovery)
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    ids, segments = _contiguous_segments(sample_ids, raw_frame_step)
    cycle_length = context + masked + recovery
    output: dict[int, EpisodeFrame] = {}
    sequence_segment_index: dict[str, int] = {}

    for segment in segments:
        sequence = sequence_frame(ids[segment[0][1]])[0]
        segment_index = sequence_segment_index.get(sequence, 0)
        sequence_segment_index[sequence] = segment_index + 1
        segment_id = f"{sequence}:{segment_index}"
        complete_rows = len(segment) // cycle_length * cycle_length

        for position, (_, row) in enumerate(segment):
            if position >= complete_rows:
                output[row] = EpisodeFrame(None, "tail", position - complete_rows, segment_id)
                continue
            offset = position % cycle_length
            if offset < context:
                phase = "context"
            elif offset < context + masked:
                phase = "masked"
            else:
                phase = "recovery"
            episode_index = position // cycle_length
            output[row] = EpisodeFrame(
                f"{segment_id}:{episode_index}", phase, offset, segment_id
            )

    return [output[row] for row in range(len(ids))]


def causal_ema(
    sample_ids, values: np.ndarray, decay: float, raw_frame_step: int
) -> np.ndarray:
    """Causal EMA that resets at every sequence boundary or raw-frame gap."""
    try:
        decay = float(decay)
    except (TypeError, ValueError) as exc:
        raise ValueError("decay must be finite and in [0, 1)") from exc
    if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
        raise ValueError("decay must be finite and in [0, 1)")
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    ids, segments = _contiguous_segments(sample_ids, raw_frame_step)
    source = np.asarray(values, dtype=np.float32)
    if source.ndim == 0 or len(source) != len(ids):
        rows = 0 if source.ndim == 0 else len(source)
        raise ValueError(f"value/sample row mismatch: {rows} != {len(ids)}")
    if not np.isfinite(source).all():
        raise ValueError("EMA values must contain only finite values")

    out = np.empty_like(source)
    keep = np.float32(decay)
    update = np.float32(1.0 - decay)
    for segment in segments:
        state = source[segment[0][1]].copy()
        out[segment[0][1]] = state
        for _, row in segment[1:]:
            state = keep * state + update * source[row]
            out[row] = state
    return out


def temporal_feature_stats(
    features70: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features70, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 70:
        raise ValueError(f"features70 must be [N, 70], got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("features70 must contain only finite values")
    rows = np.asarray(train_idx)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("train_idx must be a non-empty one-dimensional index")
    if not np.issubdtype(rows.dtype, np.integer) or np.issubdtype(rows.dtype, np.bool_):
        raise ValueError("train_idx must contain integers")
    rows = rows.astype(np.int64, copy=False)
    if np.any(rows < 0) or np.any(rows >= len(features)):
        raise ValueError("train_idx contains an out-of-range row")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("train_idx contains duplicate rows")
    train = features[rows]
    mean = train.mean(axis=0).astype(np.float32)
    std = np.maximum(train.std(axis=0), 1e-4).astype(np.float32)
    std[STUDENT_FEATURE_DIM - 5 : STUDENT_FEATURE_DIM] = 1.0
    return mean, std


def prepare_temporal_cache(
    cache: dict[str, np.ndarray],
    sensor_mode: str,
    seed: int,
    raw_frame_step: int = 1,
    aug_noise_std: float = 0.0,
    aug_dropout: float = 0.0,
    aug_bias_std: float = 0.0,
    aug_scale_range: float = 0.0,
) -> dict[str, np.ndarray]:
    if sensor_mode not in {"normal", "zero"}:
        raise ValueError(f"unsupported sensor_mode: {sensor_mode}")
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    values = {
        "aug_noise_std": float(aug_noise_std),
        "aug_dropout": float(aug_dropout),
        "aug_bias_std": float(aug_bias_std),
        "aug_scale_range": float(aug_scale_range),
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("augmentation controls must be finite")
    if values["aug_noise_std"] < 0.0 or values["aug_bias_std"] < 0.0:
        raise ValueError("augmentation standard deviations must be non-negative")
    if not 0.0 <= values["aug_dropout"] <= 1.0:
        raise ValueError("aug_dropout must be in [0, 1]")
    if not 0.0 <= values["aug_scale_range"] <= 1.0:
        raise ValueError("aug_scale_range must be in [0, 1]")

    out = copy.deepcopy(cache)
    ids, segments = _contiguous_segments(out["sample_id"], raw_frame_step)
    rope = np.asarray(out["input_rope_norm"], dtype=np.float32).copy()
    valid = np.asarray(out["rope_valid"], dtype=bool).copy()
    if rope.shape != (len(ids), 5) or valid.shape != rope.shape:
        raise ValueError(
            f"input_rope_norm and rope_valid must be {(len(ids), 5)}, "
            f"got {rope.shape} and {valid.shape}"
        )
    if not np.isfinite(rope).all():
        raise ValueError("input_rope_norm must contain only finite values")

    if sensor_mode == "zero":
        rope.fill(0.0)
        valid.fill(False)
        out["input_rope_norm"] = rope
        out["rope_valid"] = valid
        return out

    rng = np.random.default_rng(seed)
    for segment in segments:
        rows = np.asarray([row for _, row in segment], dtype=np.int64)
        scale = 1.0 + rng.uniform(
            -values["aug_scale_range"], values["aug_scale_range"]
        )
        bias = rng.normal(0.0, values["aug_bias_std"], size=5).astype(np.float32)
        rope[rows] = rope[rows] * np.float32(scale) + bias
    if values["aug_noise_std"] > 0.0:
        rope += rng.normal(0.0, values["aug_noise_std"], size=rope.shape).astype(
            np.float32
        )
    rope = np.clip(rope, 0.0, 1.0)
    if values["aug_dropout"] > 0.0:
        valid &= rng.random(valid.shape) >= values["aug_dropout"]
    rope[~valid] = 0.0
    out["input_rope_norm"] = rope
    out["rope_valid"] = valid
    return out


def shuffle_history(
    windows: np.ndarray, valid: np.ndarray, seed: int
) -> np.ndarray:
    source = np.asarray(windows)
    mask = np.asarray(valid)
    if source.ndim < 3:
        raise ValueError(f"windows must be [N, history, ...], got {source.shape}")
    if mask.dtype != np.bool_ or mask.shape != source.shape[:2]:
        raise ValueError(
            f"valid must be a boolean mask with shape {source.shape[:2]}, got {mask.shape}"
        )
    lengths = mask.sum(axis=1)
    expected = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if np.any(lengths <= 0) or not np.array_equal(mask, expected):
        raise ValueError("valid must be non-empty and left-aligned")

    out = source.copy()
    rng = np.random.default_rng(seed)
    for row, length in enumerate(lengths):
        history = int(length) - 1
        if history > 1:
            out[row, :history] = source[row, rng.permutation(history)]
    return out


def build_causal_windows(
    sample_ids,
    features,
    history_length: int,
    raw_frame_step: int,
    history_step: int,
):
    history_length = _positive_int("history_length", history_length)
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    history_step = _positive_int("history_step", history_step)
    if history_step % raw_frame_step:
        raise ValueError("history_step must be divisible by raw_frame_step")

    ids, segments = _contiguous_segments(sample_ids, raw_frame_step)
    feature_rows = np.asarray(features)
    if feature_rows.ndim == 0 or len(feature_rows) != len(ids):
        rows = 0 if feature_rows.ndim == 0 else len(feature_rows)
        raise ValueError(f"feature/sample row mismatch: {rows} != {len(ids)}")

    windows = np.zeros(
        (len(ids), history_length, *feature_rows.shape[1:]), dtype=feature_rows.dtype
    )
    valid = np.zeros((len(ids), history_length), dtype=bool)

    for segment in segments:
        row_by_frame = dict(segment)
        for frame, output_row in segment:
            source_rows = [
                row_by_frame[target]
                for offset in range(history_length - 1, -1, -1)
                if (target := frame - offset * history_step) in row_by_frame
            ]
            windows[output_row, : len(source_rows)] = feature_rows[source_rows]
            valid[output_row, : len(source_rows)] = True

    return windows, valid, valid.sum(axis=1).astype(np.int64)


def temporal_features(cache: dict[str, np.ndarray], raw_frame_step: int = 1) -> np.ndarray:
    base = np.asarray(features_from_cache(cache), dtype=np.float32)
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    ids, segments = _contiguous_segments(cache["sample_id"], raw_frame_step)
    input_rope = np.asarray(cache["input_rope_norm"], dtype=np.float32)

    if base.shape != (len(ids), STUDENT_FEATURE_DIM):
        raise ValueError(
            f"base feature/sample shape mismatch: {base.shape} != "
            f"({len(ids)}, {STUDENT_FEATURE_DIM})"
        )
    if input_rope.shape != (len(ids), 5):
        raise ValueError(f"input_rope_norm must be {(len(ids), 5)}, got {input_rope.shape}")

    delta = np.zeros((len(ids), 5), dtype=np.float32)
    for segment in segments:
        for (_, previous_row), (_, current_row) in zip(segment, segment[1:]):
            delta[current_row] = input_rope[current_row] - input_rope[previous_row]

    return np.concatenate([base, delta], axis=1).astype(np.float32, copy=False)


def temporal_alpha(
    cache: dict[str, np.ndarray], checkpoint: Path, device: str
) -> tuple[np.ndarray, dict]:
    """Infer causal alphas from a self-contained temporal checkpoint."""
    model, config, framewise, framewise_config = load_temporal_checkpoint(
        checkpoint, device
    )
    for key in (
        "action_space",
        "gate_threshold",
        "history_length",
        "raw_frame_step",
        "history_step",
    ):
        if key not in config:
            raise ValueError(f"temporal checkpoint missing {key}")
    action_space = config["action_space"]
    if action_space not in ACTION_SPACES:
        raise ValueError(f"unsupported temporal action_space: {action_space}")
    if int(config["out_dim"]) != alpha_dim(action_space):
        raise ValueError("temporal out_dim does not match action_space")
    if int(config["in_dim"]) != 70:
        raise ValueError("temporal in_dim must be 70")

    gate = config["gate_threshold"]
    if gate is not None:
        try:
            gate = float(gate)
        except (TypeError, ValueError) as exc:
            raise ValueError("temporal gate_threshold must be non-negative and finite") from exc
        if not math.isfinite(gate) or gate < 0.0:
            raise ValueError("temporal gate_threshold must be non-negative and finite")

    if framewise_config.get("action_space") != action_space:
        raise ValueError("nested framewise action_space does not match temporal action_space")
    if framewise_config.get("gate_threshold") != config["gate_threshold"]:
        raise ValueError("nested framewise gate_threshold does not match temporal gate_threshold")
    if int(framewise_config.get("in_dim", -1)) != STUDENT_FEATURE_DIM:
        raise ValueError("nested framewise in_dim must be 65")
    if int(framewise_config.get("image_feature_dim", 0)) != 0:
        raise ValueError("temporal checkpoints cannot use framewise image features")

    frame_mean = np.asarray(framewise_config.get("feature_mean"), dtype=np.float32)
    frame_std = np.asarray(framewise_config.get("feature_std"), dtype=np.float32)
    if frame_mean.shape != (STUDENT_FEATURE_DIM,) or frame_std.shape != (
        STUDENT_FEATURE_DIM,
    ):
        raise ValueError("nested framewise feature statistics must both be [65]")
    if (
        not np.isfinite(frame_mean).all()
        or not np.isfinite(frame_std).all()
        or np.any(frame_std <= 0.0)
    ):
        raise ValueError("nested framewise feature statistics must be finite with positive std")

    history_length = _positive_int("history_length", config["history_length"])
    raw_frame_step = _positive_int("raw_frame_step", config["raw_frame_step"])
    history_step = _positive_int("history_step", config["history_step"])
    frame_features = normalize_features(
        features_from_cache(cache), frame_mean, frame_std
    ).astype(np.float32, copy=False)
    with torch.no_grad():
        base_alpha = framewise(torch.from_numpy(frame_features).to(device))

    features70 = temporal_features(cache, raw_frame_step=raw_frame_step)
    temporal_mean = np.asarray(config["temporal_feature_mean"], dtype=np.float32)
    temporal_std = np.asarray(config["temporal_feature_std"], dtype=np.float32)
    normalized = ((features70 - temporal_mean) / temporal_std).astype(
        np.float32, copy=False
    )
    windows, valid, lengths = build_causal_windows(
        cache["sample_id"],
        normalized,
        history_length,
        raw_frame_step,
        history_step,
    )
    rows = np.arange(len(normalized))
    if not valid[rows, lengths - 1].all() or not np.allclose(
        windows[rows, lengths - 1], normalized
    ):
        raise ValueError("every temporal inference window must include its current frame")
    with torch.no_grad():
        alpha = model(
            torch.from_numpy(windows).to(device),
            torch.from_numpy(lengths).to(device),
            base_alpha,
        )
    result = alpha.cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("temporal inference produced non-finite alpha")
    return result, config
