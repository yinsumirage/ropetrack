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
    student_from_payload,
)


@dataclass(frozen=True)
class SequenceSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]


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
