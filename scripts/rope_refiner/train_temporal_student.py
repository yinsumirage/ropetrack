#!/usr/bin/env python3
"""Train a causal GRU residual on top of a frozen framewise alpha student."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.actions import ACTION_SPACES, alpha_dim
from ropetrack.refine.alpha_student import (
    STUDENT_FEATURE_DIM,
    features_from_cache,
    normalize_features,
    student_from_payload,
)
from ropetrack.refine.temporal import (
    PastResidualRopeAlphaStudent,
    TemporalRopeAlphaStudent,
    build_causal_windows,
    deterministic_sequence_split,
    load_temporal_checkpoint,
    prepare_temporal_cache,
    save_past_residual_checkpoint,
    save_temporal_checkpoint,
    shuffle_history as shuffle_temporal_history,
    temporal_feature_stats,
    temporal_features,
)

ALPHA_L2 = 1e-4


def _load_teacher(teacher_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    with np.load(teacher_dir / "refiner_eval_cache.npz", allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    alpha = np.asarray(
        np.load(teacher_dir / "alpha.npy", allow_pickle=False), dtype=np.float32
    )
    summary = json.loads((teacher_dir / "summary.json").read_text(encoding="utf-8"))
    return cache, alpha, summary


def _subset_cache(
    cache: dict[str, np.ndarray], rows: np.ndarray
) -> dict[str, np.ndarray]:
    indices = np.asarray(rows, dtype=np.int64)
    return {
        key: np.asarray(cache[key])[indices].copy()
        for key in (
            "sample_id",
            "base_hand_pose",
            "base_rope_norm",
            "input_rope_norm",
            "rope_valid",
        )
    }


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _framewise_feature_stats(config: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(config.get("feature_mean"), dtype=np.float32)
    std = np.asarray(config.get("feature_std"), dtype=np.float32)
    if mean.shape != (STUDENT_FEATURE_DIM,) or std.shape != (STUDENT_FEATURE_DIM,):
        raise ValueError("framewise feature_mean/feature_std must both be [65]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.0):
        raise ValueError("framewise feature statistics must be finite with positive std")
    return mean, std


def _framewise_alpha(
    model: torch.nn.Module,
    config: dict,
    cache: dict[str, np.ndarray],
    device: str,
) -> torch.Tensor:
    mean, std = _framewise_feature_stats(config)
    features = normalize_features(features_from_cache(cache), mean, std)
    with torch.no_grad():
        return model(torch.from_numpy(features).to(device)).detach()


def _shuffle_rope_within(
    cache: dict[str, np.ndarray],
    partitions: tuple[np.ndarray, ...],
    seed: int,
) -> None:
    sample_ids = np.asarray(cache["sample_id"])
    if sample_ids.ndim != 1:
        raise ValueError("sample_id must be one-dimensional")
    rng = np.random.default_rng(seed)
    for partition in partitions:
        rows = np.asarray(partition, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= len(sample_ids)):
            raise ValueError("shuffle partition contains an out-of-range row")
        permutation = rng.permutation(rows)
        for key in ("base_rope_norm", "input_rope_norm", "rope_valid"):
            values = np.asarray(cache[key])
            cache[key][rows] = values[permutation]


def _normalized_windows(
    cache: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    history_length: int,
    raw_frame_step: int,
    history_step: int,
    *,
    delta_step: int | None = None,
    shuffle: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = (
        temporal_features(cache, raw_frame_step=raw_frame_step)
        if delta_step is None
        else temporal_features(
            cache, raw_frame_step=raw_frame_step, delta_step=delta_step
        )
    )
    normalized = ((features - mean) / std).astype(np.float32)
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
        raise ValueError("every temporal window must include its current frame")
    if shuffle:
        windows = shuffle_temporal_history(windows, valid, seed)
    return windows, valid, lengths


def _past_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    logit_residual: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.l1_loss(prediction, target) + ALPHA_L2 * (
        logit_residual * logit_residual
    ).mean()


def _base_temporal_alpha(
    model: torch.nn.Module,
    config: dict,
    framewise: torch.nn.Module,
    framewise_config: dict,
    cache: dict[str, np.ndarray],
    device: str,
    raw_frame_step: int,
    history_step: int,
    feature_delta_step: int,
) -> torch.Tensor:
    base_alpha = _framewise_alpha(framewise, framewise_config, cache, device)
    mean = np.asarray(config["temporal_feature_mean"], dtype=np.float32)
    std = np.asarray(config["temporal_feature_std"], dtype=np.float32)
    features = temporal_features(
        cache,
        raw_frame_step=raw_frame_step,
        delta_step=feature_delta_step,
    )
    normalized = ((features - mean) / std).astype(np.float32, copy=False)
    windows, _, lengths = build_causal_windows(
        cache["sample_id"],
        normalized,
        history_length=1,
        raw_frame_step=raw_frame_step,
        history_step=history_step,
    )
    with torch.no_grad():
        return model(
            torch.from_numpy(windows).to(device),
            torch.from_numpy(lengths).to(device),
            base_alpha,
        ).detach()


def _validate_framewise_provenance(
    config: dict,
    split,
    split_seed: int,
    action_space: str,
    out_dim: int,
    teacher_dir: Path,
    num_samples: int,
) -> tuple[float, float | None]:
    if config.get("action_space") != action_space:
        raise ValueError("framewise action_space does not match the teacher")
    if int(config.get("out_dim", -1)) != out_dim:
        raise ValueError("framewise out_dim does not match the teacher")
    if int(config.get("in_dim", -1)) != STUDENT_FEATURE_DIM:
        raise ValueError("framewise in_dim must be 65")
    if int(config.get("image_feature_dim", 0)) != 0:
        raise ValueError("framewise checkpoint must not use image features")
    try:
        max_alpha = float(config["max_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("framewise max_alpha is required") from exc
    if not math.isfinite(max_alpha) or max_alpha <= 0.0:
        raise ValueError("framewise max_alpha must be positive and finite")
    if config.get("split_by") != "sequence":
        raise ValueError("framewise split_by must be sequence")
    if int(config.get("split_seed", -1)) != int(split_seed):
        raise ValueError("framewise split_seed does not match the temporal split")
    if list(config.get("train_sequences", [])) != list(split.train_sequences):
        raise ValueError("framewise train_sequences do not match the temporal split")
    if list(config.get("val_sequences", [])) != list(split.val_sequences):
        raise ValueError("framewise val_sequences do not match the temporal split")
    if int(config.get("num_train", -1)) != len(split.train_idx):
        raise ValueError("framewise num_train does not match the temporal split")
    if int(config.get("num_val", -1)) != len(split.val_idx):
        raise ValueError("framewise num_val does not match the temporal split")
    sources = config.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ValueError("framewise config.sources must contain exactly one teacher")
    source = sources[0]
    if not isinstance(source, dict) or "dir" not in source:
        raise ValueError("framewise config.sources must record the teacher dir")
    try:
        source_dir = Path(source["dir"]).expanduser().resolve()
        source_count = int(source["num_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("framewise config.sources has invalid provenance") from exc
    if source_dir != teacher_dir.expanduser().resolve() or source_count != num_samples:
        raise ValueError("framewise config.sources does not match the temporal teacher")
    _, feature_std = _framewise_feature_stats(config)
    if np.any(feature_std[STUDENT_FEATURE_DIM - 5 : STUDENT_FEATURE_DIM] != 1.0):
        raise ValueError("framewise validity feature std must be 1.0")
    gate = config.get("gate_threshold")
    return max_alpha, None if gate is None else float(gate)


def _validate_base_temporal_provenance(
    config: dict,
    framewise_config: dict,
    split,
    split_seed: int,
    action_space: str,
    out_dim: int,
    teacher_dir: Path,
    num_samples: int,
    raw_frame_step: int,
    history_step: int,
    feature_delta_step: int,
) -> tuple[float, float | None]:
    max_alpha, gate = _validate_framewise_provenance(
        framewise_config,
        split,
        split_seed,
        action_space,
        out_dim,
        teacher_dir,
        num_samples,
    )
    for key, expected in (
        ("model_type", "causal_gru"),
        ("action_space", action_space),
        ("out_dim", out_dim),
        ("in_dim", 70),
        ("history_length", 1),
        ("raw_frame_step", raw_frame_step),
        ("history_step", history_step),
        ("split_by", "sequence"),
        ("split_seed", int(split_seed)),
        ("train_sequences", list(split.train_sequences)),
        ("val_sequences", list(split.val_sequences)),
        ("num_train", len(split.train_idx)),
        ("num_val", len(split.val_idx)),
        (
            "teacher_source",
            {"dir": str(teacher_dir), "num_samples": num_samples},
        ),
    ):
        if config.get(key) != expected:
            raise ValueError(f"base temporal {key} does not match V2 training")
    if config.get("gate_threshold") != gate:
        raise ValueError("base temporal gate_threshold does not match framewise")
    if float(config.get("max_alpha", float("nan"))) != max_alpha:
        raise ValueError("base temporal max_alpha does not match framewise")
    if feature_delta_step != raw_frame_step:
        raise ValueError(
            "feature_delta_step must match the schema-v1 base raw_frame_step"
        )
    for key in ("temporal_feature_mean", "temporal_feature_std"):
        values = np.asarray(config.get(key), dtype=np.float32)
        if values.shape != (70,) or not np.isfinite(values).all():
            raise ValueError(f"base temporal {key} must be finite [70]")
        if key.endswith("std") and np.any(values <= 0.0):
            raise ValueError("base temporal temporal_feature_std must be positive")
    return max_alpha, gate


def train_temporal_student(
    teacher_dir: Path,
    framewise_checkpoint: Path | None,
    action_space: str,
    out_dir: Path,
    *,
    base_temporal_checkpoint: Path | None = None,
    history_length: int = 8,
    raw_frame_step: int = 1,
    inference_raw_frame_step: int | None = None,
    history_step: int = 1,
    feature_delta_step: int | None = None,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 20,
    val_frac: float = 0.2,
    split_seed: int = 20260710,
    seed: int = 0,
    sensor_mode: str = "normal",
    shuffle_rope: bool = False,
    shuffle_history: bool = False,
    aug_noise_std: float = 0.05,
    aug_dropout: float = 0.1,
    aug_bias_std: float = 0.0,
    aug_scale_range: float = 0.0,
    device: str = "cpu",
) -> dict:
    if isinstance(teacher_dir, (list, tuple)):
        raise ValueError("temporal training supports exactly one teacher_dir")
    teacher_dir = Path(teacher_dir)
    out_dir = Path(out_dir)
    is_v2 = base_temporal_checkpoint is not None
    if (framewise_checkpoint is None) == (base_temporal_checkpoint is None):
        raise ValueError(
            "provide exactly one of framewise_checkpoint or base_temporal_checkpoint"
        )
    framewise_checkpoint = (
        None if framewise_checkpoint is None else Path(framewise_checkpoint)
    )
    base_temporal_checkpoint = (
        None
        if base_temporal_checkpoint is None
        else Path(base_temporal_checkpoint)
    )
    if action_space not in ACTION_SPACES:
        raise ValueError(f"unsupported action_space: {action_space}")
    history_length = _positive_int("history_length", history_length)
    raw_frame_step = _positive_int("raw_frame_step", raw_frame_step)
    history_step = _positive_int("history_step", history_step)
    if is_v2:
        if inference_raw_frame_step is None or feature_delta_step is None:
            raise ValueError(
                "V2 training requires inference_raw_frame_step and feature_delta_step"
            )
        inference_raw_frame_step = _positive_int(
            "inference_raw_frame_step", inference_raw_frame_step
        )
        feature_delta_step = _positive_int(
            "feature_delta_step", feature_delta_step
        )
        if history_length <= 1:
            raise ValueError("V2 history_length must include at least one past slot")
        if history_step % raw_frame_step or feature_delta_step % raw_frame_step:
            raise ValueError(
                "history_step and feature_delta_step must be divisible by raw_frame_step"
            )
        if (
            history_step % inference_raw_frame_step
            or feature_delta_step % inference_raw_frame_step
        ):
            raise ValueError(
                "inference_raw_frame_step must divide history_step and feature_delta_step"
            )
    elif inference_raw_frame_step is not None or feature_delta_step is not None:
        raise ValueError(
            "inference_raw_frame_step and feature_delta_step require base_temporal_checkpoint"
        )
    hidden_dim = _positive_int("hidden_dim", hidden_dim)
    batch_size = _positive_int("batch_size", batch_size)
    max_epochs = _positive_int("max_epochs", max_epochs)
    patience = _positive_int("patience", patience)
    if not math.isfinite(float(lr)) or lr <= 0.0:
        raise ValueError("lr must be positive and finite")

    cache, teacher_alpha, teacher_summary = _load_teacher(teacher_dir)
    num_samples = len(cache["sample_id"])
    out_dim = alpha_dim(action_space)
    if teacher_summary.get("action_space") != action_space:
        raise ValueError("teacher action_space does not match the requested action_space")
    if teacher_alpha.shape != (num_samples, out_dim):
        raise ValueError(
            f"teacher alpha must be {(num_samples, out_dim)}, got {teacher_alpha.shape}"
        )
    if not np.isfinite(teacher_alpha).all():
        raise ValueError("teacher alpha must contain only finite values")
    if int(teacher_summary.get("num_samples", -1)) != num_samples:
        raise ValueError("teacher summary num_samples does not match the cache")

    split = deterministic_sequence_split(cache["sample_id"], val_frac, split_seed)
    train_cache = _subset_cache(cache, split.train_idx)
    val_cache = _subset_cache(cache, split.val_idx)
    train_target = teacher_alpha[split.train_idx].copy()
    val_target = teacher_alpha[split.val_idx].copy()
    base_temporal = None
    base_temporal_config = None
    base_temporal_payload = None
    if is_v2:
        base_temporal, base_temporal_config, framewise, framewise_config = (
            load_temporal_checkpoint(base_temporal_checkpoint, device)
        )
        base_temporal_payload = torch.load(
            base_temporal_checkpoint, map_location=device, weights_only=True
        )
        framewise_payload = base_temporal_payload["framewise"]
        max_alpha, gate_threshold = _validate_base_temporal_provenance(
            base_temporal_config,
            framewise_config,
            split,
            split_seed,
            action_space,
            out_dim,
            teacher_dir,
            num_samples,
            raw_frame_step,
            history_step,
            feature_delta_step,
        )
        base_temporal.requires_grad_(False)
    else:
        framewise_payload = torch.load(
            framewise_checkpoint, map_location=device, weights_only=True
        )
        framewise_config = framewise_payload["config"]
        max_alpha, gate_threshold = _validate_framewise_provenance(
            framewise_config,
            split,
            split_seed,
            action_space,
            out_dim,
            teacher_dir,
            num_samples,
        )
        framewise, _ = student_from_payload(framewise_payload, device)
    teacher_gate = teacher_summary.get("optimization", {}).get(
        "gate_residual_threshold"
    )
    if teacher_gate != gate_threshold:
        raise ValueError("teacher and framewise gate thresholds do not match")
    teacher_max_alpha = teacher_summary.get("optimization", {}).get("max_alpha")
    if teacher_max_alpha is not None and float(teacher_max_alpha) != max_alpha:
        raise ValueError("teacher and framewise max_alpha do not match")
    if float(np.abs(teacher_alpha).max(initial=0.0)) > max_alpha + 1e-6:
        raise ValueError("teacher alpha exceeds framewise max_alpha")

    framewise.eval()
    framewise.requires_grad_(False)
    if any(parameter.requires_grad for parameter in framewise.parameters()):
        raise ValueError("framewise parameters must be frozen")

    clean_features = (
        temporal_features(
            train_cache,
            raw_frame_step=raw_frame_step,
            delta_step=feature_delta_step,
        )
        if is_v2
        else temporal_features(train_cache, raw_frame_step=raw_frame_step)
    )
    temporal_mean, temporal_std = temporal_feature_stats(
        clean_features, np.arange(len(train_cache["sample_id"]))
    )
    clean_val_cache = prepare_temporal_cache(
        val_cache,
        sensor_mode=sensor_mode,
        seed=seed + 3,
        raw_frame_step=raw_frame_step,
    )
    val_rows = np.arange(len(val_cache["sample_id"]))
    if shuffle_rope:
        _shuffle_rope_within(
            clean_val_cache,
            (val_rows,),
            seed + 1,
        )
    clean_windows, _, clean_lengths = _normalized_windows(
        clean_val_cache,
        temporal_mean,
        temporal_std,
        history_length,
        raw_frame_step,
        history_step,
        delta_step=feature_delta_step if is_v2 else None,
        shuffle=shuffle_history,
        seed=seed + 2,
    )
    clean_base_alpha = (
        _base_temporal_alpha(
            base_temporal,
            base_temporal_config,
            framewise,
            framewise_config,
            clean_val_cache,
            device,
            raw_frame_step,
            history_step,
            feature_delta_step,
        )
        if is_v2
        else _framewise_alpha(framewise, framewise_config, clean_val_cache, device)
    )

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_class = PastResidualRopeAlphaStudent if is_v2 else TemporalRopeAlphaStudent
    model = model_class(
        in_dim=70,
        out_dim=out_dim,
        hidden_dim=hidden_dim,
        max_alpha=max_alpha,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_target_t = torch.from_numpy(train_target).to(device)
    val_target_t = torch.from_numpy(val_target).to(device)
    clean_windows_t = torch.from_numpy(clean_windows).to(device)
    clean_lengths_t = torch.from_numpy(clean_lengths).to(device)

    def clean_val_l1() -> float:
        model.eval()
        with torch.no_grad():
            prediction = model(
                clean_windows_t,
                clean_lengths_t,
                clean_base_alpha,
            )
            value = float(
                torch.nn.functional.l1_loss(prediction, val_target_t)
            )
        if not math.isfinite(value):
            raise FloatingPointError("non-finite clean validation loss")
        return value

    framewise_zero_baseline = clean_val_l1()
    best_val = framewise_zero_baseline
    best_epoch = -1
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    log = []

    for epoch in range(max_epochs):
        augmented = prepare_temporal_cache(
            train_cache,
            sensor_mode=sensor_mode,
            seed=seed + 100 + epoch,
            raw_frame_step=raw_frame_step,
            aug_noise_std=aug_noise_std,
            aug_dropout=aug_dropout,
            aug_bias_std=aug_bias_std,
            aug_scale_range=aug_scale_range,
        )
        train_rows = np.arange(len(train_cache["sample_id"]))
        if shuffle_rope:
            _shuffle_rope_within(
                augmented,
                (train_rows,),
                seed + 1_000 + epoch,
            )
        windows, _, lengths = _normalized_windows(
            augmented,
            temporal_mean,
            temporal_std,
            history_length,
            raw_frame_step,
            history_step,
            delta_step=feature_delta_step if is_v2 else None,
            shuffle=shuffle_history,
            seed=seed + 2_000 + epoch,
        )
        base_alpha = (
            _base_temporal_alpha(
                base_temporal,
                base_temporal_config,
                framewise,
                framewise_config,
                augmented,
                device,
                raw_frame_step,
                history_step,
                feature_delta_step,
            )
            if is_v2
            else _framewise_alpha(framewise, framewise_config, augmented, device)
        )
        windows_t = torch.from_numpy(windows).to(device)
        lengths_t = torch.from_numpy(lengths).to(device)

        model.train()
        order = rng.permutation(len(train_target))
        loss_sum = 0.0
        rows_seen = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            index = torch.from_numpy(batch).to(device)
            if is_v2:
                prediction, logit_residual = model.forward_with_residual(
                    windows_t[index], lengths_t[index], base_alpha[index]
                )
                loss = _past_residual_loss(
                    prediction, train_target_t[index], logit_residual
                )
            else:
                prediction = model(
                    windows_t[index], lengths_t[index], base_alpha[index]
                )
                l1 = torch.nn.functional.l1_loss(
                    prediction, train_target_t[index]
                )
                loss = l1 + ALPHA_L2 * (prediction * prediction).mean()
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite train loss at epoch {epoch}, batch {start // batch_size}"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss_value * len(batch)
            rows_seen += len(batch)

        val_l1 = clean_val_l1()
        log.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(rows_seen, 1),
                "clean_val_teacher_alpha_l1": val_l1,
            }
        )
        if val_l1 < best_val - 1e-8:
            best_val = val_l1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    config = {
        "model_type": "causal_gru",
        "in_dim": 70,
        "out_dim": out_dim,
        "hidden_dim": hidden_dim,
        "max_alpha": max_alpha,
        "action_space": action_space,
        "gate_threshold": gate_threshold,
        "history_length": history_length,
        "raw_frame_step": raw_frame_step,
        "history_step": history_step,
        "temporal_feature_mean": temporal_mean.tolist(),
        "temporal_feature_std": temporal_std.tolist(),
        "split_by": "sequence",
        "split_seed": int(split_seed),
        "val_frac": float(val_frac),
        "train_sequences": list(split.train_sequences),
        "val_sequences": list(split.val_sequences),
        "num_train": int(len(split.train_idx)),
        "num_val": int(len(split.val_idx)),
        "seed": int(seed),
        "lr": float(lr),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "training_device": str(device),
        "framewise_frozen": True,
        "framewise_split_seed": int(framewise_config["split_seed"]),
        "framewise_seed": framewise_config.get("seed"),
        "framewise_sources": copy.deepcopy(framewise_config["sources"]),
        "sensor_mode": sensor_mode,
        "shuffle_rope": bool(shuffle_rope),
        "shuffle_history": bool(shuffle_history),
        "augmentation": {
            "noise_std": float(aug_noise_std),
            "dropout": float(aug_dropout),
            "bias_std": float(aug_bias_std),
            "scale_range": float(aug_scale_range),
        },
        "clean_validation": {
            "augmentation": False,
            "sensor_mode": sensor_mode,
            "shuffle_rope": bool(shuffle_rope),
            "shuffle_history": bool(shuffle_history),
        },
        "alpha_l2": ALPHA_L2,
        "teacher_source": {"dir": str(teacher_dir), "num_samples": num_samples},
        "framewise_checkpoint": str(framewise_checkpoint),
    }
    if is_v2:
        config = {
            "schema_version": 2,
            "model_type": "causal_past_residual_gru",
            "in_dim": 70,
            "out_dim": out_dim,
            "hidden_dim": hidden_dim,
            "max_alpha": max_alpha,
            "action_space": action_space,
            "gate_threshold": gate_threshold,
            "history_length": history_length,
            "train_raw_frame_step": raw_frame_step,
            "inference_raw_frame_step": inference_raw_frame_step,
            "history_step": history_step,
            "feature_delta_step": feature_delta_step,
            "temporal_feature_mean": temporal_mean.tolist(),
            "temporal_feature_std": temporal_std.tolist(),
            "split_by": "sequence",
            "split_seed": int(split_seed),
            "val_frac": float(val_frac),
            "train_sequences": list(split.train_sequences),
            "val_sequences": list(split.val_sequences),
            "num_train": int(len(split.train_idx)),
            "num_val": int(len(split.val_idx)),
            "seed": int(seed),
            "lr": float(lr),
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "training_device": str(device),
            "base_temporal_frozen": True,
            "framewise_frozen": True,
            "sensor_mode": sensor_mode,
            "shuffle_rope": bool(shuffle_rope),
            "shuffle_history": bool(shuffle_history),
            "augmentation": {
                "noise_std": float(aug_noise_std),
                "dropout": float(aug_dropout),
                "bias_std": float(aug_bias_std),
                "scale_range": float(aug_scale_range),
            },
            "clean_validation": {
                "augmentation": False,
                "sensor_mode": sensor_mode,
                "shuffle_rope": bool(shuffle_rope),
                "shuffle_history": bool(shuffle_history),
            },
            "logit_residual_l2": ALPHA_L2,
            "teacher_source": {
                "dir": str(teacher_dir),
                "num_samples": num_samples,
            },
            "base_temporal_checkpoint": str(base_temporal_checkpoint),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_v2:
        save_past_residual_checkpoint(
            out_dir / "temporal_student.pt", model, config, base_temporal_payload
        )
        summary = {
            "base_temporal_zero_baseline_val_l1": framewise_zero_baseline,
            "best_temporal_val_l1": best_val,
            "beats_base_temporal": bool(best_val < framewise_zero_baseline - 1e-8),
            "best_epoch": best_epoch,
            "epochs_run": len(log),
            "config": config,
        }
    else:
        save_temporal_checkpoint(
            out_dir / "temporal_student.pt", model, config, framewise_payload
        )
        summary = {
            "framewise_zero_baseline_val_l1": framewise_zero_baseline,
            "best_temporal_val_l1": best_val,
            "beats_framewise": bool(best_val < framewise_zero_baseline - 1e-8),
            "best_epoch": best_epoch,
            "epochs_run": len(log),
            "config": config,
        }
    (out_dir / "train_log.json").write_text(
        json.dumps({"summary": summary, "log": log}, indent=2), encoding="utf-8"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--framewise-checkpoint", type=Path)
    base.add_argument("--base-temporal-checkpoint", type=Path)
    parser.add_argument("--action-space", choices=list(ACTION_SPACES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--raw-frame-step", type=int, default=1)
    parser.add_argument("--inference-raw-frame-step", type=int, default=None)
    parser.add_argument("--history-step", type=int, default=1)
    parser.add_argument("--feature-delta-step", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260710)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensor-mode", choices=("normal", "zero"), default="normal")
    parser.add_argument("--shuffle-rope", action="store_true")
    parser.add_argument("--shuffle-history", action="store_true")
    parser.add_argument("--aug-noise-std", type=float, default=0.05)
    parser.add_argument("--aug-dropout", type=float, default=0.1)
    parser.add_argument("--aug-bias-std", type=float, default=0.0)
    parser.add_argument("--aug-scale-range", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = train_temporal_student(
        args.teacher_dir,
        args.framewise_checkpoint,
        args.action_space,
        args.out_dir,
        base_temporal_checkpoint=args.base_temporal_checkpoint,
        history_length=args.history_length,
        raw_frame_step=args.raw_frame_step,
        inference_raw_frame_step=args.inference_raw_frame_step,
        history_step=args.history_step,
        feature_delta_step=args.feature_delta_step,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        split_seed=args.split_seed,
        seed=args.seed,
        sensor_mode=args.sensor_mode,
        shuffle_rope=args.shuffle_rope,
        shuffle_history=args.shuffle_history,
        aug_noise_std=args.aug_noise_std,
        aug_dropout=args.aug_dropout,
        aug_bias_std=args.aug_bias_std,
        aug_scale_range=args.aug_scale_range,
        device=args.device,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
