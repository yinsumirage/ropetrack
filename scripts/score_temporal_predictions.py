#!/usr/bin/env python3
"""Gap-safe temporal metrics for causal hand-pose predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ropetrack.io import load_pred_json, read_json, read_jsonl  # noqa: E402
from ropetrack.refine.analysis import json_sanitize, summarize_rope_residuals  # noqa: E402
from scripts.score_predictions import align_w_scale, point_distances  # noqa: E402
from scripts.score_sliced_predictions import (  # noqa: E402
    finger_joint_map,
    occluded_fingers_for_row,
    per_joint_pa_distances,
)


MASKED_HORIZONS = (1, 5, 15, 30, 60, 120, 240)

BOOTSTRAP_METRICS = (
    "pa_mpjpe_mm",
    "pa_mpvpe_mm",
    "masked_pa_mpjpe_mm",
    "masked_occluded_tip_pa_mpjpe_mm",
    "recovery_pa_mpjpe_mm",
    *(f"masked_h{horizon}_pa_mpjpe_mm" for horizon in MASKED_HORIZONS),
    "velocity_error_mm_s",
    "acceleration_error_mm_s2",
    "masked_velocity_error_mm_s",
    "masked_acceleration_error_mm_s2",
    "masked_jitter_mm_s2",
)


def _sequence_frame(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).replace("\\", "/").split("/")
    if len(parts) < 2 or not parts[-2] or not parts[-1].isdigit():
        raise ValueError(f"invalid temporal sample_id: {sample_id}")
    return parts[-2], int(parts[-1])


def _segments(sample_ids, raw_frame_step: int) -> list[np.ndarray]:
    if not isinstance(raw_frame_step, (int, np.integer)) or isinstance(raw_frame_step, (bool, np.bool_)) or raw_frame_step <= 0:
        raise ValueError("raw_frame_step must be a positive integer")
    ids = np.asarray(sample_ids)
    if ids.ndim != 1:
        raise ValueError(f"sample_ids must be one-dimensional, got {ids.shape}")
    parsed = [_sequence_frame(value) for value in ids]
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate temporal (sequence, frame) rows")
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row, (sequence, frame) in enumerate(parsed):
        grouped.setdefault(sequence, []).append((frame, row))
    out: list[np.ndarray] = []
    for sequence in sorted(grouped):
        rows = sorted(grouped[sequence])
        start = 0
        for index in range(1, len(rows)):
            if rows[index][0] - rows[index - 1][0] != raw_frame_step:
                out.append(np.asarray([row for _, row in rows[start:index]], dtype=np.int64))
                start = index
        out.append(np.asarray([row for _, row in rows[start:]], dtype=np.int64))
    return out


def phase_lag(gt, pred, max_lag: int = 15) -> int | None:
    """Correlation lag where positive means the prediction trails the GT."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.ndim != 1 or pred.shape != gt.shape or len(gt) < 2:
        raise ValueError(f"lag inputs must be matching one-dimensional arrays, got {gt.shape} and {pred.shape}")
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("lag inputs must contain only finite values")
    if not isinstance(max_lag, (int, np.integer)) or isinstance(max_lag, (bool, np.bool_)) or max_lag < 0:
        raise ValueError("max_lag must be a non-negative integer")

    best: tuple[float, int, int] | None = None
    for lag in range(-min(max_lag, len(gt) - 2), min(max_lag, len(gt) - 2) + 1):
        if lag < 0:
            left, right = gt[-lag:], pred[:lag]
        elif lag > 0:
            left, right = gt[:-lag], pred[lag:]
        else:
            left, right = gt, pred
        left = left - left.mean()
        right = right - right.mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0.0:
            continue
        correlation = float(np.dot(left, right) / denominator)
        candidate = (correlation, -abs(lag), -lag)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else -best[2]


def recovery_frames(
    error_mm,
    context: int,
    masked: int,
    recovery: int,
    margin_mm: float = 0.5,
    stable: int = 3,
) -> int:
    """First stable recovery offset, or the recovery length if unresolved."""
    values = np.asarray(error_mm, dtype=np.float64)
    counts = (context, masked, recovery, stable)
    if values.ndim != 1 or any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in counts):
        raise ValueError("error must be one-dimensional and phase lengths must be integers")
    if context <= 0 or masked < 0 or recovery <= 0 or stable <= 0 or stable > recovery:
        raise ValueError("invalid recovery phase lengths")
    if len(values) != context + masked + recovery or not np.isfinite(values).all():
        raise ValueError("error length must match finite context/masked/recovery phases")
    margin_mm = float(margin_mm)
    if not math.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError("margin_mm must be non-negative and finite")

    threshold = float(np.median(values[:context]) + margin_mm)
    phase = values[context + masked :]
    for offset in range(recovery - stable + 1):
        if np.all(phase[offset : offset + stable] <= threshold):
            return offset
    return recovery


def temporal_motion_metrics(sample_ids, gt_xyz, pred_xyz, fps: float, raw_frame_step: int) -> dict:
    """Root-relative motion errors, resetting at every sequence or frame gap."""
    gt = np.asarray(gt_xyz, dtype=np.float64)
    pred = np.asarray(pred_xyz, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[1:] != (21, 3):
        raise ValueError(f"expected matching [N, 21, 3] arrays, got gt={gt.shape} pred={pred.shape}")
    if len(sample_ids) != len(gt):
        raise ValueError(f"sample/order length mismatch: samples={len(sample_ids)} xyz={len(gt)}")
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("motion inputs must contain only finite values")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")

    gt = gt - gt[:, :1]
    pred = pred - pred[:, :1]
    segments = _segments(sample_ids, raw_frame_step)
    velocity_errors = []
    acceleration_errors = []
    prediction_accelerations = []
    for rows in segments:
        gt_velocity = np.diff(gt[rows], axis=0) * fps
        pred_velocity = np.diff(pred[rows], axis=0) * fps
        if len(gt_velocity):
            velocity_errors.append(np.linalg.norm(pred_velocity - gt_velocity, axis=-1).reshape(-1))
        gt_acceleration = np.diff(gt_velocity, axis=0) * fps
        pred_acceleration = np.diff(pred_velocity, axis=0) * fps
        if len(gt_acceleration):
            acceleration_errors.append(np.linalg.norm(pred_acceleration - gt_acceleration, axis=-1).reshape(-1))
            prediction_accelerations.append(np.linalg.norm(pred_acceleration, axis=-1).reshape(-1))

    velocity = np.concatenate(velocity_errors) if velocity_errors else np.empty(0)
    acceleration = np.concatenate(acceleration_errors) if acceleration_errors else np.empty(0)
    pred_acceleration = np.concatenate(prediction_accelerations) if prediction_accelerations else np.empty(0)
    return {
        "velocity_error_mm_s": float(velocity.mean() * 1000.0) if velocity.size else None,
        "acceleration_error_mm_s2": float(acceleration.mean() * 1000.0) if acceleration.size else None,
        "prediction_acceleration_mm_s2": float(pred_acceleration.mean() * 1000.0) if pred_acceleration.size else None,
        "jitter_mm_s2": float(pred_acceleration.mean() * 1000.0) if pred_acceleration.size else None,
        "num_velocity_edges": int(sum(max(len(rows) - 1, 0) for rows in segments)),
        "num_acceleration_triplets": int(sum(max(len(rows) - 2, 0) for rows in segments)),
    }


def _sample_ids(values) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _validate_order(run_meta: Path) -> list[str]:
    payload = read_json(run_meta)
    if not isinstance(payload, dict) or not isinstance(payload.get("sample_order"), list):
        raise ValueError(f"run_meta missing list sample_order: {run_meta}")
    order = _sample_ids(payload["sample_order"])
    if not order:
        raise ValueError("sample_order must not be empty")
    parsed = [_sequence_frame(sample_id) for sample_id in order]
    if len(set(parsed)) != len(parsed):
        raise ValueError("duplicate sample_id in run_meta")
    return order


def _xyz_array(values, name: str, rows: int, points: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if array.ndim != 3 or array.shape[0] != rows or array.shape[2] != 3:
        raise ValueError(f"{name} must be [{rows}, P, 3], got {array.shape}")
    if points is not None and array.shape[1] != points:
        raise ValueError(f"{name} must have {points} points, got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _load_ground_truth(gt_dir: Path, rows: int) -> tuple[np.ndarray, np.ndarray]:
    gt_xyz = _xyz_array(read_json(gt_dir / "evaluation_xyz.json"), "GT xyz", rows, 21)
    gt_verts = _xyz_array(read_json(gt_dir / "evaluation_verts.json"), "GT verts", rows)
    return gt_xyz, gt_verts


def _assert_id_order(have, wanted: list[str], source: Path) -> None:
    ids = _sample_ids(np.asarray(have).tolist())
    if ids != wanted:
        raise ValueError(f"sample_id order mismatch for {source}")


def _load_rope_closure(path: Path, order: list[str]) -> float | None:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"sample_id", "base_rope_residual", "refined_rope_residual", "rope_valid"}
        missing = required - set(loaded.files)
        if missing:
            raise ValueError(f"{path} missing keys: {sorted(missing)}")
        _assert_id_order(loaded["sample_id"], order, path)
        base = np.asarray(loaded["base_rope_residual"], dtype=np.float64)
        refined = np.asarray(loaded["refined_rope_residual"], dtype=np.float64)
        valid = np.asarray(loaded["rope_valid"])
    expected = (len(order), 5)
    if base.shape != expected or refined.shape != expected or valid.shape != expected or valid.dtype != np.bool_:
        raise ValueError(f"rope residual arrays must all be {expected} with boolean rope_valid")
    if not np.isfinite(base[valid]).all() or not np.isfinite(refined[valid]).all():
        raise ValueError("valid rope residual entries must be finite")
    summary = summarize_rope_residuals(base, refined, valid)
    closure = float(summary["closure_frac"])
    return closure if math.isfinite(closure) else None


def _load_timing(path: Path, rows: int) -> dict:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be an object: {path}")
    num_samples = payload.get("num_samples")
    if type(num_samples) is not int or num_samples != rows:
        raise ValueError(f"summary num_samples must be integer {rows}: {path}")
    timing = payload.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"summary missing timing object: {path}")
    out = {}
    for key in ("wall_seconds", "per_sample_ms"):
        raw = timing.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"summary timing.{key} must be numeric: {path}")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"summary timing.{key} must be non-negative and finite: {path}")
        out[key] = value
    return out


def _load_method(name: str, directory: Path, order: list[str], gt_verts: np.ndarray) -> dict:
    if not directory.is_dir():
        raise FileNotFoundError(f"method directory does not exist: {directory}")
    xyz_rows, vert_rows = load_pred_json(directory / "pred.json")
    xyz = _xyz_array(xyz_rows, f"{name} xyz", len(order), 21)
    verts = _xyz_array(vert_rows, f"{name} verts", len(order), gt_verts.shape[1])

    sample_path = directory / "sample_id.npy"
    if sample_path.exists():
        _assert_id_order(np.load(sample_path, allow_pickle=False), order, sample_path)

    residual_path = directory / "rope_residuals.npz"
    summary_path = directory / "summary.json"
    if name != "base" and (not residual_path.exists() or not summary_path.exists()):
        raise FileNotFoundError(f"method {name} requires rope_residuals.npz and summary.json")
    closure = _load_rope_closure(residual_path, order) if residual_path.exists() else None
    timing = _load_timing(summary_path, len(order)) if summary_path.exists() else None
    return {"xyz": xyz, "verts": verts, "rope_closure": closure, "timing": timing}


def _per_frame_pa_mpvpe(gt_verts: np.ndarray, pred_verts: np.ndarray) -> np.ndarray:
    values = np.empty(len(gt_verts), dtype=np.float64)
    for row in range(len(gt_verts)):
        values[row] = point_distances(gt_verts[row], align_w_scale(gt_verts[row], pred_verts[row])).mean() * 1000.0
    return values


def _lag_metrics(sample_ids, gt_xyz: np.ndarray, pred_xyz: np.ndarray, fps: float, raw_frame_step: int) -> dict:
    gt = gt_xyz - gt_xyz[:, :1]
    pred = pred_xyz - pred_xyz[:, :1]
    lags = []
    excluded = 0
    eligible = 0
    for rows in _segments(sample_ids, raw_frame_step):
        if len(rows) < 31:
            continue
        eligible += 1
        gt_speed = np.linalg.norm(np.diff(gt[rows], axis=0) * fps, axis=-1).mean(axis=1)
        pred_speed = np.linalg.norm(np.diff(pred[rows], axis=0) * fps, axis=-1).mean(axis=1)
        lag = phase_lag(gt_speed, pred_speed, max_lag=15)
        if lag is None:
            excluded += 1
        else:
            lags.append(lag)
    return {
        "phase_lag_frames": float(np.mean(lags)) if lags else None,
        "num_lag_segments": len(lags),
        "num_lag_segments_excluded": excluded,
        "num_lag_segments_eligible": eligible,
    }


def _load_episode_manifest(path: Path, order: list[str], raw_frame_step: int) -> list[dict]:
    rows = list(read_jsonl(path))
    if len(rows) != len(order):
        raise ValueError(f"hard manifest rows ({len(rows)}) != samples ({len(order)})")
    manifest_ids = [str(row.get("sample_id")) for row in rows]
    if manifest_ids != order:
        raise ValueError("hard manifest sample_id order mismatch")
    required = {"episode_id", "episode_phase", "episode_offset", "segment_id"}
    allowed = {"context", "masked", "recovery", "tail"}
    episode_segments = {}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"hard manifest missing episode fields: {sorted(missing)}")
        if row["episode_phase"] not in allowed:
            raise ValueError(f"unsupported episode_phase: {row['episode_phase']}")
        if row["episode_phase"] != "tail" and row["episode_id"] is None:
            raise ValueError("non-tail episode row requires episode_id")
        if row["episode_phase"] != "tail":
            episode_id = str(row["episode_id"])
            segment_id = str(row["segment_id"])
            if episode_id in episode_segments and episode_segments[episode_id] != segment_id:
                raise ValueError("episode_id must be globally unique across segments")
            episode_segments[episode_id] = segment_id
        offset = row["episode_offset"]
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("episode_offset must be a non-negative integer")

    seen_segment_ids = set()
    for segment in _segments(order, raw_frame_step):
        ids = {str(rows[index]["segment_id"]) for index in segment}
        if len(ids) != 1:
            raise ValueError("segment_id changes inside a contiguous raw-frame segment")
        segment_id = next(iter(ids))
        if segment_id in seen_segment_ids:
            raise ValueError("segment_id cannot bridge a raw-frame gap")
        seen_segment_ids.add(segment_id)
    return rows


def _masked_occluded_tip_mask(manifest: list[dict]) -> np.ndarray | None:
    _, tips = finger_joint_map("ho3d")
    mask = np.zeros((len(manifest), 21), dtype=bool)
    for row_index, row in enumerate(manifest):
        if row["episode_phase"] != "masked":
            continue
        occluded = occluded_fingers_for_row(row, (640, 480))
        if occluded is None:
            return None
        for finger_index, is_occluded in enumerate(occluded):
            mask[row_index, tips[finger_index]] = is_occluded
    return mask if mask.any() else None


def _episode_phase_lengths(manifest: list[dict]) -> dict[str, int]:
    episodes: dict[tuple[str, str], list[dict]] = {}
    for row in manifest:
        if row["episode_phase"] != "tail":
            episodes.setdefault((str(row["segment_id"]), str(row["episode_id"])), []).append(row)
    layouts = set()
    for key, rows in episodes.items():
        ordered = sorted(rows, key=lambda row: row["episode_offset"])
        if [row["episode_offset"] for row in ordered] != list(range(len(ordered))):
            raise ValueError(f"episode {key} must use contiguous offsets")
        labels = [row["episode_phase"] for row in ordered]
        context = next((index for index, label in enumerate(labels) if label != "context"), len(labels))
        masked_end = next((index for index in range(context, len(labels)) if labels[index] != "masked"), len(labels))
        recovery_end = next((index for index in range(masked_end, len(labels)) if labels[index] != "recovery"), len(labels))
        if context == 0 or masked_end == context or recovery_end != len(labels):
            raise ValueError(f"episode {key} must use contiguous context/masked/recovery phases")
        layouts.add((context, masked_end - context, recovery_end - masked_end))
    if len(layouts) != 1:
        raise ValueError("all complete episodes must use one common phase layout")
    context, masked, recovery = layouts.pop()
    return {"context": context, "masked": masked, "recovery": recovery}


def _phase_metrics(
    frame_pa: np.ndarray,
    manifest: list[dict],
    sample_ids,
    raw_frame_step: int,
) -> dict:
    out = {}
    layout = _episode_phase_lengths(manifest)
    phases = np.asarray([row["episode_phase"] for row in manifest])
    for phase in ("context", "masked", "recovery"):
        values = frame_pa[phases == phase]
        out[f"{phase}_pa_mpjpe_mm"] = float(values.mean()) if values.size else None
    offsets = np.asarray([row["episode_offset"] for row in manifest])
    for horizon in MASKED_HORIZONS:
        values = frame_pa[(phases == "masked") & (offsets == layout["context"] - 1 + horizon)]
        out[f"masked_h{horizon}_pa_mpjpe_mm"] = float(values.mean()) if values.size else None

    episodes: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(manifest):
        if row["episode_phase"] != "tail":
            episodes.setdefault((str(row["segment_id"]), str(row["episode_id"])), []).append(index)
    recoveries = []
    for key, indices in episodes.items():
        indices = sorted(indices, key=lambda index: manifest[index]["episode_offset"])
        offsets = [manifest[index]["episode_offset"] for index in indices]
        labels = [manifest[index]["episode_phase"] for index in indices]
        expected = (["context"] * layout["context"] + ["masked"] * layout["masked"]
                    + ["recovery"] * layout["recovery"])
        if offsets != list(range(len(expected))) or labels != expected:
            raise ValueError(f"episode {key} does not match the common phase layout")
        parsed = [_sequence_frame(sample_ids[index]) for index in indices]
        if len({sequence for sequence, _ in parsed}) != 1 or any(
            parsed[index][1] - parsed[index - 1][1] != raw_frame_step
            for index in range(1, len(parsed))
        ):
            raise ValueError(f"episode {key} crosses a sequence or raw-frame gap")
        if layout["recovery"]:
            recoveries.append(
                recovery_frames(
                    frame_pa[indices],
                    context=layout["context"],
                    masked=layout["masked"],
                    recovery=layout["recovery"],
                    margin_mm=0.5,
                    stable=min(3, layout["recovery"]),
                )
            )
    out["recovery_frames"] = float(np.mean(recoveries)) if recoveries else None
    out["num_recovery_episodes"] = len(recoveries)
    return out


def _masked_motion_metrics(
    sample_ids,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    manifest: list[dict],
    fps: float,
    raw_frame_step: int,
) -> dict:
    selected = np.flatnonzero([row["episode_phase"] == "masked" for row in manifest])
    motion = temporal_motion_metrics(
        np.asarray(sample_ids)[selected], gt_xyz[selected], pred_xyz[selected], fps, raw_frame_step
    )
    return {
        "masked_velocity_error_mm_s": motion["velocity_error_mm_s"],
        "masked_acceleration_error_mm_s2": motion["acceleration_error_mm_s2"],
        "masked_prediction_acceleration_mm_s2": motion["prediction_acceleration_mm_s2"],
        "masked_jitter_mm_s2": motion["jitter_mm_s2"],
        "num_masked_velocity_edges": motion["num_velocity_edges"],
        "num_masked_acceleration_triplets": motion["num_acceleration_triplets"],
    }


def _sequence_rows(sample_ids) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for row, sample_id in enumerate(sample_ids):
        grouped.setdefault(_sequence_frame(sample_id)[0], []).append(row)
    return {key: np.asarray(value, dtype=np.int64) for key, value in sorted(grouped.items())}


def _sequence_metrics(
    sample_ids,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    frame_pa: np.ndarray,
    frame_pv: np.ndarray,
    joint_pa: np.ndarray,
    occluded_tip_mask: np.ndarray | None,
    fps: float,
    raw_frame_step: int,
    manifest: list[dict] | None,
) -> dict[str, dict[str, tuple[float, int]]]:
    out = {metric: {} for metric in BOOTSTRAP_METRICS}
    phases = None if manifest is None else np.asarray([row["episode_phase"] for row in manifest])
    layout = None if manifest is None else _episode_phase_lengths(manifest)
    ids = np.asarray(sample_ids)
    for sequence, rows in _sequence_rows(sample_ids).items():
        out["pa_mpjpe_mm"][sequence] = (float(frame_pa[rows].mean()), len(rows))
        out["pa_mpvpe_mm"][sequence] = (float(frame_pv[rows].mean()), len(rows))
        motion = temporal_motion_metrics(ids[rows], gt_xyz[rows], pred_xyz[rows], fps, raw_frame_step)
        for metric in ("velocity_error_mm_s", "acceleration_error_mm_s2"):
            if motion[metric] is not None:
                count_key = "num_velocity_edges" if metric == "velocity_error_mm_s" else "num_acceleration_triplets"
                out[metric][sequence] = (motion[metric], motion[count_key])
        if phases is not None:
            for phase in ("masked", "recovery"):
                selected = rows[phases[rows] == phase]
                if selected.size:
                    out[f"{phase}_pa_mpjpe_mm"][sequence] = (float(frame_pa[selected].mean()), len(selected))
            if occluded_tip_mask is not None:
                tip_mask = occluded_tip_mask[rows]
                if tip_mask.any():
                    values = joint_pa[rows][tip_mask]
                    out["masked_occluded_tip_pa_mpjpe_mm"][sequence] = (float(values.mean()), len(values))
            offsets = np.asarray([manifest[row]["episode_offset"] for row in rows])
            for horizon in MASKED_HORIZONS:
                selected = rows[(phases[rows] == "masked") & (offsets == layout["context"] - 1 + horizon)]
                if selected.size:
                    out[f"masked_h{horizon}_pa_mpjpe_mm"][sequence] = (
                        float(frame_pa[selected].mean()), len(selected)
                    )
            selected = rows[phases[rows] == "masked"]
            if selected.size:
                masked_motion = temporal_motion_metrics(
                    ids[selected], gt_xyz[selected], pred_xyz[selected], fps, raw_frame_step
                )
                for metric, count_key in (
                    ("masked_velocity_error_mm_s", "num_velocity_edges"),
                    ("masked_acceleration_error_mm_s2", "num_acceleration_triplets"),
                    ("masked_jitter_mm_s2", "num_acceleration_triplets"),
                ):
                    source = metric.removeprefix("masked_")
                    if masked_motion[source] is not None:
                        out[metric][sequence] = (masked_motion[source], masked_motion[count_key])
    return out


def _value_weight(entry) -> tuple[float, float]:
    if isinstance(entry, (tuple, list)):
        if len(entry) != 2:
            raise ValueError("sequence metric tuple must be (value, weight)")
        value, weight = float(entry[0]), float(entry[1])
    else:
        value, weight = float(entry), 1.0
    if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("sequence metric values must be finite with positive weights")
    return value, weight


def sequence_bootstrap_ci(
    sequence_keys,
    sequence_metrics: dict[str, dict[str, dict[str, float]]],
    reference: str,
    iterations: int = 2000,
    seed: int = 20260710,
) -> dict:
    keys = list(sequence_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("sequence_keys must be non-empty and unique")
    if reference not in sequence_metrics:
        raise ValueError(f"reference method not found: {reference}")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(iterations, len(keys)))
    report = {}
    for method, metrics in sequence_metrics.items():
        report[method] = {}
        for metric in BOOTSTRAP_METRICS:
            reference_values = sequence_metrics[reference].get(metric, {})
            method_values = metrics.get(metric, {})
            paired = np.full(len(keys), np.nan, dtype=np.float64)
            weights = np.zeros(len(keys), dtype=np.float64)
            for index, key in enumerate(keys):
                if key not in method_values or key not in reference_values:
                    continue
                method_value, method_weight = _value_weight(method_values[key])
                reference_value, reference_weight = _value_weight(reference_values[key])
                if method_weight != reference_weight:
                    raise ValueError(f"sequence metric weights differ for {metric}/{key}")
                paired[index] = method_value - reference_value
                weights[index] = method_weight
            if not np.isfinite(paired).any():
                continue
            sampled = paired[draws]
            sampled_weights = weights[draws]
            sampled_weights[~np.isfinite(sampled)] = 0.0
            counts = sampled_weights.sum(axis=1)
            estimates = np.divide(
                np.nansum(sampled * sampled_weights, axis=1),
                counts,
                out=np.full(iterations, np.nan),
                where=counts > 0,
            )
            finite = estimates[np.isfinite(estimates)]
            if finite.size:
                report[method][metric] = [float(value) for value in np.percentile(finite, [2.5, 97.5])]
    return report


def _parse_methods(specs: list[str]) -> dict[str, Path]:
    methods = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"method must be NAME=DIR: {spec}")
        name, directory = spec.split("=", 1)
        name = name.strip()
        if not name or name in methods:
            raise ValueError(f"method names must be non-empty and unique: {name!r}")
        methods[name] = Path(directory)
    return methods


def build_report(
    method_dirs: dict[str, Path],
    reference: str,
    gt_dir: Path,
    run_meta: Path,
    hard_manifest: Path | None,
    fps: float,
    raw_frame_step: int,
) -> dict:
    if reference not in method_dirs:
        raise ValueError(f"reference method not found: {reference}")
    order = _validate_order(run_meta)
    gt_xyz, gt_verts = _load_ground_truth(gt_dir, len(order))
    manifest = _load_episode_manifest(hard_manifest, order, raw_frame_step) if hard_manifest is not None else None
    occluded_tip_mask = _masked_occluded_tip_mask(manifest) if manifest is not None else None
    methods = {}
    sequence_values = {}
    rope_closure = {}
    timing = {}
    for name, directory in method_dirs.items():
        data = _load_method(name, directory, order, gt_verts)
        joint_pa = per_joint_pa_distances(gt_xyz, data["xyz"]) * 1000.0
        frame_pa = joint_pa.mean(axis=1)
        frame_pv = _per_frame_pa_mpvpe(gt_verts, data["verts"])
        metrics = {
            "num_samples": len(order),
            "pa_mpjpe_mm": float(frame_pa.mean()),
            "pa_mpvpe_mm": float(frame_pv.mean()),
            **temporal_motion_metrics(order, gt_xyz, data["xyz"], fps, raw_frame_step),
            **_lag_metrics(order, gt_xyz, data["xyz"], fps, raw_frame_step),
        }
        if manifest is not None:
            metrics.update(_phase_metrics(frame_pa, manifest, order, raw_frame_step))
            metrics.update(_masked_motion_metrics(order, gt_xyz, data["xyz"], manifest, fps, raw_frame_step))
            metrics["masked_occluded_tip_pa_mpjpe_mm"] = (
                None if occluded_tip_mask is None else float(joint_pa[occluded_tip_mask].mean())
            )
        methods[name] = metrics
        sequence_values[name] = _sequence_metrics(
            order,
            gt_xyz,
            data["xyz"],
            frame_pa,
            frame_pv,
            joint_pa,
            occluded_tip_mask,
            fps,
            raw_frame_step,
            manifest,
        )
        rope_closure[name] = data["rope_closure"]
        timing[name] = data["timing"]

    paired_deltas = {}
    for name, metrics in methods.items():
        paired_deltas[name] = {
            metric: metrics[metric] - methods[reference][metric]
            for metric in BOOTSTRAP_METRICS
            if metrics.get(metric) is not None and methods[reference].get(metric) is not None
        }
    sequence_keys = list(_sequence_rows(order))
    return {
        "methods": methods,
        "reference": reference,
        "paired_deltas": paired_deltas,
        "bootstrap_ci": sequence_bootstrap_ci(sequence_keys, sequence_values, reference),
        "rope_closure": rope_closure,
        "timing": timing,
        "protocol": {
            "fps": float(fps),
            "raw_frame_step": int(raw_frame_step),
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260710,
            "num_sequences": len(sequence_keys),
            "episode_phases": _episode_phase_lengths(manifest) if manifest is not None else None,
            "masked_occluded_tip_defined": occluded_tip_mask is not None,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gap-safe temporal prediction scorer.")
    parser.add_argument("--method", action="append", required=True, help="Repeated NAME=DIR method specification.")
    parser.add_argument("--reference", required=True, help="Reference method name for paired deltas.")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--raw-frame-step", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    report = build_report(
        _parse_methods(args.method),
        args.reference,
        args.gt_dir,
        args.run_meta,
        args.hard_manifest,
        args.fps,
        args.raw_frame_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_sanitize(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Temporal scores written to: {args.output}")
    return args.output


if __name__ == "__main__":
    main()
