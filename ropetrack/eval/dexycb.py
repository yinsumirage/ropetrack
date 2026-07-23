#!/usr/bin/env python3
"""Score aligned DexYCB predictions with protocol diagnostics and paired CIs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("pa_joint_mm", "root_relative_joint_mm", "camera_joint_mm", "root_translation_mm", "global_orientation_deg")


def read_manifest(root: Path) -> tuple[list[dict], str]:
    for split_file in ("evaluation", "training"):
        path = root / f"{split_file}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()], split_file
    raise FileNotFoundError(f"manifest missing under {root}")


def read_predictions(pred_path: Path, order_path: Path) -> tuple[list[str], np.ndarray]:
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    if len(payload) != 2:
        raise ValueError(f"prediction payload must be [xyz, vertices]: {pred_path}")
    xyz = np.asarray(payload[0], dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[1:] != (21, 3):
        raise ValueError(f"prediction joints must be [N,21,3], got {xyz.shape}")
    if order_path.suffix == ".npy":
        order = np.load(order_path).astype(str).tolist()
    else:
        order_payload = json.loads(order_path.read_text(encoding="utf-8"))
        order = list(map(str, order_payload.get("sample_order", order_payload)))
    if len(order) != len(xyz) or len(order) != len(set(order)):
        raise ValueError(f"prediction order invalid: order={len(order)} xyz={len(xyz)} unique={len(set(order))}")
    return order, xyz


def align_rows(target_ids: list[str], source_ids: list[str], values: np.ndarray) -> np.ndarray:
    by_id = {sample_id: index for index, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in target_ids if sample_id not in by_id]
    extras = sorted(set(source_ids) - set(target_ids))
    if missing or extras:
        raise ValueError(f"sample alignment mismatch: missing={missing[:5]} extras={extras[:5]}")
    return values[np.asarray([by_id[sample_id] for sample_id in target_ids])]


def procrustes(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean = pred.mean(axis=0)
    target_mean = target.mean(axis=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_norm = np.linalg.norm(pred_centered) + 1e-12
    target_norm = np.linalg.norm(target_centered) + 1e-12
    source = pred_centered / pred_norm
    destination = target_centered / target_norm
    u, singular, vt = np.linalg.svd(destination.T @ source, full_matrices=False)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    scale = float(singular.sum()) * target_norm
    return source @ rotation.T * scale + target_mean


def procrustes_batch(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_mean = pred.mean(axis=1, keepdims=True)
    target_mean = target.mean(axis=1, keepdims=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_norm = np.linalg.norm(pred_centered, axis=(1, 2), keepdims=True) + 1e-12
    target_norm = np.linalg.norm(target_centered, axis=(1, 2), keepdims=True) + 1e-12
    source = pred_centered / pred_norm
    destination = target_centered / target_norm
    covariance = np.einsum("bji,bjk->bik", destination, source)
    u, singular, vt = np.linalg.svd(covariance, full_matrices=False)
    rotation = np.einsum("bij,bjk->bik", u, vt)
    reflected = np.linalg.det(rotation) < 0
    vt[reflected, -1] *= -1
    singular[reflected, -1] *= -1
    rotation = np.einsum("bij,bjk->bik", u, vt)
    scale = singular.sum(axis=1)[:, None, None] * target_norm
    return np.einsum("bji,bki->bjk", source, rotation) * scale + target_mean


def axis_angle_to_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    angles = np.linalg.norm(vectors, axis=1)
    matrices = np.repeat(np.eye(3)[None], len(vectors), axis=0)
    nonzero = angles > 1e-12
    axes = np.zeros_like(vectors)
    axes[nonzero] = vectors[nonzero] / angles[nonzero, None]
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
    skew = np.stack((
        np.stack((np.zeros(len(vectors)), -z, y), axis=1),
        np.stack((z, np.zeros(len(vectors)), -x), axis=1),
        np.stack((-y, x, np.zeros(len(vectors))), axis=1),
    ), axis=1)
    sine = np.sin(angles)[:, None, None]
    cosine = np.cos(angles)[:, None, None]
    matrices = matrices + sine * skew + (1.0 - cosine) * np.einsum("bij,bjk->bik", skew, skew)
    return matrices


def orientation_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_r = axis_angle_to_matrix(predicted)
    target_r = axis_angle_to_matrix(target)
    relative = np.einsum("bij,bkj->bik", pred_r, target_r)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def load_mano_translation_diagnostic(path: Path, target_ids: list[str], target_pose_path: Path) -> np.ndarray:
    with np.load(path) as cache:
        ids = cache["sample_id"].astype(str).tolist()
        cam_t = align_rows(target_ids, ids, np.asarray(cache["base_cam_t"], dtype=np.float32))
    with np.load(target_pose_path) as target:
        target_pose = align_rows(target_ids, target["sample_id"].astype(str).tolist(), np.asarray(target["pose_m"], dtype=np.float32))
    return np.linalg.norm(cam_t - target_pose[:, 48:51], axis=1) * 1000.0


def palm_frames(joints: np.ndarray) -> np.ndarray:
    value = np.asarray(joints, dtype=np.float64)
    x_axis = value[:, 5] - value[:, 17]
    y_hint = value[:, 9] - value[:, 0]
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True).clip(min=1e-12)
    z_axis = np.cross(x_axis, y_hint)
    z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True).clip(min=1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True).clip(min=1e-12)
    return np.stack((x_axis, y_axis, z_axis), axis=2)


def palm_orientation_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_frame = palm_frames(predicted)
    target_frame = palm_frames(target)
    relative = np.einsum("bij,bkj->bik", pred_frame, target_frame)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def sample_metrics(gt: np.ndarray, pred: np.ndarray, mano_translation: np.ndarray) -> dict[str, np.ndarray]:
    pa = np.linalg.norm(procrustes_batch(pred, gt) - gt, axis=2).mean(axis=1) * 1000.0
    root_gt = gt - gt[:, :1]
    root_pred = pred - pred[:, :1]
    return {
        "pa_joint_mm": pa,
        "root_relative_joint_mm": np.linalg.norm(root_pred - root_gt, axis=2).mean(axis=1) * 1000.0,
        "camera_joint_mm": np.linalg.norm(pred - gt, axis=2).mean(axis=1) * 1000.0,
        "root_translation_mm": np.linalg.norm(pred[:, 0] - gt[:, 0], axis=1) * 1000.0,
        "global_orientation_deg": palm_orientation_error_deg(pred, gt),
        "mano_translation_parameter_mm": mano_translation,
    }


def summarize(metrics: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict:
    if indices is None:
        indices = np.arange(len(next(iter(metrics.values()))))
    return {
        "count": int(len(indices)),
        **{name: float(np.mean(values[indices])) if len(indices) else None for name, values in metrics.items()},
    }


def grouped_summary(metrics: dict[str, np.ndarray], labels: list[str]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    return {label: summarize(metrics, np.asarray(indices)) for label, indices in sorted(groups.items())}


def bootstrap_delta(
    candidate: np.ndarray,
    reference: np.ndarray,
    episodes: list[str],
    iterations: int = 2000,
    seed: int = 20260720,
) -> list[float]:
    delta = np.asarray(candidate) - np.asarray(reference)
    episode_sums, episode_counts = [], []
    episode_array = np.asarray(episodes)
    for episode in sorted(set(episodes)):
        indices = np.flatnonzero(episode_array == episode)
        episode_sums.append(float(delta[indices].sum()))
        episode_counts.append(len(indices))
    sums = np.asarray(episode_sums)
    counts = np.asarray(episode_counts)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(sums), size=(iterations, len(sums)))
    deltas = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return np.percentile(deltas, [2.5, 97.5]).tolist()


def visibility_labels(rows: list[dict], thresholds: tuple[float, float]) -> list[str]:
    low, high = thresholds
    result = []
    for row in rows:
        value = float(row["hand_segmentation_pixels"])
        result.append("low_visible" if value <= low else "high_visible" if value >= high else "mid_visible")
    return result


def parse_prediction_args(values: list[list[str]]) -> list[tuple[str, Path, Path, Path]]:
    result = []
    names = set()
    for name, pred, order, mano_cache in values:
        if name in names:
            raise ValueError(f"duplicate method name: {name}")
        names.add(name)
        result.append((name, Path(pred), Path(order), Path(mano_cache)))
    return result


def score(args) -> Path:
    rows, split_file = read_manifest(args.gt_root)
    target_ids = [row["sample_id"] for row in rows]
    gt = np.asarray(json.loads((args.gt_root / f"{split_file}_xyz.json").read_text(encoding="utf-8")), dtype=np.float32)
    if gt.shape != (len(rows), 21, 3):
        raise ValueError(f"GT shape mismatch: {gt.shape} rows={len(rows)}")

    if args.protocol_split == "test":
        freeze = json.loads(args.test_freeze_file.read_text(encoding="utf-8")) if args.test_freeze_file else {}
        if freeze.get("status") != "frozen":
            raise PermissionError("official test scoring requires frozen recipe")
        access_log = args.output_root / "test_score_access.json"
        if access_log.exists() or (args.output_root / "scores.json").exists():
            raise PermissionError("official DexYCB test scoring has already been attempted")
        args.output_root.mkdir(parents=True, exist_ok=True)
        access_log.write_text(json.dumps({
            "status": "started",
            "policy": "one-shot official S1 test scoring after recipe freeze",
            "freeze_file": str(args.test_freeze_file),
            "sample_count": len(rows),
        }, indent=2) + "\n", encoding="utf-8")

    if args.visibility_thresholds:
        visibility_payload = json.loads(args.visibility_thresholds.read_text(encoding="utf-8"))
        visibility_thresholds = tuple(map(float, visibility_payload["visible_hand_pixels_terciles"]))
        visibility_source = str(args.visibility_thresholds)
    else:
        pixels = np.asarray([row["hand_segmentation_pixels"] for row in rows])
        visibility_thresholds = tuple(np.percentile(pixels, [33.333, 66.667]).tolist())
        visibility_source = f"{args.protocol_split} manifest"

    pose_path = args.gt_root / f"{split_file}_mano.npz"
    method_metrics = {}
    for name, pred_path, order_path, mano_cache in parse_prediction_args(args.prediction):
        source_ids, prediction = read_predictions(pred_path, order_path)
        prediction = align_rows(target_ids, source_ids, prediction)
        mano_translation = load_mano_translation_diagnostic(mano_cache, target_ids, pose_path)
        method_metrics[name] = sample_metrics(gt, prediction, mano_translation)

    visibility = visibility_labels(rows, visibility_thresholds)
    report = {
        "dataset": "DexYCB",
        "protocol": "official S1 unseen-subject",
        "protocol_split": args.protocol_split,
        "sample_count": len(rows),
        "sample_id_sha256": __import__("hashlib").sha256("\n".join(target_ids).encode()).hexdigest(),
        "metrics": {name: summarize(metrics) for name, metrics in method_metrics.items()},
        "per_subject": {name: grouped_summary(metrics, [row["subject_id"] for row in rows]) for name, metrics in method_metrics.items()},
        "per_camera_serial": {name: grouped_summary(metrics, [row["camera_serial"] for row in rows]) for name, metrics in method_metrics.items()},
        "per_visibility_proxy": {name: grouped_summary(metrics, visibility) for name, metrics in method_metrics.items()},
        "visibility_proxy": {
            "measure": "official seg==255 visible hand pixels; lower is treated as more occluded",
            "tercile_thresholds": list(visibility_thresholds),
            "threshold_source": visibility_source,
        },
        "metric_definitions": {
            "global_orientation_deg": "geodesic angle between camera-frame palm bases built from wrist and index/middle/little MCP joints; defined identically for official left and right hands",
            "mano_translation_parameter_mm": "diagnostic distance between WiLoR cam_t and official pose_m translation; not used for checkpoint selection",
        },
        "signed_deltas": {},
        "bootstrap_95ci": {},
        "bootstrap": {"unit": "subject/sequence episode", "iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed},
    }
    pairs = []
    if {"wilor_base", "rgb_only"} <= set(method_metrics):
        pairs.append(("rgb_only_minus_wilor_base", "rgb_only", "wilor_base"))
    if {"rgb_only", "rgb_rope"} <= set(method_metrics):
        pairs.append(("rgb_rope_minus_rgb_only", "rgb_rope", "rgb_only"))
    episodes = [row["episode_id"] for row in rows]
    for label, candidate, reference in pairs:
        report["signed_deltas"][label] = {}
        report["bootstrap_95ci"][label] = {}
        for metric in METRICS:
            cand = method_metrics[candidate][metric]
            ref = method_metrics[reference][metric]
            report["signed_deltas"][label][metric] = float(np.mean(cand - ref))
            report["bootstrap_95ci"][label][metric] = bootstrap_delta(
                cand, ref, episodes, args.bootstrap_iterations, args.bootstrap_seed
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "scores.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "visibility_thresholds.json").write_text(json.dumps({
        "visible_hand_pixels_terciles": list(visibility_thresholds),
        "source": visibility_source,
    }, indent=2) + "\n", encoding="utf-8")
    lines = ["| Method | PA mm | Root-relative mm | Camera mm | Root mm | Global orientation deg |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in report["metrics"].items():
        lines.append(
            f"| {name} | {values['pa_joint_mm']:.3f} | {values['root_relative_joint_mm']:.3f} | "
            f"{values['camera_joint_mm']:.3f} | {values['root_translation_mm']:.3f} | {values['global_orientation_deg']:.3f} |"
        )
    (args.output_root / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.protocol_split == "test":
        access_log = args.output_root / "test_score_access.json"
        payload = json.loads(access_log.read_text(encoding="utf-8"))
        payload["status"] = "completed"
        payload["score_json"] = str(output)
        access_log.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": report["metrics"], "signed_deltas": report["signed_deltas"]}, indent=2))
    return output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction", nargs=4, action="append", required=True, metavar=("NAME", "PRED_JSON", "ORDER", "MANO_CACHE"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-split", choices=("internal_val", "val", "test"), required=True)
    parser.add_argument("--visibility-thresholds", type=Path)
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    return parser.parse_args(argv)


def main(argv=None):
    return score(parse_args(argv))


if __name__ == "__main__":
    main()
