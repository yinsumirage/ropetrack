#!/usr/bin/env python3
"""Score aligned InterHand one-view single-hand predictions with frame CIs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.score_dexycb import palm_orientation_error_deg, procrustes
from scripts.validate_interhand26m_coordinates import load_mano_layers


METRICS = (
    "pa_joint_mm", "root_relative_joint_mm", "camera_joint_mm", "mpvpe_mm",
    "root_translation_mm", "global_orientation_deg",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ground_truth(root: Path) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray], str]:
    split_file = "training" if (root / "training.jsonl").is_file() else "evaluation"
    rows = [json.loads(line) for line in (root / f"{split_file}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    xyz = np.asarray(json.loads((root / f"{split_file}_xyz.json").read_text(encoding="utf-8")), dtype=np.float32)
    with np.load(root / f"{split_file}_mano.npz") as loaded:
        mano = {key: np.asarray(loaded[key]) for key in loaded.files}
    ids = [row["sample_id"] for row in rows]
    if xyz.shape != (len(rows), 21, 3) or mano["sample_id"].astype(str).tolist() != ids:
        raise ValueError("InterHand GT manifest/xyz/MANO order differs")
    return rows, xyz, mano, split_file


def read_order(path: Path) -> list[str]:
    if path.suffix == ".npy":
        return np.load(path).astype(str).tolist()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(map(str, payload.get("sample_order", payload)))


def align(target_ids: list[str], source_ids: list[str], values: np.ndarray) -> np.ndarray:
    by_id = {sample_id: index for index, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in target_ids if sample_id not in by_id]
    extras = sorted(set(source_ids) - set(target_ids))
    if missing or extras or len(by_id) != len(source_ids):
        raise ValueError(f"prediction alignment differs: missing={missing[:5]} extras={extras[:5]}")
    return values[np.asarray([by_id[sample_id] for sample_id in target_ids])]


def read_prediction(path: Path, order_path: Path, target_ids: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 2:
        raise ValueError(f"prediction must be [xyz,vertices]: {path}")
    source_ids = read_order(order_path)
    xyz = np.asarray(payload[0], dtype=np.float32)
    if xyz.shape != (len(source_ids), 21, 3):
        raise ValueError(f"prediction joint shape differs: {xyz.shape}")
    vertices = None
    if payload[1] and all(value is not None for value in payload[1]):
        vertices = np.asarray(payload[1], dtype=np.float32)
        if vertices.shape != (len(source_ids), 778, 3):
            raise ValueError(f"prediction vertex shape differs: {vertices.shape}")
        vertices = align(target_ids, source_ids, vertices)
    xyz = align(target_ids, source_ids, xyz)
    del payload
    gc.collect()
    return xyz, vertices


def decode_gt_vertices(mano: dict[str, np.ndarray], model_root: Path, batch_size: int) -> np.ndarray:
    import torch

    valid = np.asarray(mano["mano_valid"], dtype=bool)
    result = np.full((len(valid), 778, 3), np.nan, dtype=np.float32)
    layers = load_mano_layers(model_root)
    for side, right in (("right", True), ("left", False)):
        indices = np.flatnonzero(valid & (np.asarray(mano["is_right"], dtype=bool) == right))
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            pose = np.asarray(mano["pose"][chosen], dtype=np.float32)
            shape = np.asarray(mano["shape"][chosen], dtype=np.float32)
            trans = np.asarray(mano["trans_world_m"][chosen], dtype=np.float32)
            with torch.no_grad():
                output = layers[side](
                    global_orient=torch.from_numpy(pose[:, :3]),
                    hand_pose=torch.from_numpy(pose[:, 3:]),
                    betas=torch.from_numpy(shape),
                    transl=torch.from_numpy(trans),
                )
            world = output.vertices.numpy().astype(np.float32)
            rotation = np.asarray(mano["camrot"][chosen], dtype=np.float32)
            position = np.asarray(mano["campos_world_mm"][chosen], dtype=np.float32) / 1000.0
            result[chosen] = np.einsum("bij,bkj->bki", rotation, world - position[:, None, :])
    return result


def sample_metrics(rows: list[dict], gt: np.ndarray, pred: np.ndarray, gt_vertices: np.ndarray, pred_vertices: np.ndarray | None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    count = len(rows)
    values = {name: np.full(count, np.nan, dtype=np.float64) for name in METRICS}
    per_joint = np.full((count, 21), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        valid = np.asarray(row["joint_valid"], dtype=bool)
        target, predicted = gt[index], pred[index]
        if int(valid.sum()) >= 3:
            aligned = procrustes(predicted[valid], target[valid])
            values["pa_joint_mm"][index] = np.linalg.norm(aligned - target[valid], axis=1).mean() * 1000.0
        target_root = target - target[:1]
        predicted_root = predicted - predicted[:1]
        distance = np.linalg.norm(predicted - target, axis=1) * 1000.0
        root_distance = np.linalg.norm(predicted_root - target_root, axis=1) * 1000.0
        per_joint[index, valid] = distance[valid]
        values["camera_joint_mm"][index] = distance[valid].mean()
        values["root_relative_joint_mm"][index] = root_distance[valid].mean()
        values["root_translation_mm"][index] = distance[0]
        palm_ids = np.asarray([0, 5, 9, 17])
        if valid[palm_ids].all():
            values["global_orientation_deg"][index] = palm_orientation_error_deg(predicted[None], target[None])[0]
        if bool(row["mano_valid"]) and pred_vertices is not None:
            values["mpvpe_mm"][index] = np.linalg.norm(pred_vertices[index] - gt_vertices[index], axis=1).mean() * 1000.0
    return values, per_joint


def summarize(metrics: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict:
    if indices is None:
        indices = np.arange(len(next(iter(metrics.values()))))
    result = {"count": int(len(indices)), "metric_counts": {}}
    for name, values in metrics.items():
        chosen = np.asarray(values)[indices]
        finite = chosen[np.isfinite(chosen)]
        result[name] = float(finite.mean()) if len(finite) else None
        result["metric_counts"][name] = int(len(finite))
    return result


def grouped_summary(metrics: dict[str, np.ndarray], labels) -> dict:
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    return {label: summarize(metrics, np.asarray(indices)) for label, indices in sorted(groups.items())}


def valid_count_bucket(value: int) -> str:
    if value <= 10:
        return "04_10"
    if value <= 15:
        return "11_15"
    if value <= 20:
        return "16_20"
    return "21"


def bbox_size(row: dict) -> float:
    box = row["bbox_xyxy"]
    return max(float(box[2] - box[0]), float(box[3] - box[1]))


def bbox_bucket(value: float, thresholds: tuple[float, float]) -> str:
    return "small" if value <= thresholds[0] else "large" if value >= thresholds[1] else "medium"


def bootstrap_delta(candidate: np.ndarray, reference: np.ndarray, groups: list[str], iterations: int, seed: int) -> list[float] | None:
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    finite = np.isfinite(candidate) & np.isfinite(reference)
    if not finite.any():
        return None
    delta = candidate - reference
    group_array = np.asarray(groups)
    sums, counts = [], []
    for group in sorted(set(groups)):
        indices = np.flatnonzero((group_array == group) & finite)
        if len(indices):
            sums.append(float(delta[indices].sum()))
            counts.append(len(indices))
    sums, counts = np.asarray(sums), np.asarray(counts)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(sums), size=(iterations, len(sums)))
    values = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return np.percentile(values, [2.5, 97.5]).tolist()


def bootstrap_mean(values: np.ndarray, groups: list[str], iterations: int, seed: int) -> list[float] | None:
    zeros = np.zeros_like(np.asarray(values), dtype=np.float64)
    return bootstrap_delta(values, zeros, groups, iterations, seed)


def relative_root_diagnostic(rows: list[dict], gt: np.ndarray, pred: np.ndarray) -> dict:
    groups = defaultdict(dict)
    for index, row in enumerate(rows):
        if int(row.get("paired_hand_count", 1)) == 2:
            groups[row["frame_group_id"]][row["mano_side"]] = index
    errors = []
    for sides in groups.values():
        if set(sides) != {"left", "right"}:
            continue
        left, right = sides["left"], sides["right"]
        target = gt[left, 0] - gt[right, 0]
        predicted = pred[left, 0] - pred[right, 0]
        errors.append(np.linalg.norm(predicted - target) * 1000.0)
    return {
        "status": "diagnostic_only; WiLoR crop translation is not promoted as trusted absolute two-hand root",
        "paired_frames": len(errors),
        "relative_root_position_error_mm": float(np.mean(errors)) if errors else None,
    }


def parse_predictions(values: list[list[str]]) -> list[tuple[str, Path, Path, Path]]:
    result, names = [], set()
    for name, pred, order, mano_cache in values:
        if name in names:
            raise ValueError(f"duplicate method: {name}")
        names.add(name)
        result.append((name, Path(pred), Path(order), Path(mano_cache)))
    return result


def score(args: argparse.Namespace) -> Path:
    rows, gt, mano, _ = read_ground_truth(args.gt_root)
    target_ids = [row["sample_id"] for row in rows]
    prediction_specs = parse_predictions(args.prediction)
    if args.protocol_split == "test":
        freeze = json.loads(args.test_freeze_file.read_text(encoding="utf-8")) if args.test_freeze_file else {}
        if freeze.get("status") != "frozen":
            raise PermissionError("InterHand test scoring requires a frozen recipe")
        args.output_root.mkdir(parents=True, exist_ok=True)
        access = args.output_root / "test_score_access.json"
        inputs = {
            "test_freeze": file_sha256(args.test_freeze_file),
            "gt_protocol": file_sha256(args.gt_root / "protocol.json"),
            "gt_manifest": file_sha256(args.gt_root / "evaluation.jsonl"),
            "predictions": {
                name: {
                    "prediction": file_sha256(pred),
                    "order": file_sha256(order),
                    "mano_cache": file_sha256(mano_cache),
                }
                for name, pred, order, mano_cache in prediction_specs
            },
        }
        if (args.output_root / "scores.json").exists():
            raise PermissionError("InterHand one-shot test score already exists")
        if access.exists():
            previous = json.loads(access.read_text(encoding="utf-8"))
            if previous.get("status") != "started" or previous.get("inputs") != inputs:
                raise PermissionError("test retry is allowed only after a crash with byte-identical frozen inputs")
            attempts = int(previous.get("attempts", 1)) + 1
        else:
            attempts = 1
        access.write_text(json.dumps({
            "status": "started", "freeze_file": str(args.test_freeze_file), "sample_count": len(rows),
            "attempts": attempts, "retry_policy": "crash-only with byte-identical frozen input hashes", "inputs": inputs,
        }, indent=2) + "\n", encoding="utf-8")
    if args.bucket_thresholds:
        threshold_payload = json.loads(args.bucket_thresholds.read_text(encoding="utf-8"))
        thresholds = tuple(map(float, threshold_payload["bbox_size_terciles_px"]))
        threshold_source = str(args.bucket_thresholds)
    else:
        thresholds = tuple(np.percentile([bbox_size(row) for row in rows], [33.333, 66.667]).tolist())
        threshold_source = f"{args.protocol_split} manifest"
    gt_vertices = decode_gt_vertices(mano, args.mano_root, args.batch_size)
    methods, per_joint, relative_root = {}, {}, {}
    for name, pred_path, order_path, _ in prediction_specs:
        predicted, vertices = read_prediction(pred_path, order_path, target_ids)
        methods[name], per_joint[name] = sample_metrics(rows, gt, predicted, gt_vertices, vertices)
        relative_root[name] = relative_root_diagnostic(rows, gt, predicted)
        del predicted, vertices
        gc.collect()
    report = {
        "dataset": "InterHand2.6M v1.0 30fps",
        "project_protocol": "interhand26m_v1_30fps_oneview_v1",
        "protocol_split": args.protocol_split,
        "sample_count": len(rows),
        "frame_group_count": len({row["frame_group_id"] for row in rows}),
        "sample_id_sha256": hashlib.sha256("\n".join(target_ids).encode()).hexdigest(),
        "metrics": {name: summarize(values) for name, values in methods.items()},
        "per_side": {name: grouped_summary(values, [row["mano_side"] for row in rows]) for name, values in methods.items()},
        "per_hand_type": {name: grouped_summary(values, ["interacting" if row["is_interacting"] else "single" for row in rows]) for name, values in methods.items()},
        "per_capture": {name: grouped_summary(values, [row["capture_id"] for row in rows]) for name, values in methods.items()},
        "per_camera": {name: grouped_summary(values, [row["camera_id"] for row in rows]) for name, values in methods.items()},
        "per_valid_joint_count": {name: grouped_summary(values, [valid_count_bucket(row["valid_joint_count"]) for row in rows]) for name, values in methods.items()},
        "per_projected_in_frame_count": {name: grouped_summary(values, [valid_count_bucket(row["projected_in_frame_joint_count"]) for row in rows]) for name, values in methods.items()},
        "per_bbox_size": {name: grouped_summary(values, [bbox_bucket(bbox_size(row), thresholds) for row in rows]) for name, values in methods.items()},
        "per_mano_valid": {name: grouped_summary(values, ["mano_valid" if row["mano_valid"] else "joint_only" for row in rows]) for name, values in methods.items()},
        "per_joint_camera_mm": {name: np.nanmean(values, axis=0).tolist() for name, values in per_joint.items()},
        "relative_root_position": relative_root,
        "metric_boundaries": {
            "pa": "similarity-aligned over each sample's valid official 3D joints",
            "root_relative": "wrist-relative camera coordinates over valid joints",
            "camera": "absolute camera coordinates; dominated by WiLoR crop translation",
            "mpvpe": "only native-NeuralAnnot-MANO-valid samples; side-specific GT mesh",
            "global_orientation": "palm-frame geometry diagnostic, separate from PA/root/translation",
            "rope": "GT-derived ideal five-rope input for rope methods; not no-GT RGB inference or physical sensor evidence",
            "visibility": "projected_in_frame is an image-boundary proxy; InterHand native occlusion visibility is not claimed",
        },
        "bucket_thresholds": {"bbox_size_terciles_px": list(thresholds), "source": threshold_source},
        "signed_deltas": {},
        "absolute_bootstrap_95ci": {},
        "bootstrap_95ci": {},
        "bootstrap": {"unit": "underlying frame_group_id; paired left/right stay together", "iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed},
    }
    pairs = []
    if "wilor_base" in methods:
        pairs.extend((f"{name}_minus_wilor_base", name, "wilor_base") for name in methods if name != "wilor_base" and name != "rgb_rope")
    if {"rgb_only", "rgb_rope"} <= set(methods):
        pairs.append(("rgb_rope_minus_rgb_only", "rgb_rope", "rgb_only"))
    groups = [row["frame_group_id"] for row in rows]
    for name, values in methods.items():
        report["absolute_bootstrap_95ci"][name] = {
            metric: bootstrap_mean(values[metric], groups, args.bootstrap_iterations, args.bootstrap_seed)
            for metric in METRICS
        }
    for label, candidate, reference in pairs:
        report["signed_deltas"][label], report["bootstrap_95ci"][label] = {}, {}
        for metric in METRICS:
            left, right = methods[candidate][metric], methods[reference][metric]
            finite = np.isfinite(left) & np.isfinite(right)
            report["signed_deltas"][label][metric] = float(np.mean(left[finite] - right[finite])) if finite.any() else None
            report["bootstrap_95ci"][label][metric] = bootstrap_delta(left, right, groups, args.bootstrap_iterations, args.bootstrap_seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "scores.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    thresholds_path = args.output_root / "bucket_thresholds.json"
    thresholds_path.write_text(json.dumps(report["bucket_thresholds"], indent=2) + "\n", encoding="utf-8")
    lines = ["| Method | PA mm | Root-relative mm | Camera mm | MPVPE mm | Root mm | Orientation deg |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in report["metrics"].items():
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
        lines.append(f"| {name} | {fmt(values['pa_joint_mm'])} | {fmt(values['root_relative_joint_mm'])} | {fmt(values['camera_joint_mm'])} | {fmt(values['mpvpe_mm'])} | {fmt(values['root_translation_mm'])} | {fmt(values['global_orientation_deg'])} |")
    (args.output_root / "scores.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.protocol_split == "test":
        access = args.output_root / "test_score_access.json"
        payload = json.loads(access.read_text(encoding="utf-8"))
        payload.update({"status": "completed", "score_json": str(output)})
        access.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": report["metrics"], "signed_deltas": report["signed_deltas"]}, indent=2))
    return output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction", nargs=4, action="append", required=True, metavar=("NAME", "PRED_JSON", "ORDER", "MANO_CACHE"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-split", choices=("internal_val", "val", "test"), required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--bucket-thresholds", type=Path)
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    return parser.parse_args(argv)


if __name__ == "__main__":
    score(parse_args())
