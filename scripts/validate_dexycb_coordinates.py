#!/usr/bin/env python3
"""Run independent DexYCB projection and official native MANO decode gates."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TIP_VERTEX_IDS = [745, 317, 444, 556, 673]
OPENPOSE_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
TEST_SUBJECTS = {"20201002-subject-08", "20201015-subject-09"}
THRESHOLDS = {
    "reprojection_median_px": 0.10,
    "reprojection_p95_px": 0.50,
    "reprojection_max_px": 2.00,
    "mano_camera_mean_mm": 0.10,
    "mano_camera_p95_mm": 0.50,
    "mano_camera_max_mm": 2.00,
    "mano_root_relative_mean_mm": 0.10,
    "mano_root_relative_p95_mm": 0.50,
}


def stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def load_rows(root: Path) -> list[dict]:
    candidates = [root / "training.jsonl", root / "evaluation.jsonl"]
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        raise FileNotFoundError(f"manifest missing under {root}")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                row["_root"] = str(root)
                row["_raw_root"] = protocol["raw_root"]
                rows.append(row)
    return rows


def select_gate_rows(rows: list[dict]) -> list[dict]:
    by_subject_camera: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_subject_camera[(row["subject_id"], row["camera_serial"])].append(row)
    selected = {}
    for key, group in sorted(by_subject_camera.items()):
        ordered_depth = sorted(group, key=lambda row: (row["root_depth_m"], row["sample_id"]))
        ordered_visible = sorted(group, key=lambda row: (row["hand_segmentation_pixels"], row["sample_id"]))
        candidates = [ordered_depth[0], ordered_depth[-1], ordered_visible[0], ordered_visible[-1]]
        for row in candidates:
            selected[row["sample_id"]] = row
    return [selected[key] for key in sorted(selected)]


def select_overlay_rows(rows: list[dict], count: int = 32) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["subject_id"], row["camera_serial"])].append(row)
    keys = sorted(groups)
    result = []
    cursor = 0
    while len(result) < min(count, len(rows)):
        before = len(result)
        for key in keys:
            group = sorted(groups[key], key=lambda row: stable_digest("overlay", row["sample_id"]))
            if cursor < len(group):
                result.append(group[cursor])
                if len(result) == min(count, len(rows)):
                    return result
        cursor += 1
        if len(result) == before:
            break
    return result


def patch_legacy_manopth_dependencies() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    for name, value in (
        ("bool", bool), ("int", int), ("float", float), ("complex", complex),
        ("object", object), ("str", str), ("unicode", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)


def load_mano(manopth_root: Path, mano_root: Path):
    patch_legacy_manopth_dependencies()
    sys.path.insert(0, str(manopth_root))
    from manopth.manolayer import ManoLayer

    layer = ManoLayer(
        flat_hand_mean=False,
        ncomps=45,
        side="right",
        mano_root=str(mano_root),
        use_pca=True,
    )
    layer.eval()
    return layer


def pose_rows_by_id(roots: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for root in roots:
        paths = [root / "training_mano.npz", root / "evaluation_mano.npz"]
        path = next((candidate for candidate in paths if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"MANO target missing under {root}")
        with np.load(path) as payload:
            for sample_id, pose, beta in zip(payload["sample_id"], payload["pose_m"], payload["betas"], strict=True):
                key = str(sample_id)
                if key in result:
                    raise ValueError(f"duplicate sample id across roots: {key}")
                result[key] = (np.asarray(pose, dtype=np.float32), np.asarray(beta, dtype=np.float32))
    return result


def decode_mano(layer, rows: list[dict], pose_by_id: dict[str, tuple[np.ndarray, np.ndarray]], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    joint_rows, vertex_rows = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            pose = np.stack([pose_by_id[row["sample_id"]][0] for row in batch])
            betas = np.stack([pose_by_id[row["sample_id"]][1] for row in batch])
            vertices, joints = layer(
                torch.from_numpy(pose[:, :48]),
                torch.from_numpy(betas),
                torch.from_numpy(pose[:, 48:51]),
            )
            joint_rows.append(joints.numpy() / 1000.0)
            vertex_rows.append(vertices.numpy() / 1000.0)
    return np.concatenate(joint_rows), np.concatenate(vertex_rows)


def label_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    joint_2d, joint_3d = [], []
    for row in rows:
        path = Path(row["_raw_root"]) / row["label_path"]
        with np.load(path) as label:
            joint_2d.append(np.asarray(label["joint_2d"], dtype=np.float32).reshape(21, 2))
            joint_3d.append(np.asarray(label["joint_3d"], dtype=np.float32).reshape(21, 3))
    return np.asarray(joint_2d), np.asarray(joint_3d)


def project(joints: np.ndarray, k: np.ndarray) -> np.ndarray:
    pixels_h = np.einsum("bij,bkj->bki", k, joints)
    return pixels_h[..., :2] / pixels_h[..., 2:3]


def fixed_rotation_diagnostic(decoded: np.ndarray, target: np.ndarray) -> dict:
    source = (decoded - decoded[:, :1]).reshape(-1, 3)
    destination = (target - target[:, :1]).reshape(-1, 3)
    covariance = source.T @ destination
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    rotated = source @ rotation
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "best_fixed_rotation_matrix": rotation.tolist(),
        "best_fixed_rotation_angle_deg": float(np.degrees(np.arccos(cosine))),
        "residual_after_fixed_rotation_mm": stats(np.linalg.norm(rotated - destination, axis=1) * 1000.0),
    }


def draw_overlay(row: dict, joint_2d: np.ndarray, projected: np.ndarray, output: Path) -> None:
    image_path = Path(row["_raw_root"]) / row["image_path"]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for label_xy, projected_xy in zip(joint_2d, projected, strict=True):
        lx, ly = map(float, label_xy)
        px, py = map(float, projected_xy)
        draw.line((lx, ly, px, py), fill=(255, 255, 0), width=1)
        draw.ellipse((lx - 2, ly - 2, lx + 2, ly + 2), fill=(0, 255, 0))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=(255, 0, 255), width=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)


def validate(args) -> Path:
    roots = [path.resolve() for path in args.roots]
    rows = [row for root in roots for row in load_rows(root)]
    subjects = {row["subject_id"] for row in rows}
    if args.mode == "pretrain" and subjects & TEST_SUBJECTS:
        raise PermissionError("pretrain coordinate gate must not read official S1 test")
    if args.mode == "final":
        freeze = json.loads(args.test_freeze_file.read_text(encoding="utf-8")) if args.test_freeze_file else {}
        if freeze.get("status") != "frozen" or not TEST_SUBJECTS <= subjects:
            raise PermissionError("final gate requires frozen recipe and both official test subjects")
    selected = select_gate_rows(rows)
    pose_by_id = pose_rows_by_id(roots)
    layer = load_mano(args.manopth_root, args.mano_root)
    decoded, vertices = decode_mano(layer, selected, pose_by_id, args.batch_size)
    label_2d, label_3d = label_rows(selected)
    k = np.asarray([row["intrinsics"] for row in selected], dtype=np.float32)
    projected = project(label_3d, k)
    reprojection = np.linalg.norm(projected - label_2d, axis=2)
    camera_error = np.linalg.norm(decoded - label_3d, axis=2) * 1000.0
    decoded_root = decoded - decoded[:, :1]
    label_root = label_3d - label_3d[:, :1]
    root_error = np.linalg.norm(decoded_root - label_root, axis=2) * 1000.0
    translation = np.stack([pose_by_id[row["sample_id"]][0][48:51] for row in selected])
    root_minus_translation = (label_3d[:, 0] - translation) * 1000.0

    overlay_rows = select_overlay_rows(selected)
    index_by_id = {row["sample_id"]: index for index, row in enumerate(selected)}
    overlay_manifest = []
    for number, row in enumerate(overlay_rows):
        index = index_by_id[row["sample_id"]]
        output = args.output_root / "overlays" / f"{number:02d}_{row['subject_id']}_{row['camera_serial']}_{row['frame_index']:06d}.jpg"
        draw_overlay(row, label_2d[index], projected[index], output)
        overlay_manifest.append({
            "sample_id": row["sample_id"],
            "path": str(output),
            "root_depth_m": row["root_depth_m"],
            "hand_segmentation_pixels": row["hand_segmentation_pixels"],
        })

    reprojection_stats = stats(reprojection)
    camera_stats = stats(camera_error)
    root_stats = stats(root_error)
    checks = {
        "reprojection_median_px": reprojection_stats["median"] <= THRESHOLDS["reprojection_median_px"],
        "reprojection_p95_px": reprojection_stats["p95"] <= THRESHOLDS["reprojection_p95_px"],
        "reprojection_max_px": reprojection_stats["max"] <= THRESHOLDS["reprojection_max_px"],
        "mano_camera_mean_mm": camera_stats["mean"] <= THRESHOLDS["mano_camera_mean_mm"],
        "mano_camera_p95_mm": camera_stats["p95"] <= THRESHOLDS["mano_camera_p95_mm"],
        "mano_camera_max_mm": camera_stats["max"] <= THRESHOLDS["mano_camera_max_mm"],
        "mano_root_relative_mean_mm": root_stats["mean"] <= THRESHOLDS["mano_root_relative_mean_mm"],
        "mano_root_relative_p95_mm": root_stats["p95"] <= THRESHOLDS["mano_root_relative_p95_mm"],
    }
    report = {
        "dataset": "DexYCB",
        "mode": args.mode,
        "status": "validated" if all(checks.values()) else "failed",
        "thresholds_predeclared": THRESHOLDS,
        "checks": checks,
        "coverage": {
            "samples": len(selected),
            "subjects": sorted(subjects),
            "subject_count": len(subjects),
            "sequences": len({row["episode_id"] for row in selected}),
            "camera_serials": sorted({row["camera_serial"] for row in selected}),
            "camera_count": len({row["camera_serial"] for row in selected}),
            "root_depth_m": stats(np.asarray([row["root_depth_m"] for row in selected])),
            "visible_hand_pixels": stats(np.asarray([row["hand_segmentation_pixels"] for row in selected])),
            "sampling": "per subject/camera: nearest, farthest, lowest and highest visible-hand segmentation area",
        },
        "projection_gate": {
            "source": "label joint_3d projected with per-camera color intrinsics versus label joint_2d",
            "error_px": reprojection_stats,
            "per_joint_mean_px": reprojection.mean(axis=0).tolist(),
        },
        "native_mano_gate": {
            "implementation": "official manopth ManoLayer; flat_hand_mean=False; PCA45; right hand; output mm divided by 1000",
            "pose_layout": "pose_m[0:3] global axis-angle; [3:48] articulated PCA45; [48:51] translation",
            "tip_vertex_ids": TIP_VERTEX_IDS,
            "joint_order": OPENPOSE_ORDER,
            "camera_error_mm": camera_stats,
            "root_relative_error_mm": root_stats,
            "per_joint_camera_mean_mm": camera_error.mean(axis=0).tolist(),
            "per_joint_root_relative_mean_mm": root_error.mean(axis=0).tolist(),
            "root_minus_pose_translation_mm": {
                "mean_xyz": root_minus_translation.mean(axis=0).tolist(),
                "std_xyz": root_minus_translation.std(axis=0).tolist(),
            },
            "fixed_rotation_diagnostic": fixed_rotation_diagnostic(decoded, label_3d),
            "decoded_vertex_shape": list(vertices.shape),
        },
        "overlays": overlay_manifest,
        "overlay_legend": "green=label joint_2d, magenta=projected joint_3d, yellow=error segment",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "coordinate_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "coverage": report["coverage"]}, indent=2))
    if report["status"] != "validated":
        raise SystemExit(2)
    return output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pretrain", "final"), required=True)
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--manopth-root", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


if __name__ == "__main__":
    validate(parse_args())
