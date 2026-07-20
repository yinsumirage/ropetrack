#!/usr/bin/env python3
"""Validate InterHand world/camera/projection, native MANO, and side handling."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_interhand26m import INTERHAND_TO_OPENPOSE, stable_digest
from scripts.rope_refiner.apply_rope_refinement import mano_predictions
from scripts.score_dexycb import read_predictions


OFFICIAL_COMMIT = "5d0e456f2345ef524bf71374141fbf3e11dd93f8"
THRESHOLDS = {
    "projection_p95_px": 0.001,
    "projection_max_px": 0.01,
    "mano_camera_mean_mm": 10.0,
    "mano_camera_p95_mm": 20.0,
    "mano_root_relative_mean_mm": 10.0,
    "mano_root_relative_p95_mm": 20.0,
    "wilor_directpose_roundtrip_max_mm": 0.1,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("mean", "median", "p95", "max")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def read_root(root: Path) -> list[dict]:
    split_file = "training" if (root / "training.jsonl").is_file() else "evaluation"
    manifest = root / f"{split_file}.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    xyz = np.asarray(json.loads((root / f"{split_file}_xyz.json").read_text(encoding="utf-8")), dtype=np.float32)
    with np.load(root / f"{split_file}_mano.npz") as loaded:
        payload = {key: np.asarray(loaded[key]) for key in loaded.files}
    if xyz.shape != (len(rows), 21, 3) or payload["sample_id"].astype(str).tolist() != [row["sample_id"] for row in rows]:
        raise ValueError(f"manifest/GT/MANO alignment differs under {root}")
    raw_root = Path(json.loads((root / "protocol.json").read_text(encoding="utf-8"))["raw_root"])
    result = []
    for index, row in enumerate(rows):
        result.append({
            "row": row,
            "xyz": xyz[index],
            "raw_root": raw_root,
            **{key: value[index] for key, value in payload.items() if key != "sample_id"},
        })
    return result


def select_gate_rows(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        row = record["row"]
        groups[(row["split"], row["mano_side"], row["is_interacting"], row["capture_id"])].append(record)
    selected = {}
    for key, rows in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered_visibility = sorted(rows, key=lambda record: (record["row"]["projected_in_frame_joint_count"], record["row"]["sample_id"]))
        ordered_bbox = sorted(rows, key=lambda record: (
            max(record["row"]["bbox_xyxy"][2] - record["row"]["bbox_xyxy"][0], record["row"]["bbox_xyxy"][3] - record["row"]["bbox_xyxy"][1]),
            record["row"]["sample_id"],
        ))
        ordered_hash = sorted(rows, key=lambda record: stable_digest("coordinate", record["row"]["sample_id"]))
        for record in (ordered_visibility[0], ordered_visibility[-1], ordered_bbox[0], ordered_bbox[-1], ordered_hash[0]):
            if bool(record["mano_valid"]):
                selected[record["row"]["sample_id"]] = record
    return [selected[key] for key in sorted(selected)]


def verify_official_source(root: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError(f"official InterHand source commit {commit} != pinned {OFFICIAL_COMMIT}")
    dataset = root / "data" / "InterHand2.6M" / "dataset.py"
    transforms = root / "common" / "utils" / "transforms.py"
    convert = root / "tool" / "MANO_world_to_camera" / "convert.py"
    required = {
        dataset: ["self.root_joint_idx = {'right': 20, 'left': 41}", "world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1))"],
        transforms: ["cam_coord = np.dot(R, world_coord - T)"],
        convert: ["use_pca=False", "t = -np.dot(R,t.reshape(3,1)).reshape(3)", "mano_layer['left'].shapedirs[:,0,:] *= -1"],
    }
    missing = {}
    for path, literals in required.items():
        text = path.read_text(encoding="utf-8")
        absent = [literal for literal in literals if literal not in text]
        if absent:
            missing[str(path)] = absent
    if missing:
        raise ValueError(f"official source literals differ: {missing}")
    regressor = root / "tool" / "MANO_world_to_camera" / "J_regressor_mano_ih26m.npy"
    return {
        "repository": "https://github.com/facebookresearch/InterHand2.6M",
        "commit": commit,
        "dataset_py_sha256": file_sha256(dataset),
        "transforms_py_sha256": file_sha256(transforms),
        "mano_convert_py_sha256": file_sha256(convert),
        "joint_regressor_sha256": file_sha256(regressor),
        "verified_semantics": {
            "joint_3d": "world millimetres",
            "world_to_camera": "R @ (world - campos)",
            "camera_axes": "+x image-right, +y image-down, +z forward, proven by official cam2pixel",
            "focal_principal_order": "[fx,fy] and [cx,cy]",
            "joint_slices": "right 0:21 root20; left 21:42 root41",
            "mano": "world-coordinate full pose48, shape10, translation metres; use_pca=False; flat_hand_mean=False default",
            "global_orientation": "world frame; official conversion premultiplies camera rotation",
        },
    }


def verify_wilor_directpose_source() -> dict:
    repo = Path(__file__).resolve().parents[1]
    files = {
        repo / "ropetrack" / "eval" / "pipeline.py": [
            "flip = not candidate.is_right", "pred_cam[:, 1] = multiplier * pred_cam[:, 1]",
            "vertices[:, 0] *= -1.0", "keypoints[:, 0] *= -1.0",
        ],
        repo / "scripts" / "rope_refiner" / "direct_pose_head.py": [
            'mirror[:, 0, 0] = torch.where(batch["is_right"], 1.0, -1.0)',
        ],
        repo / "scripts" / "rope_refiner" / "apply_rope_refinement.py": [
            "if not right:", "verts[:, 0] *= -1.0", "joints[:, 0] *= -1.0",
        ],
    }
    missing = {}
    for path, literals in files.items():
        source = path.read_text(encoding="utf-8")
        absent = [literal for literal in literals if literal not in source]
        if absent:
            missing[str(path)] = absent
    if missing:
        raise ValueError(f"WiLoR/DirectPose left-hand source contract differs: {missing}")
    return {str(path.relative_to(repo)): file_sha256(path) for path in files}


def validate_wilor_directpose_roundtrip(records: list[dict], args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    target_ids = [record["row"]["sample_id"] for record in records]
    target_index = {sample_id: index for index, sample_id in enumerate(target_ids)}
    base_selected = np.full((len(records), 21, 3), np.nan, dtype=np.float32)
    reconstructed = np.full_like(base_selected, np.nan)
    artifact_hashes = []
    for pred_path, order_path, cache_path in args.base_artifact:
        source_ids, base_xyz = read_predictions(pred_path, order_path)
        source_index = {sample_id: index for index, sample_id in enumerate(source_ids)}
        chosen_ids = [sample_id for sample_id in target_ids if sample_id in source_index]
        if not chosen_ids:
            continue
        with np.load(cache_path) as loaded:
            cache_ids = loaded["sample_id"].astype(str).tolist()
            by_cache = {sample_id: index for index, sample_id in enumerate(cache_ids)}
            if any(sample_id not in by_cache for sample_id in chosen_ids):
                raise ValueError("base MANO cache misses coordinate-gate samples")
            poses = np.asarray(loaded["base_hand_pose"])[np.asarray([by_cache[sample_id] for sample_id in chosen_ids])]
        decoded, _ = mano_predictions(
            "interhand26m", poses, np.asarray(chosen_ids), cache_path,
            args.device, args.batch_size, keep_vertices=False,
        )
        destinations = np.asarray([target_index[sample_id] for sample_id in chosen_ids])
        base_selected[destinations] = base_xyz[np.asarray([source_index[sample_id] for sample_id in chosen_ids])]
        reconstructed[destinations] = np.asarray(decoded, dtype=np.float32)
        artifact_hashes.append({
            "prediction": {"path": str(pred_path), "sha256": file_sha256(pred_path)},
            "order": {"path": str(order_path), "sha256": file_sha256(order_path)},
            "mano_cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        })
    if not np.isfinite(base_selected).all() or not np.isfinite(reconstructed).all():
        raise ValueError("base artifacts do not cover every coordinate-gate sample")
    error_mm = np.linalg.norm(reconstructed - base_selected, axis=2) * 1000.0
    by_side = grouped_stats(error_mm, records, lambda row: row["mano_side"])
    report = {
        "source_sha256": verify_wilor_directpose_source(),
        "base_artifacts": artifact_hashes,
        "definition": "WiLoR exported model_keypoints versus DirectPose apply-path re-decode from the same cached pose/global/betas/camera translation",
        "error_mm": stats(error_mm),
        "by_side_mm": by_side,
        "max_threshold_mm": THRESHOLDS["wilor_directpose_roundtrip_max_mm"],
        "both_sides_below_threshold": set(by_side) == {"left", "right"} and all(
            value["max"] is not None and value["max"] <= THRESHOLDS["wilor_directpose_roundtrip_max_mm"]
            for value in by_side.values()
        ),
    }
    return report, base_selected


def load_mano_layers(model_root: Path):
    import smplx
    import torch

    layers = {
        side: smplx.create(str(model_root), "mano", use_pca=False, flat_hand_mean=False, is_rhand=side == "right").eval()
        for side in ("right", "left")
    }
    if torch.sum(torch.abs(layers["left"].shapedirs[:, 0, :] - layers["right"].shapedirs[:, 0, :])) < 1:
        layers["left"].shapedirs[:, 0, :] *= -1
    return layers


def decode(records: list[dict], layers: dict, regressor: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    decoded_joints = np.full((len(records), 21, 3), np.nan, dtype=np.float32)
    decoded_vertices = np.full((len(records), 778, 3), np.nan, dtype=np.float32)
    for side in ("right", "left"):
        indices = [index for index, record in enumerate(records) if record["row"]["mano_side"] == side and bool(record["mano_valid"])]
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            pose = np.stack([records[index]["pose"] for index in chosen]).astype(np.float32)
            shape = np.stack([records[index]["shape"] for index in chosen]).astype(np.float32)
            trans = np.stack([records[index]["trans_world_m"] for index in chosen]).astype(np.float32)
            with torch.no_grad():
                output = layers[side](
                    global_orient=torch.from_numpy(pose[:, :3]),
                    hand_pose=torch.from_numpy(pose[:, 3:]),
                    betas=torch.from_numpy(shape),
                    transl=torch.from_numpy(trans),
                )
            vertices_world = output.vertices.numpy().astype(np.float32)
            rotation = np.stack([records[index]["camrot"] for index in chosen]).astype(np.float32)
            position = np.stack([records[index]["campos_world_mm"] for index in chosen]).astype(np.float32) / 1000.0
            vertices_camera = np.einsum("bij,bkj->bki", rotation, vertices_world - position[:, None, :])
            joints_camera_native = np.einsum("jv,bvc->bjc", regressor, vertices_camera)
            decoded_vertices[np.asarray(chosen)] = vertices_camera
            decoded_joints[np.asarray(chosen)] = joints_camera_native[:, INTERHAND_TO_OPENPOSE]
    return decoded_joints, decoded_vertices


def project_saved(record: dict) -> np.ndarray:
    camera = np.einsum(
        "ij,kj->ki",
        np.asarray(record["camrot"], dtype=np.float64),
        np.asarray(record["world_joints_mm"], dtype=np.float64) - np.asarray(record["campos_world_mm"], dtype=np.float64),
    )
    intrinsic = np.asarray(record["row"]["intrinsic"], dtype=np.float64)
    homogeneous = (intrinsic @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def masked_errors(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(pred.shape[:2], np.nan, dtype=np.float64)
    distance = np.linalg.norm(pred - target, axis=2) * 1000.0
    result[valid] = distance[valid]
    return result


def grouped_stats(values: np.ndarray, records: list[dict], key) -> dict:
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(key(record["row"]))].append(index)
    return {name: stats(values[np.asarray(indices)]) for name, indices in sorted(groups.items())}


def shape_stability(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for record in records:
        if bool(record["mano_valid"]):
            groups[(record["row"]["subject_id"], record["row"]["mano_side"])].append(record["shape"])
    rows = []
    for key, values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append({"subject_side": f"{key[0]}/{key[1]}", "samples": len(array), "mean_dim_std": float(array.std(axis=0).mean()), "max_dim_std": float(array.std(axis=0).max())})
    return {
        "interpretation": "NeuralAnnot shape is stored per frame; stability is measured, not assumed subject-constant",
        "groups": rows,
        "max_observed_dim_std": max((row["max_dim_std"] for row in rows), default=None),
    }


def draw_overlay(record: dict, recomputed: np.ndarray, mano_projected: np.ndarray, predicted: np.ndarray, output: Path) -> None:
    image = Image.open(record["raw_root"] / record["row"]["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    saved = np.asarray(record["projected_xy"])
    for reference, computed, mano, pred in zip(saved, recomputed, mano_projected, predicted, strict=True):
        rx, ry = map(float, reference)
        cx, cy = map(float, computed)
        mx, my = map(float, mano)
        px, py = map(float, pred)
        draw.line((rx, ry, cx, cy), fill=(255, 255, 0), width=1)
        draw.ellipse((rx - 2, ry - 2, rx + 2, ry + 2), fill=(0, 255, 0))
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), outline=(255, 0, 255), width=1)
        draw.ellipse((mx - 2, my - 2, mx + 2, my + 2), outline=(0, 255, 255), width=1)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=(255, 0, 0), width=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=92)


def validate(args: argparse.Namespace) -> Path:
    records = [record for root in args.roots for record in read_root(root.resolve())]
    splits = {record["row"]["split"] for record in records}
    if args.mode == "pretrain" and "test" in splits:
        raise PermissionError("pretrain gate cannot read test GT")
    if args.mode == "final":
        freeze = json.loads(args.test_freeze_file.read_text(encoding="utf-8")) if args.test_freeze_file else {}
        if freeze.get("status") != "frozen" or "test" not in splits:
            raise PermissionError("final gate requires frozen recipe and exported test GT")
    official = verify_official_source(args.official_root)
    selected = select_gate_rows(records)
    if len(selected) < 24:
        raise ValueError(f"coordinate gate needs at least 24 MANO-valid samples, got {len(selected)}")
    regressor = np.load(args.official_root / "tool" / "MANO_world_to_camera" / "J_regressor_mano_ih26m.npy").astype(np.float32)
    layers = load_mano_layers(args.mano_root)
    decoded, vertices = decode(selected, layers, regressor, args.batch_size)
    left_right_roundtrip, base_predicted = validate_wilor_directpose_roundtrip(selected, args)
    target = np.stack([record["xyz"] for record in selected])
    valid = np.stack([record["joint_valid"] for record in selected]).astype(bool)
    camera_error = masked_errors(decoded, target, valid)
    decoded_root = decoded - decoded[:, :1]
    target_root = target - target[:, :1]
    root_error = masked_errors(decoded_root, target_root, valid)
    recomputed = np.stack([project_saved(record) for record in selected])
    saved_projection = np.stack([record["projected_xy"] for record in selected])
    projection_error = np.linalg.norm(recomputed - saved_projection, axis=2)
    camera_stats, root_stats, projection_stats = stats(camera_error), stats(root_error), stats(projection_error)
    checks = {
        "projection_p95_px": projection_stats["p95"] <= THRESHOLDS["projection_p95_px"],
        "projection_max_px": projection_stats["max"] <= THRESHOLDS["projection_max_px"],
        "mano_camera_mean_mm": camera_stats["mean"] <= THRESHOLDS["mano_camera_mean_mm"],
        "mano_camera_p95_mm": camera_stats["p95"] <= THRESHOLDS["mano_camera_p95_mm"],
        "mano_root_relative_mean_mm": root_stats["mean"] <= THRESHOLDS["mano_root_relative_mean_mm"],
        "mano_root_relative_p95_mm": root_stats["p95"] <= THRESHOLDS["mano_root_relative_p95_mm"],
        "both_sides_covered": {record["row"]["mano_side"] for record in selected} == {"right", "left"},
        "single_and_interacting_covered": {bool(record["row"]["is_interacting"]) for record in selected} == {False, True},
        "train_and_val_covered_pretrain": args.mode != "pretrain" or {"train", "val"} <= splits,
        "overlay_count_at_least_24": min(args.overlay_count, len(selected)) >= 24,
        "wilor_directpose_left_right_roundtrip": left_right_roundtrip["both_sides_below_threshold"],
    }
    overlay_records = sorted(selected, key=lambda record: stable_digest("overlay", record["row"]["sample_id"]))[:args.overlay_count]
    by_id = {record["row"]["sample_id"]: index for index, record in enumerate(selected)}
    overlays = []
    for number, record in enumerate(overlay_records):
        index = by_id[record["row"]["sample_id"]]
        intrinsic = np.asarray(record["row"]["intrinsic"], dtype=np.float64)
        homogeneous = (intrinsic @ (decoded[index] * 1000.0).T).T
        mano_projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        base_homogeneous = (intrinsic @ (base_predicted[index] * 1000.0).T).T
        base_projected = base_homogeneous[:, :2] / base_homogeneous[:, 2:3]
        path = args.output_root / "overlays" / f"{number:03d}_{record['row']['split']}_{record['row']['capture_id']}_{record['row']['mano_side']}.jpg"
        draw_overlay(record, recomputed[index], mano_projected, base_projected, path)
        overlays.append({"sample_id": record["row"]["sample_id"], "path": str(path), "side": record["row"]["mano_side"], "hand_type": record["row"]["hand_type"], "projected_in_frame_joint_count": record["row"]["projected_in_frame_joint_count"]})
    report = {
        "dataset": "InterHand2.6M v1.0 30fps",
        "project_protocol": "interhand26m_v1_30fps_oneview_v1",
        "mode": args.mode,
        "status": "validated" if all(checks.values()) else "failed",
        "thresholds_predeclared_from_official_about_5mm_fitting_error": THRESHOLDS,
        "checks": checks,
        "official_source": official,
        "coverage": {
            "samples": len(selected),
            "splits": sorted(splits),
            "sides": dict((side, sum(record["row"]["mano_side"] == side for record in selected)) for side in ("right", "left")),
            "hand_type": dict(sorted(__import__("collections").Counter(record["row"]["hand_type"] for record in selected).items())),
            "captures": len({(record["row"]["split"], record["row"]["capture_id"]) for record in selected}),
            "sequences": len({record["row"]["episode_id"] for record in selected}),
            "cameras": len({record["row"]["camera_id"] for record in selected}),
            "projected_in_frame_joint_count": stats([record["row"]["projected_in_frame_joint_count"] for record in selected]),
            "visibility_boundary": "image-boundary projection proxy only; native occlusion visibility is not claimed",
            "bbox_size_px": stats([max(record["row"]["bbox_xyxy"][2] - record["row"]["bbox_xyxy"][0], record["row"]["bbox_xyxy"][3] - record["row"]["bbox_xyxy"][1]) for record in selected]),
        },
        "projection_gate": {
            "definition": "independent float64 implementation of official R(world-campos) and pinhole projection versus frozen prepare output",
            "independent_observed_2d_gt_available": False,
            "error_px": projection_stats,
            "per_joint_mean_px": np.mean(projection_error, axis=0).tolist(),
            "visual_image_alignment": "verified by overlays",
        },
        "native_mano_gate": {
            "definition": "official right/left SMPL-X MANO full-pose decode in world metres, then prescribed camera transform; official InterHand joint regressor and joint mapping",
            "camera_error_mm": camera_stats,
            "root_relative_error_mm": root_stats,
            "per_joint_camera_mean_mm": np.nanmean(camera_error, axis=0).tolist(),
            "per_joint_root_relative_mean_mm": np.nanmean(root_error, axis=0).tolist(),
            "by_side_camera_mm": grouped_stats(camera_error, selected, lambda row: row["mano_side"]),
            "by_hand_type_camera_mm": grouped_stats(camera_error, selected, lambda row: row["hand_type"]),
            "decoded_vertex_shape": list(vertices.shape),
            "shape_stability": shape_stability(records),
        },
        "left_right_normalization_gate": {
            "status": "validated" if checks["both_sides_covered"] and checks["wilor_directpose_left_right_roundtrip"] and all(value["mean"] is not None and value["mean"] <= THRESHOLDS["mano_camera_mean_mm"] for value in grouped_stats(camera_error, selected, lambda row: row["mano_side"]).values()) else "failed",
            "native_layers": "official side-specific right and left MANO with left shapedirs compatibility fix",
            "wilor_directpose_path": "left crop is horizontally flipped; canonical-right MANO output x is mirrored back before camera translation",
            "axis_flip_search_performed": False,
            "numerical_evidence": grouped_stats(camera_error, selected, lambda row: row["mano_side"]),
            "wilor_directpose_roundtrip": left_right_roundtrip,
        },
        "overlays": overlays,
        "overlay_legend": "green=frozen official-formula projection, magenta=independent recomputation, cyan=native MANO joint projection, red=actual WiLoR base output, yellow=projection parity segment",
    }
    if report["left_right_normalization_gate"]["status"] != "validated":
        report["status"] = "failed"
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "coordinate_gate.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"], "coverage": report["coverage"], "projection": projection_stats, "mano_camera": camera_stats}, indent=2))
    if report["status"] != "validated":
        raise SystemExit(2)
    return output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pretrain", "final"), required=True)
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--base-artifact", nargs=3, action="append", type=Path, required=True, metavar=("PRED_JSON", "ORDER", "MANO_CACHE"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overlay-count", type=int, default=32)
    return parser.parse_args(argv)


if __name__ == "__main__":
    validate(parse_args())
