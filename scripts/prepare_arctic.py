#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


EGO_SCALE = 0.3
IMAGE_SIZE = (840, 600)
OPENPOSE_ORDER = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
JOINT_ORDER = [
    "wrist",
    "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
SPLIT_SUBJECTS = {
    "train": ("s01", "s02", "s04", "s06", "s07", "s08", "s09", "s10"),
    "val": ("s05",),
}
MANIFEST_SPLITS = {"train": "training", "val": "evaluation"}


def annotation_index(frame_index: int, ioi_offset: int) -> int:
    return int(frame_index) - int(ioi_offset)


def scaled_intrinsic(intrinsic: np.ndarray, scale: float = EGO_SCALE) -> np.ndarray:
    out = np.asarray(intrinsic, dtype=np.float32).copy()
    out[:2] *= float(scale)
    return out


def hand_bbox(points: np.ndarray, margin: float, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(xy).all(axis=1)
    if not valid.any():
        raise ValueError("no finite hand joints for bbox")
    lo, hi = xy[valid].min(axis=0), xy[valid].max(axis=0)
    center = (lo + hi) / 2
    size = max(float((hi - lo).max()), 1.0)
    pad = float(margin) * size
    xyxy = np.asarray([lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad], dtype=np.float32)
    width, height = image_size
    xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0.0, width - 1.0)
    xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0.0, height - 1.0)
    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        raise ValueError(f"degenerate clipped hand bbox: {xyxy.tolist()}")
    return xyxy


def sample_id(sequence: str, side: str, frame_index: int) -> str:
    return f"{sequence}/{side}/{frame_index:05d}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_array(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for idx, row in enumerate(rows):
            if idx:
                handle.write(",")
            handle.write(json.dumps(np.asarray(row).tolist(), separators=(",", ":")))
        handle.write("]\n")
    temporary.replace(path)


def decode_gt_vertices(records: list[dict], data: dict, model_dir: Path, batch_size: int) -> np.ndarray:
    import torch
    from smplx import MANO

    vertices = np.empty((len(records), 778, 3), dtype=np.float32)
    for side, short in (("right", "r"), ("left", "l")):
        layer = MANO(str(model_dir), use_pca=False, flat_hand_mean=False, is_rhand=side == "right").eval()
        indices = [idx for idx, record in enumerate(records) if record["side"] == side]
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            rot = np.stack([
                data[records[idx]["sequence"]]["cam_coord"][f"rot_{short}_cam"][records[idx]["vidx"], 0]
                for idx in selected
            ]).astype(np.float32)
            pose = np.stack([
                data[records[idx]["sequence"]]["params"][f"pose_{short}"][records[idx]["vidx"]]
                for idx in selected
            ]).astype(np.float32)
            shape = np.stack([
                data[records[idx]["sequence"]]["params"][f"shape_{short}"][records[idx]["vidx"]]
                for idx in selected
            ]).astype(np.float32)
            roots = np.stack([
                data[records[idx]["sequence"]]["cam_coord"][f"joints.{side}"][records[idx]["vidx"], 0, 0]
                for idx in selected
            ]).astype(np.float32)
            with torch.no_grad():
                output = layer(
                    global_orient=torch.from_numpy(rot),
                    hand_pose=torch.from_numpy(pose),
                    betas=torch.from_numpy(shape),
                )
            decoded = output.vertices.numpy().astype(np.float32)
            decoded += (roots - output.joints[:, 0].numpy())[:, None, :]
            vertices[np.asarray(selected)] = decoded
    return vertices


def build_records(
    split: dict,
    arctic_root: Path,
    misc: dict,
    margin: float,
    frame_stride: int,
    subjects: tuple[str, ...] = SPLIT_SUBJECTS["val"],
    check_images: bool = True,
    limit: int = 0,
) -> tuple[list, np.ndarray]:
    records, xyz = [], []
    for image_name in split["imgnames"]:
        sid, sequence_name, view, filename = Path(image_name).parts[-4:]
        if sid not in subjects or view != "0":
            raise ValueError(f"P2 expected subjects={subjects}/view0, got {image_name}")
        frame = int(Path(filename).stem)
        if (frame - 10) % frame_stride:
            continue
        sequence = f"{sid}/{sequence_name}"
        seq = split["data_dict"][sequence]
        vidx = annotation_index(frame, misc[sid]["ioi_offset"])
        image_path = arctic_root / "cropped_images" / sid / sequence_name / view / filename
        if check_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not bool(seq["cam_coord"]["is_valid"][vidx, 0]):
            continue
        intrinsic = scaled_intrinsic(seq["params"]["K_ego"][vidx])
        for side in ("right", "left"):
            if not bool(seq["cam_coord"][f"{side}_valid"][vidx, 0]):
                continue
            joints_native = np.asarray(seq["cam_coord"][f"joints.{side}"][vidx, 0], dtype=np.float32)
            bbox_points = np.asarray(seq["2d"][f"joints.{side}"][vidx, 9], dtype=np.float32) * EGO_SCALE
            row = {
                "sample_id": sample_id(sequence, side, frame),
                "image_path": str(image_path),
                "bbox_xyxy": hand_bbox(bbox_points, margin).tolist(),
                "is_right": side == "right",
                "episode_id": f"{sequence}/{side}",
                "frame_index": frame,
                "intrinsic": intrinsic.tolist(),
                "joint_confidence": [1.0] * 21,
                "sequence": sequence,
                "annotation_index": vidx,
                "ioi_offset": int(misc[sid]["ioi_offset"]),
                "bbox_source": "distorted_2d_view9_x0.3",
            }
            records.append({"manifest": row, "sequence": sequence, "vidx": vidx, "side": side})
            xyz.append(joints_native[OPENPOSE_ORDER])
            if limit > 0 and len(records) >= limit:
                return records, np.asarray(xyz, dtype=np.float32)
    return records, np.asarray(xyz, dtype=np.float32)


def verify_records(records: list[dict], xyz: np.ndarray, verts: np.ndarray | None) -> Counter:
    ids = [record["manifest"]["sample_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ARCTIC sample ids")
    if xyz.shape != (len(records), 21, 3):
        raise ValueError(f"GT joint shape mismatch: xyz={xyz.shape} records={len(records)}")
    if verts is not None and verts.shape != (len(records), 778, 3):
        raise ValueError(f"GT mesh shape mismatch: verts={verts.shape} records={len(records)}")
    if not np.isfinite(xyz).all() or (verts is not None and not np.isfinite(verts).all()):
        raise ValueError("non-finite ARCTIC GT")
    return Counter(record["side"] for record in records)


def prepare(args: argparse.Namespace) -> Path:
    split_path = args.arctic_root / f"splits/p2_{args.split}.npy"
    split = np.load(split_path, allow_pickle=True).item()
    misc = json.loads((args.arctic_root / "meta/misc.json").read_text())
    records, xyz = build_records(
        split,
        args.arctic_root,
        misc,
        args.bbox_margin,
        args.frame_stride,
        SPLIT_SUBJECTS[args.split],
        not args.skip_image_check,
        args.limit,
    )

    write_verts = args.write_verts or args.split == "val"
    verts = None
    if write_verts:
        import torch
        torch.set_num_threads(args.num_threads)
        verts = decode_gt_vertices(records, split["data_dict"], args.model_dir, args.batch_size)
    counts = verify_records(records, xyz, verts)
    if args.split == "val" and args.frame_stride == 1 and args.limit <= 0:
        expected = {"right": 20005, "left": 18916}
        if len(split["data_dict"]) != 34 or len(split["imgnames"]) != 25203 or dict(counts) != expected:
            raise ValueError(
                f"full P2 val manifest mismatch: sequences={len(split['data_dict'])} "
                f"images={len(split['imgnames'])} sides={dict(counts)} expected={expected}"
            )
    if args.split == "train" and args.frame_stride == 1 and args.limit <= 0:
        if len(split["data_dict"]) != 267 or len(split["imgnames"]) != 187050:
            raise ValueError(
                f"full P2 train manifest mismatch: sequences={len(split['data_dict'])} "
                f"images={len(split['imgnames'])}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest_split = MANIFEST_SPLITS[args.split]
    manifest_path = args.output_root / f"{manifest_split}.jsonl"
    manifest_tmp = manifest_path.with_suffix(".jsonl.tmp")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["manifest"], separators=(",", ":")) + "\n")
    manifest_tmp.replace(manifest_path)
    xyz_path = args.output_root / f"{manifest_split}_xyz.json"
    write_json_array(xyz_path, xyz)
    output_hashes = {
        manifest_path.name: file_sha256(manifest_path),
        xyz_path.name: file_sha256(xyz_path),
    }
    if verts is not None:
        verts_path = args.output_root / f"{manifest_split}_verts.json"
        write_json_array(verts_path, verts)
        output_hashes[verts_path.name] = file_sha256(verts_path)

    protocol = {
        "dataset": "arctic",
        "setup": "p2",
        "split": args.split,
        "subjects": list(SPLIT_SUBJECTS[args.split]),
        "views": [0],
        "source_images": "cropped_images",
        "image_size": list(IMAGE_SIZE),
        "ego_scale": EGO_SCALE,
        "bbox_2d_view": 9,
        "gt_3d_view": 0,
        "first_last_frames_excluded": 10,
        "frame_index_rule": "image_basename_minus_ioi_offset",
        "frame_stride": args.frame_stride,
        "sample_side_order": ["right", "left"],
        "joint_order": JOINT_ORDER,
        "joint_source": "official_mano_kinematic_joints_reordered_openpose",
        "mesh_source": "native_left_right_mano_root_aligned_to_official_wrist",
        "mesh_exported": verts is not None,
        "coordinate_frame": "opencv_camera_m",
        "num_sequences": len(split["data_dict"]),
        "num_samples": len(records),
        "side_counts": dict(counts),
        "bbox_margin": args.bbox_margin,
        "source_split": str(split_path),
        "sha256": output_hashes,
    }
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({key: protocol[key] for key in ("num_sequences", "num_samples", "side_counts", "frame_stride")}, indent=2))
    return args.output_root


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build RopeTrack ARCTIC P2 hand manifests and GT.")
    parser.add_argument("--arctic-root", type=Path, default=Path("/data/wentao/datasets/arctic/unpack/arctic_data/data"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", choices=sorted(SPLIT_SUBJECTS), default="val")
    parser.add_argument("--write-verts", action="store_true", help="Also decode native-side MANO meshes; val always does this.")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip per-row stat calls after an external image-tree integrity audit.")
    parser.add_argument("--model-dir", type=Path, default=repo / "mano_data")
    parser.add_argument("--bbox-margin", type=float, default=0.25)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    if args.bbox_margin < 0 or args.frame_stride < 1 or args.batch_size < 1 or args.num_threads < 1:
        parser.error("bbox margin must be >=0; stride, batch size, and threads must be positive")
    if args.output_root is None:
        args.output_root = Path(f"/data/wentao/ropetrack/processed/arctic/p2_{args.split}")
    return args


if __name__ == "__main__":
    prepare(parse_args())
