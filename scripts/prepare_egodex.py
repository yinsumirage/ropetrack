#!/usr/bin/env python3
"""Decode EgoDex videos into RopeTrack's image + 21-joint manifest format.

EgoDex stores world-space ARKit/visionOS transforms, not MANO parameters.
This exporter converts joint translations into an OpenCV camera frame and
keeps the native confidence values.  MANO predictions are produced later by
``scripts/eval.py --save-mano-cache``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

FINGER_JOINTS = {
    "Thumb": ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
    "IndexFinger": ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
    "MiddleFinger": ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
    "RingFinger": ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
    "LittleFinger": ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip"),
}
RIGHT_FRAME_OFFSET = 1_000_000


def joint_names(side: str) -> tuple[str, ...]:
    prefix = side.lower()
    names = [f"{prefix}Hand"]
    for finger, joints in FINGER_JOINTS.items():
        names.extend(f"{prefix}{finger}{joint}" for joint in joints)
    if len(names) != 21:
        raise AssertionError(f"expected 21 joints, got {len(names)}")
    return tuple(names)


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return token or "episode"


def temporal_sample_id(task: str, episode: str, side: str, frame_index: int) -> str:
    """Unique row id while keeping both hands in one episode split group."""
    encoded_frame = frame_index + (RIGHT_FRAME_OFFSET if side == "right" else 0)
    return f"{task}__{episode}/{encoded_frame:07d}"


def camera_joints(h5: h5py.File, names: tuple[str, ...], frame_index: int) -> np.ndarray:
    """World-space translations -> camera coordinates used by EgoDex K (m)."""
    world_from_camera = np.asarray(h5["transforms/camera"][frame_index], dtype=np.float64)
    camera_from_world = np.linalg.inv(world_from_camera)
    world = np.stack([
        np.asarray(h5[f"transforms/{name}"][frame_index, :3, 3], dtype=np.float64)
        for name in names
    ])
    homogeneous = np.concatenate([world, np.ones((len(world), 1))], axis=1)
    # This intentionally matches Apple's visualize_2d.py: inv(cam_ext) @ tf,
    # followed directly by cv2.projectPoints with no axis sign conversion.
    return (camera_from_world @ homogeneous.T).T[:, :3].astype(np.float32)


def joint_confidences(h5: h5py.File, names: tuple[str, ...], frame_index: int) -> np.ndarray:
    if "confidences" not in h5:
        return np.ones(21, dtype=np.float32)
    return np.asarray([h5[f"confidences/{name}"][frame_index] for name in names], dtype=np.float32)


def padded_bbox(joints: np.ndarray, intrinsic: np.ndarray, width: int, height: int, margin: float) -> np.ndarray | None:
    z = joints[:, 2]
    valid = np.isfinite(joints).all(axis=1) & (z > 1e-4)
    if not valid.any():
        return None
    uvw = (np.asarray(intrinsic, dtype=np.float32) @ joints[valid].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    valid_uv = np.isfinite(uv).all(axis=1)
    if not valid_uv.any():
        return None
    xy_min = uv[valid_uv].min(axis=0)
    xy_max = uv[valid_uv].max(axis=0)
    pad = margin * max(float(np.max(xy_max - xy_min)), 1.0)
    box = np.asarray([
        max(0.0, float(xy_min[0] - pad)),
        max(0.0, float(xy_min[1] - pad)),
        min(float(width), float(xy_max[0] + pad)),
        min(float(height), float(xy_max[1] + pad)),
    ], dtype=np.float32)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def source_pairs(input_root: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for h5_path in sorted(input_root.glob("*/*.hdf5")):
        video_path = h5_path.with_suffix(".mp4")
        if not video_path.exists():
            raise FileNotFoundError(f"matching video missing: {video_path}")
        pairs.append((h5_path, video_path))
    if not pairs:
        raise FileNotFoundError(f"no */*.hdf5 files under {input_root}")
    return pairs


def export(args: argparse.Namespace) -> dict:
    import cv2

    output_root = args.output_root
    manifest_path = output_root / f"{args.split}.jsonl"
    xyz_path = output_root / f"{args.split}_xyz.json"
    if manifest_path.exists() or xyz_path.exists():
        raise FileExistsError(f"completed split already exists under {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(".jsonl.tmp")
    xyz_tmp = xyz_path.with_suffix(".json.tmp")
    sides = ("left", "right") if args.hands == "both" else (args.hands,)
    pairs = source_pairs(args.input_root)
    if args.episode_limit > 0:
        pairs = pairs[: args.episode_limit]

    rows = frames_written = episodes = 0
    with manifest_tmp.open("w", encoding="utf-8") as manifest_file, xyz_tmp.open("w", encoding="utf-8") as xyz_file:
        xyz_file.write("[")
        first_xyz = True
        stop = False
        for h5_path, video_path in pairs:
            if stop:
                break
            task = safe_token(h5_path.parent.name)
            episode = safe_token(h5_path.stem)
            with h5py.File(h5_path, "r") as h5:
                intrinsic = np.asarray(h5["camera/intrinsic"], dtype=np.float32)
                annotation_frames = int(h5["transforms/camera"].shape[0])
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open video: {video_path}")
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if video_frames > 0 and video_frames != annotation_frames:
                    capture.release()
                    raise ValueError(
                        f"frame mismatch for {h5_path}: HDF5={annotation_frames}, video={video_frames}"
                    )

                for frame_index in range(annotation_frames):
                    ok, image = capture.read()
                    if not ok:
                        capture.release()
                        raise RuntimeError(f"video ended at frame {frame_index}: {video_path}")
                    if frame_index % args.frame_stride:
                        continue

                    sample_rows = []
                    for side in sides:
                        names = joint_names(side)
                        joints = camera_joints(h5, names, frame_index)
                        confidence = joint_confidences(h5, names, frame_index)
                        if float(confidence.mean()) < args.min_mean_confidence:
                            continue
                        bbox = padded_bbox(joints, intrinsic, width, height, args.bbox_margin)
                        if bbox is None:
                            continue
                        sample_id = temporal_sample_id(task, episode, side, frame_index)
                        image_rel = Path(args.split) / "frames" / task / episode / f"{frame_index:06d}.jpg"
                        sample_rows.append(({
                            "sample_id": sample_id,
                            "image_path": image_rel.as_posix(),
                            "episode_id": f"{task}/{episode}",
                            "frame_index": frame_index,
                            "is_right": side == "right",
                            "bbox_xyxy": bbox.tolist(),
                            "intrinsic": intrinsic.tolist(),
                            "joint_confidence": confidence.tolist(),
                            "mean_confidence": float(confidence.mean()),
                            "source_hdf5": h5_path.relative_to(args.input_root).as_posix(),
                        }, joints))

                    if not sample_rows:
                        continue
                    image_path = output_root / sample_rows[0][0]["image_path"]
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                        raise RuntimeError(f"failed to write image: {image_path}")
                    frames_written += 1
                    for row, joints in sample_rows:
                        if args.sample_limit > 0 and rows >= args.sample_limit:
                            stop = True
                            break
                        manifest_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                        if not first_xyz:
                            xyz_file.write(",")
                        xyz_file.write(json.dumps(joints.tolist(), separators=(",", ":")))
                        first_xyz = False
                        rows += 1
                        if args.sample_limit > 0 and rows >= args.sample_limit:
                            stop = True
                            break
                    if stop:
                        break
                capture.release()
            episodes += 1
        xyz_file.write("]\n")

    manifest_tmp.replace(manifest_path)
    xyz_tmp.replace(xyz_path)
    meta = {
        "dataset": "egodex",
        "source_root": str(args.input_root),
        "split": args.split,
        "joint_order": ["wrist", *[
            f"{finger.lower()}_{joint.lower()}" for finger, joints in FINGER_JOINTS.items() for joint in joints
        ]],
        "coordinate_frame": "opencv_camera_m",
        "native_mano": False,
        "frame_stride": args.frame_stride,
        "hands": args.hands,
        "min_mean_confidence": args.min_mean_confidence,
        "bbox_margin": args.bbox_margin,
        "episodes": episodes,
        "decoded_frames": frames_written,
        "samples": rows,
    }
    (output_root / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Raw EgoDex split, e.g. .../egodex/test")
    parser.add_argument("--output-root", type=Path, required=True, help="Derived RopeTrack dataset root")
    parser.add_argument("--split", choices=["evaluation", "training"], default="evaluation")
    parser.add_argument("--hands", choices=["both", "left", "right"], default="both")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--episode-limit", type=int, default=0, help="<=0 means all episodes")
    parser.add_argument("--sample-limit", type=int, default=0, help="<=0 means all hand samples")
    parser.add_argument("--min-mean-confidence", type=float, default=0.0)
    parser.add_argument("--bbox-margin", type=float, default=0.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if not 0.0 <= args.min_mean_confidence <= 1.0:
        parser.error("--min-mean-confidence must be in [0, 1]")
    if args.bbox_margin < 0.0:
        parser.error("--bbox-margin must be >= 0")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    return args


def main() -> None:
    meta = export(parse_args())
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
