#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RGB_STREAM_ID = "214-1"
REQUIRED_MASKS = (
    "mask_qa_pass.csv",
    "mask_good_exposure.csv",
    "mask_headset_pose_available.csv",
    "mask_hand_pose_available.csv",
)
OPENPOSE_ORDER = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
SKELETON = tuple(
    (chain[idx], chain[idx + 1])
    for chain in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                  (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for idx in range(4)
)


def clamp_bbox(box, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    out = np.asarray(box, dtype=np.float32).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, width - 1.0)
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, height - 1.0)
    if out[2] <= out[0] or out[3] <= out[1]:
        raise ValueError(f"degenerate clipped bbox: {out.tolist()}")
    return out


def transform_points(transform, points) -> np.ndarray:
    matrix = np.asarray(transform.to_matrix(), dtype=np.float64)
    xyz = np.asarray(points, dtype=np.float64)
    return ((matrix[:3, :3] @ xyz.T).T + matrix[:3, 3]).astype(np.float32)


def camera_intrinsic(calibration) -> np.ndarray:
    fx, fy = map(float, calibration.get_focal_lengths())
    cx, cy = map(float, calibration.get_principal_point())
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def project_fisheye(calibration, points) -> np.ndarray:
    rows = []
    for point in np.asarray(points):
        uv = calibration.project(point.astype(np.float64))
        rows.append([np.nan, np.nan] if uv is None else np.asarray(uv, dtype=np.float32))
    return np.asarray(rows, dtype=np.float32)


def load_mask(path: Path, stream_id: str = RGB_STREAM_ID) -> dict[int, bool]:
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] == stream_id:
                values[int(row["timestamp[ns]"])] = row["mask"].strip().lower() == "true"
    return values


def load_selection(path: Path | None, sequence: str) -> dict[tuple[int, int], dict] | None:
    if path is None:
        return None
    selected = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["sequence"] != sequence:
                continue
            key = (int(row["timestamp_ns"]), int(row["hand_index"]))
            if key in selected:
                raise ValueError(f"duplicate HOT3D selection row: {sequence} {key}")
            selected[key] = row
    return selected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_array(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump([np.asarray(row).tolist() for row in rows], handle, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


def draw_projection(image_rgb, bbox, uv, side: str, sample_id: str):
    import cv2

    canvas = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = np.rint(bbox).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 220, 40), 3)
    for left, right in SKELETON:
        if np.isfinite(uv[[left, right]]).all():
            cv2.line(canvas, tuple(np.rint(uv[left]).astype(int)), tuple(np.rint(uv[right]).astype(int)), (30, 80, 255), 3)
    for point in uv:
        if np.isfinite(point).all():
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 4, (255, 180, 20), -1)
    cv2.putText(canvas, f"GT {side}: {sample_id}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 220, 40), 2)
    return canvas


def prepare(args: argparse.Namespace) -> Path:
    toolkit_python = args.toolkit_root / "hot3d"
    if not toolkit_python.is_dir():
        raise FileNotFoundError(f"HOT3D toolkit Python root missing: {toolkit_python}")
    sys.path.insert(0, str(toolkit_python))

    import cv2
    from data_loaders.AriaDataProvider import AriaDataProvider
    from data_loaders.HandBox2dDataProvider import load_box2d_trajectory_from_csv
    from data_loaders.HeadsetPose3dProvider import load_headset_pose_provider_from_csv
    from data_loaders.loader_hand_poses import Handedness, load_mano_shape_params
    from data_loaders.mano_layer import MANOHandModel
    from data_loaders.ManoHandDataProvider import MANOHandDataProvider
    from projectaria_tools.core.calibration import FISHEYE624
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    metadata = json.loads((args.sequence_root / "metadata.json").read_text())
    if str(metadata.get("headset", metadata.get("headset_type", ""))).lower() != "aria":
        raise ValueError(f"Aria-only exporter received non-Aria metadata: {metadata}")
    if not bool(metadata.get("have_hand_object_pose_gt")):
        raise ValueError("sequence does not declare public hand/object GT")

    stream_id = StreamId(RGB_STREAM_ID)
    device = AriaDataProvider(str(args.sequence_root / "recording.vrs"), str(args.sequence_root / "mps"))
    boxes = load_box2d_trajectory_from_csv(str(args.sequence_root / "box2d_hands.csv"))
    headset = load_headset_pose_provider_from_csv(str(args.sequence_root / "headset_trajectory.csv"))
    mano_model = MANOHandModel(str(args.model_dir), joint_mapper=None)
    mano_path = args.sequence_root / "mano_hand_pose_trajectory.jsonl"
    hands = MANOHandDataProvider(str(mano_path), mano_model)
    mano_betas = np.asarray(load_mano_shape_params(str(mano_path)), dtype=np.float32)
    if mano_betas.shape != (10,):
        raise ValueError(f"expected 10 MANO betas, got {mano_betas.shape}")

    masks = [load_mask(args.sequence_root / "masks" / name) for name in REQUIRED_MASKS]
    timestamps = boxes.get_timestamp_ns_list(stream_id)
    if not timestamps:
        raise ValueError(f"no RGB hand boxes for {args.sequence_root.name}")
    selection = load_selection(args.selection, args.sequence_root.name)
    selected_timestamps = {key[0] for key in selection} if selection is not None else None

    args.output_root.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_root / "images"
    projection_dir = args.output_root / "projection_checks"
    image_dir.mkdir(exist_ok=True)
    projection_dir.mkdir(exist_ok=True)

    records, xyz_rows, vert_rows = [], [], []
    counts = Counter()
    kept_frames = 0
    for source_frame_index, timestamp_ns in enumerate(timestamps):
        if selected_timestamps is not None and timestamp_ns not in selected_timestamps:
            continue
        if not all(mask.get(timestamp_ns, False) for mask in masks):
            continue
        if kept_frames % args.frame_stride:
            kept_frames += 1
            continue
        kept_frames += 1

        bbox_with_dt = boxes.get_bbox_at_timestamp(stream_id, timestamp_ns, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE)
        pose_with_dt = hands.get_pose_at_timestamp(timestamp_ns, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE, 0)
        headset_with_dt = headset.get_pose_at_timestamp(timestamp_ns, TimeQueryOptions.CLOSEST, TimeDomain.TIME_CODE, 0)
        if bbox_with_dt is None or pose_with_dt is None or headset_with_dt is None:
            continue
        image_rgb = device.get_image(timestamp_ns, stream_id)
        if image_rgb is None:
            continue
        image_size = (int(image_rgb.shape[1]), int(image_rgb.shape[0]))
        T_device_camera, calibration = device.get_online_camera_calibration(
            stream_id, timestamp_ns, camera_model=FISHEYE624
        )
        T_world_camera = headset_with_dt.pose3d.T_world_device @ T_device_camera
        T_camera_world = T_world_camera.inverse()
        intrinsic = camera_intrinsic(calibration)
        image_rel = Path("images") / f"{timestamp_ns}.jpg"
        image_path = args.output_root / image_rel
        if not image_path.exists():
            cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])

        for hand_index, handedness, side in (
            (1, Handedness.Right, "right"),
            (0, Handedness.Left, "left"),
        ):
            selected = selection.get((timestamp_ns, hand_index)) if selection is not None else None
            if selection is not None and selected is None:
                continue
            hand_box = bbox_with_dt.box2d_collection.box2ds.get(hand_index)
            hand_pose = pose_with_dt.pose3d_collection.poses.get(handedness)
            if hand_box is None or hand_box.box2d is None or hand_pose is None:
                continue
            if hand_box.visibility_ratio is None or hand_box.visibility_ratio < args.min_visibility:
                continue
            raw_box = hand_box.box2d
            bbox = clamp_bbox([raw_box.left, raw_box.top, raw_box.right, raw_box.bottom], image_size)
            vertices_world = hands.get_hand_mesh_vertices(hand_pose).detach().cpu().numpy()
            joints_world = hands.get_hand_landmarks(hand_pose).detach().cpu().numpy()[OPENPOSE_ORDER]
            vertices_camera = transform_points(T_camera_world, vertices_world)
            joints_camera = transform_points(T_camera_world, joints_world)
            if vertices_camera.shape != (778, 3) or joints_camera.shape != (21, 3):
                raise ValueError(f"decoded GT shape mismatch: {vertices_camera.shape}, {joints_camera.shape}")
            if not np.isfinite(vertices_camera).all() or not np.isfinite(joints_camera).all():
                raise ValueError("non-finite HOT3D GT")

            sample_id = f"{args.sequence_root.name}/{side}/{timestamp_ns}"
            record = {
                "sample_id": sample_id,
                "image_path": image_rel.as_posix(),
                "bbox_xyxy": bbox.tolist(),
                "is_right": side == "right",
                "episode_id": selected["episode_id"] if selected is not None else f"{args.sequence_root.name}/{side}",
                "frame_index": int(selected["frame_index"]) if selected is not None else source_frame_index,
                "source_frame_index": source_frame_index,
                "intrinsic": intrinsic.tolist(),
                "joint_confidence": [float(hand_box.visibility_ratio)] * 21,
                "stream_id": RGB_STREAM_ID,
                "timestamp_ns": timestamp_ns,
                "visibility_ratio": float(hand_box.visibility_ratio),
                "bbox_source": "official_box2d_hands_clamped",
                "camera_model": "FISHEYE624_online_calibration",
                "mano_pose_pca15": [float(value) for value in hand_pose.joint_angles],
                "mano_betas10": mano_betas.tolist(),
                "mano_wrist_world_matrix": np.asarray(hand_pose.wrist_pose.to_matrix()).tolist(),
            }
            if selected is not None:
                record.update({
                    "phase": selected["phase"],
                    "phase_index": int(selected["phase_index"]),
                    "selection_visibility_ratio": float(selected["visibility_ratio"]),
                })
            records.append(record)
            xyz_rows.append(joints_camera)
            vert_rows.append(vertices_camera)
            counts[side] += 1

            if len(records) <= args.projection_count:
                uv = project_fisheye(calibration, joints_camera)
                overlay = draw_projection(image_rgb, bbox, uv, side, sample_id)
                cv2.imwrite(str(projection_dir / f"{len(records):03d}_{side}_{timestamp_ns}.jpg"), overlay)
            if args.limit > 0 and len(records) >= args.limit:
                break
        if args.limit > 0 and len(records) >= args.limit:
            break

    if not records:
        raise RuntimeError("no HOT3D samples survived masks, visibility, and GT checks")
    manifest_path = args.output_root / "evaluation.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    xyz_path = args.output_root / "evaluation_xyz.json"
    verts_path = args.output_root / "evaluation_verts.json"
    write_json_array(xyz_path, xyz_rows)
    write_json_array(verts_path, vert_rows)

    protocol = {
        "dataset": "hot3d",
        "subset": "aria_public_gt",
        "sequence": args.sequence_root.name,
        "headset": "Aria",
        "stream_id": RGB_STREAM_ID,
        "source_images": "raw_vrs_rgb",
        "camera_model": "online_FISHEYE624",
        "manifest_intrinsic": "focal_and_principal_point_only_for_current_pinhole_model_interface",
        "coordinate_frame": "opencv_camera_m",
        "bbox_source": "official_box2d_hands_clamped",
        "masks": list(REQUIRED_MASKS),
        "min_visibility": args.min_visibility,
        "frame_stride_after_masks": args.frame_stride,
        "joint_source": "native_left_right_mano_kinematic_joints_reordered_openpose",
        "mesh_source": "native_left_right_mano",
        "mano_parameters": "manifest_pose_pca15_betas10_wrist_world_matrix",
        "sample_side_order": ["right", "left"],
        "num_samples": len(records),
        "side_counts": dict(counts),
        "sha256": {path.name: file_sha256(path) for path in (manifest_path, xyz_path, verts_path)},
    }
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({key: protocol[key] for key in ("sequence", "num_samples", "side_counts")}, indent=2))
    return args.output_root


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export an Aria public-GT HOT3D smoke set for RopeTrack.")
    parser.add_argument("--sequence-root", type=Path, default=Path("/data/wentao/datasets/hot3d/P0003_c701bd11"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/wentao/ropetrack/processed/hot3d/aria_smoke"))
    parser.add_argument("--toolkit-root", type=Path, default=repo / "third_party" / "hot3d")
    parser.add_argument("--model-dir", type=Path, default=repo / "mano_data")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--projection-count", type=int, default=6)
    parser.add_argument("--selection", type=Path, default=None,
                        help="Optional JSONL of exact sequence/hand_index/timestamp rows for natural episodes.")
    args = parser.parse_args()
    if not 0.0 <= args.min_visibility <= 1.0 or args.frame_stride < 1 or args.limit < 0 or args.projection_count < 0:
        parser.error("visibility must be in [0,1]; stride >=1; limit and projection count >=0")
    return args


if __name__ == "__main__":
    prepare(parse_args())
