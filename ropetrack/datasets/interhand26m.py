#!/usr/bin/env python3
"""Build the RopeTrack InterHand2.6M v1.0 30fps one-view protocol."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATASET_VERSION = "InterHand2.6M v1.0 30fps"
PROTOCOL_NAME = "interhand26m_v1_30fps_oneview_v1"
TRAIN_NAME = "interhand26m_train27k_oneview_v2"
TRAIN_COUNT = 27_000
SMOKE_COUNT = 256
SEED = 20260720
BBOX_MARGIN = 0.25
SIDE_SLICE = {"right": slice(0, 21), "left": slice(21, 42)}
SIDE_ROOT = {"right": 20, "left": 41}
INTERHAND_TO_OPENPOSE = np.asarray(
    [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16],
    dtype=np.int64,
)


def stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def annotation_path(raw_root: Path, split: str, kind: str) -> Path:
    return raw_root / "annotations" / split / f"InterHand2.6M_{split}_{kind}.json"


def subject_splits(path: Path) -> dict[str, set[str]]:
    result = {name: set() for name in ("train", "val", "test")}
    for (split, _), subject in subject_assignments(path).items():
        result[split].add(subject)
    return result


def subject_assignments(path: Path) -> dict[tuple[str, int], str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        subject = fields[0]
        for location in fields[1:-1]:
            split, capture = location.split("/", 1)
            if split not in {"train", "val", "test"} or not capture.startswith("Capture"):
                continue
            key = (split, int(capture.removeprefix("Capture")))
            if key in result and result[key] != subject:
                raise ValueError(f"subject.txt assigns {key} to both {result[key]} and {subject}")
            result[key] = subject
    return result


def frame_group_id(split: str, image: dict) -> str:
    return f"{split}/Capture{image['capture']}/{image['seq_name']}/{int(image['frame_idx']):06d}"


def episode_id(split: str, image: dict) -> str:
    return f"{split}/Capture{image['capture']}/{image['seq_name']}"


def sample_id(candidate: dict, side: str) -> str:
    return (
        f"{candidate['split']}/Capture{candidate['capture_id']}/{candidate['sequence_id']}/"
        f"cam{candidate['camera_id']}/{candidate['frame_id']:06d}/{side}"
    )


def side_candidates(annotation: dict) -> list[str]:
    if not bool(annotation.get("hand_type_valid", 0)):
        return []
    declared = annotation.get("hand_type")
    sides = [declared] if declared in {"right", "left"} else ["right", "left"] if declared == "interacting" else []
    valid = np.asarray(annotation.get("joint_valid", []), dtype=np.float32).reshape(-1)
    if valid.size != 42:
        raise ValueError(f"data JSON joint_valid must contain 42 values, got {valid.shape}")
    return [
        side for side in sides
        if bool(valid[SIDE_ROOT[side]]) and int(np.count_nonzero(valid[SIDE_SLICE[side]])) >= 4
    ]


def select_oneview(raw_root: Path, split: str, capture_subjects: dict[tuple[str, int], str] | None = None) -> tuple[list[dict], dict]:
    path = annotation_path(raw_root, split, "data")
    payload = json.loads(path.read_text(encoding="utf-8"))
    images, annotations = payload["images"], payload["annotations"]
    if len(images) != len(annotations):
        raise ValueError(f"{split} image/annotation count differs: {len(images)} != {len(annotations)}")
    selected = {}
    rejected_hand_type = 0
    for image, annotation in zip(images, annotations, strict=True):
        if int(image["id"]) != int(annotation["image_id"]):
            raise ValueError("data JSON image/annotation order is not one-to-one")
        sides = side_candidates(annotation)
        if not sides:
            rejected_hand_type += 1
            continue
        group = frame_group_id(split, image)
        capture_id = int(image["capture"])
        official_subject = capture_subjects[(split, capture_id)] if capture_subjects is not None else str(image["subject"])
        camera = str(image["camera"])
        rank = stable_digest(PROTOCOL_NAME, group, camera)
        if group not in selected or rank < selected[group][0]:
            selected[group] = (rank, {
                "split": split,
                "image_id": int(image["id"]),
                "image_path": f"images/{split}/{image['file_name']}",
                "width": int(image["width"]),
                "height": int(image["height"]),
                "capture_id": capture_id,
                "subject_id": official_subject,
                "sequence_id": str(image["seq_name"]),
                "camera_id": camera,
                "frame_id": int(image["frame_idx"]),
                "frame_group_id": group,
                "episode_id": episode_id(split, image),
                "hand_type": str(annotation["hand_type"]),
                "hand_type_valid": int(annotation["hand_type_valid"]),
                "annotation_joint_valid": np.asarray(annotation["joint_valid"], dtype=np.int8).reshape(-1).tolist(),
                "candidate_sides": sides,
            })
    candidates = [selected[key][1] for key in sorted(selected)]
    diagnostics = {
        "data_json": str(path),
        "data_json_sha256": file_sha256(path),
        "all_camera_images": len(images),
        "oneview_frames": len(candidates),
        "candidate_single_hand_instances": sum(len(row["candidate_sides"]) for row in candidates),
        "rejected_images_before_oneview_for_invalid_hand_type_or_fewer_than_four_2d_valid_joints": rejected_hand_type,
        "captures": sorted({row["capture_id"] for row in candidates}),
        "subjects": sorted({row["subject_id"] for row in candidates}),
        "sequences": len({row["episode_id"] for row in candidates}),
        "camera_counts": dict(sorted(Counter(row["camera_id"] for row in candidates).items())),
        "selection": "minimum SHA256(protocol, frame_group_id, camera_id) among available cameras",
        "selection_uses_model_error_or_GT_values": False,
        "subject_source": "annotations/subject.txt mapping from (split, capture); data JSON image.subject is not used as project subject_id",
    }
    del payload, images, annotations, selected
    gc.collect()
    return candidates, diagnostics


def validity_vector(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size % 42:
        raise ValueError(f"joint validity does not describe 42 joints: {array.shape}")
    return array.reshape(42, -1).min(axis=1) > 0


def world_to_camera(world_mm: np.ndarray, rotation: np.ndarray, position_mm: np.ndarray) -> np.ndarray:
    return (rotation @ (world_mm.T - position_mm.reshape(3, 1))).T


def project(camera_mm: np.ndarray, focal: np.ndarray, principal: np.ndarray) -> np.ndarray:
    return camera_mm[:, :2] / camera_mm[:, 2:3] * focal.reshape(1, 2) + principal.reshape(1, 2)


def bbox_from_points(points: np.ndarray, valid: np.ndarray, width: int, height: int) -> list[float]:
    xy = np.asarray(points, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(xy).all(axis=1)
    if int(mask.sum()) < 4:
        raise ValueError("fewer than four valid projected joints")
    low, high = xy[mask].min(axis=0), xy[mask].max(axis=0)
    span = high - low
    if float(span.min()) < 4.0:
        raise ValueError(f"projected bbox span below four pixels: {span.tolist()}")
    low -= BBOX_MARGIN * span
    high += BBOX_MARGIN * span
    box = np.asarray([low[0], low[1], high[0], high[1]], dtype=np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, width - 1.0)
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, height - 1.0)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("bbox empty after clipping")
    return box.tolist()


def mano_parameters(entry: dict | None) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    nan_pose = np.full(48, np.nan, dtype=np.float32)
    nan_shape = np.full(10, np.nan, dtype=np.float32)
    nan_trans = np.full(3, np.nan, dtype=np.float32)
    if not entry:
        return False, nan_pose, nan_shape, nan_trans
    pose = np.asarray(entry.get("pose"), dtype=np.float32)
    shape = np.asarray(entry.get("shape"), dtype=np.float32)
    trans = np.asarray(entry.get("trans"), dtype=np.float32)
    valid = pose.shape == (48,) and shape.shape == (10,) and trans.shape == (3,) and all(
        np.isfinite(value).all() for value in (pose, shape, trans)
    )
    return (True, pose, shape, trans) if valid else (False, nan_pose, nan_shape, nan_trans)


def materialize(raw_root: Path, split: str, candidates: list[dict], require_full_train_gt: bool) -> tuple[list[dict], dict]:
    camera = json.loads(annotation_path(raw_root, split, "camera").read_text(encoding="utf-8"))
    joints = json.loads(annotation_path(raw_root, split, "joint_3d").read_text(encoding="utf-8"))
    mano = json.loads(annotation_path(raw_root, split, "MANO_NeuralAnnot").read_text(encoding="utf-8"))
    records, rejected = [], []
    for candidate in candidates:
        capture = str(candidate["capture_id"])
        frame = str(candidate["frame_id"])
        cam = str(candidate["camera_id"])
        try:
            world_entry = joints[capture][frame]
            world = np.asarray(world_entry["world_coord"], dtype=np.float32).reshape(42, 3)
            valid3d = validity_vector(world_entry["joint_valid"])
            cam_entry = camera[capture]
            rotation = np.asarray(cam_entry["camrot"][cam], dtype=np.float32).reshape(3, 3)
            position = np.asarray(cam_entry["campos"][cam], dtype=np.float32).reshape(3)
            focal = np.asarray(cam_entry["focal"][cam], dtype=np.float32).reshape(2)
            principal = np.asarray(cam_entry["princpt"][cam], dtype=np.float32).reshape(2)
            camera_mm = world_to_camera(world, rotation, position)
            projected = project(camera_mm, focal, principal)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"frame_group_id": candidate["frame_group_id"], "error": f"frame metadata: {error}"})
            continue
        built = []
        for side in candidate["candidate_sides"]:
            side_slice = SIDE_SLICE[side]
            native_world = world[side_slice]
            native_camera = camera_mm[side_slice]
            native_projected = projected[side_slice]
            native_valid3d = valid3d[side_slice] & np.isfinite(native_camera).all(axis=1) & (native_camera[:, 2] > 0)
            sid = sample_id(candidate, side)
            try:
                if not native_valid3d[20] or int(native_valid3d.sum()) < 4:
                    raise ValueError("root invalid or fewer than four valid 3D joints")
                bbox = bbox_from_points(native_projected, native_valid3d, candidate["width"], candidate["height"])
                mano_entry = mano.get(capture, {}).get(frame, {}).get(side)
                mano_valid, pose, shape, trans = mano_parameters(mano_entry)
                if require_full_train_gt and (not native_valid3d.all() or not mano_valid):
                    raise ValueError("train27k requires all 21 3D joints and native MANO")
                image = raw_root / candidate["image_path"]
                if not image.is_file():
                    raise FileNotFoundError(image)
            except (FileNotFoundError, ValueError) as error:
                rejected.append({"sample_id": sid, "error": str(error)})
                continue
            order = INTERHAND_TO_OPENPOSE
            annotation_valid = np.asarray(candidate["annotation_joint_valid"], dtype=bool)[side_slice][order]
            valid3d_ordered = native_valid3d[order]
            projected_ordered = native_projected[order]
            projected_in_frame = (
                valid3d_ordered
                & (projected_ordered[:, 0] >= 0.0)
                & (projected_ordered[:, 0] < candidate["width"])
                & (projected_ordered[:, 1] >= 0.0)
                & (projected_ordered[:, 1] < candidate["height"])
            )
            intrinsic = [[float(focal[0]), 0.0, float(principal[0])], [0.0, float(focal[1]), float(principal[1])], [0.0, 0.0, 1.0]]
            row = {
                "sample_id": sid,
                "image_path": candidate["image_path"],
                "split": split,
                "capture_id": candidate["capture_id"],
                "subject_id": candidate["subject_id"],
                "sequence_id": candidate["sequence_id"],
                "camera_id": candidate["camera_id"],
                "frame_id": candidate["frame_id"],
                "frame_index": candidate["frame_id"],
                "frame_group_id": candidate["frame_group_id"],
                "episode_id": candidate["episode_id"],
                "is_right": side == "right",
                "mano_side": side,
                "hand_type": candidate["hand_type"],
                "hand_type_valid": candidate["hand_type_valid"],
                "is_interacting": candidate["hand_type"] == "interacting",
                "bbox_xyxy": bbox,
                "intrinsic": intrinsic,
                "camera_transform_ref": {"capture_id": candidate["capture_id"], "camera_id": candidate["camera_id"]},
                "joint_valid": valid3d_ordered.astype(int).tolist(),
                "annotation_joint_valid": annotation_valid.astype(int).tolist(),
                "projected_in_frame": projected_in_frame.astype(int).tolist(),
                "valid_joint_count": int(valid3d_ordered.sum()),
                "projected_in_frame_joint_count": int(projected_in_frame.sum()),
                "mano_valid": bool(mano_valid),
                "gt_ref": {"capture_id": candidate["capture_id"], "frame_id": candidate["frame_id"], "side": side},
                "bbox_source": "side-specific projected valid world joints; 25 percent x/y margin; clipped to image",
            }
            built.append({
                "row": row,
                "xyz": (native_camera[order] / 1000.0).astype(np.float32),
                "world_mm": native_world[order].astype(np.float32),
                "projected_xy": native_projected[order].astype(np.float32),
                "pose": pose,
                "shape": shape,
                "trans": trans,
                "camrot": rotation,
                "campos_mm": position,
            })
        for record in built:
            record["row"]["paired_hand_count"] = len(built)
        records.extend(built)
    diagnostics = {
        "candidate_frames": len(candidates),
        "accepted_samples": len(records),
        "rejected_samples": len(rejected),
        "rejected_reason_counts": dict(sorted(Counter(row["error"] for row in rejected).items())),
        "rejected": rejected,
        "camera_json_sha256": file_sha256(annotation_path(raw_root, split, "camera")),
        "joint_3d_json_sha256": file_sha256(annotation_path(raw_root, split, "joint_3d")),
        "mano_json_sha256": file_sha256(annotation_path(raw_root, split, "MANO_NeuralAnnot")),
    }
    del camera, joints, mano
    gc.collect()
    return records, diagnostics


def group_records(records: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["row"]["frame_group_id"]].append(record)
    return dict(grouped)


def select_group_balanced(records: list[dict], count: int, seed: int) -> list[dict]:
    groups = group_records(records)
    by_episode_stratum = defaultdict(lambda: defaultdict(list))
    episode_capture = {}
    for frame_group, rows in groups.items():
        row = rows[0]["row"]
        episode = row["episode_id"]
        capture = str(row.get("capture_id", episode.split("/Capture", 1)[1].split("/", 1)[0]))
        sides = "paired" if len(rows) == 2 else row.get("mano_side", row["sample_id"].rsplit("/", 1)[-1])
        stratum = (row.get("camera_id", "unknown"), row.get("hand_type", sides), sides)
        by_episode_stratum[episode][stratum].append((frame_group, rows))
        episode_capture[episode] = capture

    pools = {}
    for episode, strata in by_episode_stratum.items():
        ordered = []
        strata = {
            name: sorted(items, key=lambda item: stable_digest(seed, item[0]))
            for name, items in strata.items()
        }
        for index in range(max(map(len, strata.values()))):
            for name in sorted(strata, key=lambda value: stable_digest(seed, episode, *value)):
                if index < len(strata[name]):
                    ordered.append(strata[name][index])
        pools[episode] = ordered

    by_capture = defaultdict(list)
    for episode, capture in episode_capture.items():
        by_capture[capture].append(episode)
    cursors = Counter()
    capture_counts = Counter()
    episode_counts = Counter()
    remaining_sizes = Counter(len(rows) for items in pools.values() for _, rows in items)
    selected = []

    def can_fill(remaining: int) -> bool:
        singles, pairs = remaining_sizes[1], remaining_sizes[2]
        minimum_singles = max(0, remaining - 2 * pairs)
        if minimum_singles % 2 != remaining % 2:
            minimum_singles += 1
        return minimum_singles <= min(singles, remaining)

    while len(selected) < count:
        captures = [
            capture for capture, episodes in by_capture.items()
            if any(cursors[episode] < len(pools[episode]) for episode in episodes)
        ]
        if not captures:
            break
        capture = min(captures, key=lambda value: (capture_counts[value], stable_digest(seed, "capture", value)))
        episodes = [episode for episode in by_capture[capture] if cursors[episode] < len(pools[episode])]
        episode = min(episodes, key=lambda value: (episode_counts[value], stable_digest(seed, "episode", value)))
        _, rows = pools[episode][cursors[episode]]
        cursors[episode] += 1
        remaining_sizes[len(rows)] -= 1
        remaining = count - len(selected) - len(rows)
        if remaining < 0 or not can_fill(remaining):
            continue
        selected.extend(rows)
        capture_counts[capture] += len(rows)
        episode_counts[episode] += len(rows)
        if len(selected) == count:
            return selected
    raise ValueError(f"could not select exactly {count} instances while keeping frame groups intact; got {len(selected)}")


def capacity_balance_report(records: list[dict], selected: list[dict]) -> dict:
    def counts(values, key):
        return Counter(str(record["row"][key]) for record in values)

    available_capture = counts(records, "capture_id")
    selected_capture = counts(selected, "capture_id")
    available_episode = counts(records, "episode_id")
    selected_episode = counts(selected, "episode_id")
    capture_rows = []
    for capture in sorted(available_capture, key=int):
        capture_rows.append({
            "capture_id": int(capture),
            "available_samples": available_capture[capture],
            "selected_samples": selected_capture[capture],
            "exhausted": selected_capture[capture] == available_capture[capture],
        })
    active_capture_counts = [row["selected_samples"] for row in capture_rows if not row["exhausted"]]
    max_capture = max(active_capture_counts, default=0)
    capture_violations = [
        row["capture_id"] for row in capture_rows
        if not row["exhausted"] and row["selected_samples"] < max_capture - 2
    ]
    episode_violations = []
    for capture in available_capture:
        episodes = [name for name in available_episode if f"/Capture{capture}/" in name]
        active = [selected_episode[name] for name in episodes if selected_episode[name] < available_episode[name]]
        maximum = max(active, default=0)
        episode_violations.extend(
            name for name in episodes
            if selected_episode[name] < available_episode[name] and selected_episode[name] < maximum - 2
        )
    return {
        "definition": "capacity-constrained capture then episode water-fill in single-hand instance counts; tolerance 2 preserves paired frame groups",
        "available_samples": len(records),
        "selected_samples": len(selected),
        "available_episodes": len(available_episode),
        "selected_episodes": sum(value > 0 for value in selected_episode.values()),
        "capture": capture_rows,
        "starved_available_captures": [row["capture_id"] for row in capture_rows if row["available_samples"] and not row["selected_samples"]],
        "underfull_non_exhausted_captures": capture_violations,
        "underfull_non_exhausted_episodes": sorted(episode_violations),
        "status": "PASS" if not capture_violations and not episode_violations and all(row["selected_samples"] for row in capture_rows) else "FAIL",
    }


def internal_episode_split(records: list[dict], fraction: float = 0.1, seed: int = 0) -> dict:
    episodes = np.asarray(sorted({record["row"]["episode_id"] for record in records}))
    rng = np.random.default_rng(seed)
    val_count = max(1, int(math.ceil(len(episodes) * fraction)))
    val_episodes = set(episodes[rng.permutation(len(episodes))[:val_count]].tolist())
    train_ids = [record["row"]["sample_id"] for record in records if record["row"]["episode_id"] not in val_episodes]
    val_ids = [record["row"]["sample_id"] for record in records if record["row"]["episode_id"] in val_episodes]
    return {
        "rule": "complete episode split; numpy RNG seed 0; ceil 10 percent",
        "train_episodes": sorted(set(episodes.tolist()) - val_episodes),
        "validation_episodes": sorted(val_episodes),
        "episode_intersection": [],
        "num_train": len(train_ids),
        "num_validation": len(val_ids),
        "train_sample_id_sha256": ids_sha256(train_ids),
        "validation_sample_id_sha256": ids_sha256(val_ids),
    }


def write_json_array(path: Path, values) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for index, value in enumerate(values):
            if index:
                handle.write(",")
            handle.write(json.dumps(np.asarray(value).tolist(), separators=(",", ":")))
        handle.write("]\n")


def distribution(records: list[dict]) -> dict:
    rows = [record["row"] for record in records]
    bbox_size = [max(row["bbox_xyxy"][2] - row["bbox_xyxy"][0], row["bbox_xyxy"][3] - row["bbox_xyxy"][1]) for row in rows]
    return {
        "samples": len(rows),
        "frames": len({row["frame_group_id"] for row in rows}),
        "episodes": len({row["episode_id"] for row in rows}),
        "subjects": sorted({row["subject_id"] for row in rows}),
        "side": dict(sorted(Counter("right" if row["is_right"] else "left" for row in rows).items())),
        "hand_type": dict(sorted(Counter(row["hand_type"] for row in rows).items())),
        "capture": dict(sorted(Counter(str(row["capture_id"]) for row in rows).items())),
        "camera": dict(sorted(Counter(row["camera_id"] for row in rows).items())),
        "valid_joint_count": dict(sorted(Counter(str(row["valid_joint_count"]) for row in rows).items())),
        "projected_in_frame_joint_count": dict(sorted(Counter(str(row["projected_in_frame_joint_count"]) for row in rows).items())),
        "mano_valid": dict(sorted(Counter(str(row["mano_valid"]).lower() for row in rows).items())),
        "bbox_size_px": {
            "min": float(np.min(bbox_size)), "median": float(np.median(bbox_size)), "max": float(np.max(bbox_size)),
        } if bbox_size else {},
    }


def export_subset(raw_root: Path, output_root: Path, records: list[dict], split_file: str, selection: dict, diagnostics: dict) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = output_root / f"{split_file}.jsonl"
    xyz = output_root / f"{split_file}_xyz.json"
    mano = output_root / f"{split_file}_mano.npz"
    rejected = output_root / "rejected_samples.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record["row"], separators=(",", ":")) + "\n")
    write_json_array(xyz, (record["xyz"] for record in records))
    np.savez(
        mano,
        sample_id=np.asarray([record["row"]["sample_id"] for record in records]),
        pose=np.asarray([record["pose"] for record in records], dtype=np.float32),
        shape=np.asarray([record["shape"] for record in records], dtype=np.float32),
        trans_world_m=np.asarray([record["trans"] for record in records], dtype=np.float32),
        camrot=np.asarray([record["camrot"] for record in records], dtype=np.float32),
        campos_world_mm=np.asarray([record["campos_mm"] for record in records], dtype=np.float32),
        world_joints_mm=np.asarray([record["world_mm"] for record in records], dtype=np.float32),
        projected_xy=np.asarray([record["projected_xy"] for record in records], dtype=np.float32),
        joint_valid=np.asarray([record["row"]["joint_valid"] for record in records], dtype=bool),
        annotation_joint_valid=np.asarray([record["row"]["annotation_joint_valid"] for record in records], dtype=bool),
        projected_in_frame=np.asarray([record["row"]["projected_in_frame"] for record in records], dtype=bool),
        mano_valid=np.asarray([record["row"]["mano_valid"] for record in records], dtype=bool),
        is_right=np.asarray([record["row"]["is_right"] for record in records], dtype=bool),
    )
    with rejected.open("w", encoding="utf-8") as handle:
        for row in diagnostics.pop("rejected", []):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    internal = internal_episode_split(records) if split_file == "training" else None
    protocol = {
        "dataset": DATASET_VERSION,
        "project_protocol": PROTOCOL_NAME,
        "split": records[0]["row"]["split"] if records else None,
        "raw_root": str(raw_root),
        "raw_data_policy": "read only; canonical images/ and annotations/ only; RGB is referenced, never copied",
        "num_samples": len(records),
        "distribution": distribution(records),
        "selection": selection,
        "internal_validation": internal,
        "sample_id": "split/Capture/sequence/camera/frame/side",
        "frame_group_id": "split/Capture/sequence/frame; paired hands stay together",
        "episode_id": "split/Capture/sequence",
        "units": "camera metres after official world-mm to camera transform",
        "joint_order": "RopeTrack OpenPose: wrist, thumb/index/middle/ring/pinky MCP/PIP/DIP/tip",
        "joint_mapping_from_official": INTERHAND_TO_OPENPOSE.tolist(),
        "bbox": {
            "source": "per-side valid projected official world joints",
            "margin_fraction_per_axis": BBOX_MARGIN,
            "clip": "image width/height from data JSON (expected 334x512)",
            "invalid": "fewer than four valid projected points, span below four pixels, empty after clipping",
            "model_output_used": False,
        },
        "mano": {
            "source": "official NeuralAnnot world-coordinate parameters",
            "pose": "48 full axis-angle values minus MANO mean pose",
            "shape": 10,
            "translation": "3 world-coordinate metres",
            "flat_hand_mean": False,
            "mesh_valid_only_where_mano_valid": True,
        },
        "input": "original RGB only; no depth, mask70, synthetic occlusion, or copied RGB",
        "visibility_boundary": "no native occlusion flag is claimed; projected_in_frame is only a geometric image-boundary proxy",
        "rope": "GT-derived ideal normalized five-rope geometry; not no-GT RGB inference or a validated physical sensor",
        "diagnostics": diagnostics,
        "sha256": {
            "manifest": file_sha256(manifest),
            "sample_ids": ids_sha256(record["row"]["sample_id"] for record in records),
            "xyz": file_sha256(xyz),
            "mano": file_sha256(mano),
            "rejected": file_sha256(rejected),
        },
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def write_test_candidates(output_root: Path, candidates: list[dict], diagnostics: dict) -> dict:
    path = output_root / "test_candidates.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in candidates:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    candidate_ids = [sample_id(row, side) for row in candidates for side in row["candidate_sides"]]
    payload = {
        "dataset": DATASET_VERSION,
        "project_protocol": PROTOCOL_NAME,
        "status": "pre-GT test index frozen from data JSON only",
        "test_joint_3d_or_MANO_values_read": 0,
        "candidate_frame_count": len(candidates),
        "candidate_sample_count": len(candidate_ids),
        "candidate_manifest": str(path),
        "candidate_manifest_sha256": file_sha256(path),
        "candidate_sample_id_sha256": ids_sha256(candidate_ids),
        "post_freeze_rule": "read test joint/MANO GT once; apply predeclared joint/root/MANO-flag/bbox validity rules; accepted IDs are an ordered subset",
        "diagnostics": diagnostics,
    }
    protocol_path = output_root / "test_candidate_protocol.json"
    protocol_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def require_test_freeze(path: Path | None, candidate_protocol: Path) -> dict:
    if path is None or not path.is_file():
        raise PermissionError("InterHand test GT export requires --test-freeze-file")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_protocol.read_text(encoding="utf-8"))
    required = {"status": "frozen", "dataset": "interhand26m_v1_30fps", "test_score_reads_before_freeze": 0}
    mismatch = {key: (freeze.get(key), value) for key, value in required.items() if freeze.get(key) != value}
    if mismatch or freeze.get("test_candidates_sha256") != candidate["candidate_manifest_sha256"]:
        raise PermissionError(f"invalid InterHand test freeze: {mismatch}")
    return {"path": str(path), "sha256": file_sha256(path)}


def overlap_report(exports: dict[str, list[dict]]) -> dict:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    def values(split, field):
        rows = [record["row"] for record in exports.get(split, [])]
        if field == "sample":
            return {f"{row['subject_id']}/Capture{row['capture_id']}/{row['sequence_id']}/{row['camera_id']}/{row['frame_id']}/{row['mano_side']}" for row in rows}
        if field == "frame":
            return {f"{row['subject_id']}/Capture{row['capture_id']}/{row['sequence_id']}/{row['frame_id']}" for row in rows}
        if field == "sequence":
            return {f"{row['subject_id']}/Capture{row['capture_id']}/{row['sequence_id']}" for row in rows}
        return {row["subject_id"] for row in rows}
    return {
        field: {
            f"{left}_{right}": sorted(values(left, field) & values(right, field))
            for left, right in pairs
        }
        for field in ("sample", "frame", "sequence", "subject")
    }


def export_all(args: argparse.Namespace) -> Path:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("derived output must not live inside the raw InterHand tree")
    output_root.mkdir(parents=True, exist_ok=True)
    requested = {value.strip() for value in args.splits.split(",") if value.strip()}
    unknown = requested - {"train", "val", "test_index", "test"}
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    subjects = subject_splits(raw_root / "annotations" / "subject.txt")
    capture_subjects = subject_assignments(raw_root / "annotations" / "subject.txt")
    held_out_subjects = subjects["val"] | subjects["test"]
    exports: dict[str, list[dict]] = {}
    split_report_path = output_root / "split_report.json"
    split_report = json.loads(split_report_path.read_text(encoding="utf-8")) if split_report_path.exists() else {
        "dataset": DATASET_VERSION,
        "project_protocol": PROTOCOL_NAME,
        "official_subjects_from_subject_txt": {key: sorted(value) for key, value in subjects.items()},
        "train_subjects_excluded_for_project_subject_disjointness": sorted(subjects["train"] & held_out_subjects),
        "splits": {},
    }
    if "test_index" in requested:
        candidates, diagnostics = select_oneview(raw_root, "test", capture_subjects)
        split_report["splits"]["test_index"] = write_test_candidates(output_root, candidates, diagnostics)
    if "val" in requested:
        candidates, oneview = select_oneview(raw_root, "val", capture_subjects)
        records, diagnostics = materialize(raw_root, "val", candidates, False)
        exports["val"] = records
        split_report["splits"]["val"] = export_subset(
            raw_root, output_root / "val", records, "evaluation",
            {"name": f"{PROTOCOL_NAME}_val", "oneview": oneview, "uses_model_error": False}, diagnostics,
        )
    if "train" in requested:
        candidates, oneview = select_oneview(raw_root, "train", capture_subjects)
        candidates = [row for row in candidates if row["subject_id"] not in held_out_subjects]
        records, diagnostics = materialize(raw_root, "train", candidates, True)
        selected = select_group_balanced(records, args.train_count, args.seed)
        capacity = capacity_balance_report(records, selected)
        if capacity["status"] != "PASS":
            raise ValueError(f"train selection capacity balance failed: {capacity}")
        exports["train"] = selected
        split_report["splits"]["train"] = export_subset(
            raw_root, output_root / "train27k", selected, "training",
            {
                "name": TRAIN_NAME,
                "rule": "capacity-constrained capture then episode water-fill; camera/hand-type/side strata within episode; complete frame groups",
                "seed": args.seed,
                "count": args.train_count,
                "official_train_only": True,
                "excluded_subjects_seen_in_official_val_or_test": sorted(subjects["train"] & held_out_subjects),
                "uses_val_or_test_error": False,
                "oneview": oneview,
                "capacity_balance": capacity,
            }, diagnostics,
        )
        smoke = select_group_balanced(
            sorted(selected, key=lambda record: stable_digest(args.seed, "smoke", record["row"]["frame_group_id"])),
            args.smoke_count,
            args.seed + 1,
        )
        split_report["splits"]["smoke"] = export_subset(
            raw_root, output_root / "smoke", smoke, "training",
            {"name": f"{TRAIN_NAME}_smoke", "source": "train27k only", "count": args.smoke_count},
            {"candidate_frames": len({row["row"]["frame_group_id"] for row in smoke}), "accepted_samples": len(smoke), "rejected_samples": 0, "rejected_reason_counts": {}},
        )
    if "test" in requested:
        candidate_protocol = output_root / "test_candidate_protocol.json"
        freeze = require_test_freeze(args.test_freeze_file, candidate_protocol)
        with (output_root / "test_candidates.jsonl").open(encoding="utf-8") as handle:
            candidates = [json.loads(line) for line in handle if line.strip()]
        records, diagnostics = materialize(raw_root, "test", candidates, False)
        exports["test"] = records
        split_report["splits"]["test"] = export_subset(
            raw_root, output_root / "test", records, "evaluation",
            {
                "name": f"{PROTOCOL_NAME}_test",
                "candidate_manifest_sha256": file_sha256(output_root / "test_candidates.jsonl"),
                "test_access_freeze": freeze,
                "uses_model_error": False,
            }, diagnostics,
        )
    manifest_exports = {}
    for name, directory, split_file in (
        ("train", output_root / "train27k", "training"),
        ("val", output_root / "val", "evaluation"),
        ("test", output_root / "test", "evaluation"),
    ):
        manifest = directory / f"{split_file}.jsonl"
        if manifest.is_file() and name not in exports:
            with manifest.open(encoding="utf-8") as handle:
                manifest_exports[name] = [{"row": json.loads(line)} for line in handle if line.strip()]
    manifest_exports.update(exports)
    overlap = overlap_report(manifest_exports)
    split_report["overlap"] = overlap
    split_report["all_required_overlap_zero"] = not any(
        values for by_pair in overlap.values() for values in by_pair.values()
    )
    split_report["test_boundary"] = {
        "pre_freeze_allowed": "data JSON, filenames, camera metadata, hand_type/joint_valid, ID-only one-view selection",
        "pre_freeze_forbidden": "test joint_3d values, MANO values, rope, prediction error, scores",
        "final_sample_ids": "pre-frozen candidate IDs filtered once after freeze by predeclared GT/MANO/bbox validity",
    }
    split_report_path.write_text(json.dumps(split_report, indent=2) + "\n", encoding="utf-8")
    root_protocol = {
        "dataset": DATASET_VERSION,
        "project_protocol": PROTOCOL_NAME,
        "raw_root": str(raw_root),
        "processed_root": str(output_root),
        "split_report": str(split_report_path),
        "raw_input_roots": [str(raw_root / "images"), str(raw_root / "annotations")],
        "downloads_used": False,
        "test_policy": "candidate IDs before freeze; GT/rope export and score once after freeze",
    }
    (output_root / "protocol.json").write_text(json.dumps(root_protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "splits": sorted(requested), "overlap_zero": split_report["all_required_overlap_zero"]}, indent=2))
    return output_root


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test_index", help="Comma separated: train,val,test_index,test")
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    parser.add_argument("--smoke-count", type=int, default=SMOKE_COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.train_count < 2 or args.smoke_count < 2 or args.smoke_count > args.train_count:
        parser.error("require 2 <= smoke-count <= train-count")
    return args


def main(argv=None):
    return export_all(parse_args(argv))


if __name__ == "__main__":
    main()
