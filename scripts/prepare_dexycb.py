#!/usr/bin/env python3
"""Audit official DexYCB S1 and export manifest-backed RopeTrack subsets.

The raw dataset is read only.  Official test labels are inaccessible unless a
separate recipe-freeze JSON is supplied and passes the guard below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


S1_SUBJECTS = {
    "train": (
        "20200709-subject-01", "20200813-subject-02", "20200820-subject-03",
        "20200903-subject-04", "20200908-subject-05", "20200918-subject-06",
        "20201022-subject-10",
    ),
    "val": ("20200928-subject-07",),
    "test": ("20201002-subject-08", "20201015-subject-09"),
}
S1_COUNTS = {"train": 407_088, "val": 58_592, "test": 116_288}
SERIALS = (
    "836212060125", "839512060362", "840412060917", "841412060263",
    "932122060857", "932122060861", "932122061900", "932122062010",
)
TOOLKIT_COMMIT = "64551b001d360ad83bc383157a559ec248fb9100"
SELECTION_SEED = 20260720
TRAIN27K_COUNT = 27_000
SMOKE_COUNT = 256
BBOX_MARGIN = 0.25
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480


class TupleSafeLoader(yaml.SafeLoader):
    pass


TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


@dataclass(frozen=True)
class SyncFrame:
    split: str
    subject_id: str
    sequence_id: str
    frame_index: int
    serials: tuple[str, ...]
    mano_calibration_id: str
    is_right: bool = True

    @property
    def episode_id(self) -> str:
        return f"{self.subject_id}/{self.sequence_id}"

    def sample_id(self, serial: str) -> str:
        return f"{self.episode_id}/{serial}/{self.frame_index:06d}"


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


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=TupleSafeLoader)


def intrinsics(raw_root: Path) -> dict[str, list[list[float]]]:
    result = {}
    for serial in SERIALS:
        path = raw_root / "calibration" / "intrinsics" / f"{serial}_640x480.yml"
        color = load_yaml(path)["color"]
        result[serial] = [
            [float(color["fx"]), 0.0, float(color["ppx"])],
            [0.0, float(color["fy"]), float(color["ppy"])],
            [0.0, 0.0, 1.0],
        ]
    return result


def enumerate_s1(raw_root: Path) -> dict[str, list[SyncFrame]]:
    result: dict[str, list[SyncFrame]] = {name: [] for name in S1_SUBJECTS}
    for split, subjects in S1_SUBJECTS.items():
        for subject in subjects:
            sequence_paths = sorted(path for path in (raw_root / subject).iterdir() if path.is_dir())
            for sequence_path in sequence_paths:
                meta = load_yaml(sequence_path / "meta.yml")
                serials = tuple(map(str, meta["serials"]))
                if set(serials) != set(SERIALS) or len(serials) != len(SERIALS):
                    raise ValueError(f"unexpected cameras in {sequence_path}: {serials}")
                mano_calib = tuple(map(str, meta["mano_calib"]))
                mano_sides = tuple(map(str, meta["mano_sides"]))
                if len(mano_calib) != 1 or len(mano_sides) != 1 or mano_sides[0] not in {"right", "left"}:
                    raise ValueError(f"expected one declared left/right hand in {sequence_path}")
                for frame_index in range(int(meta["num_frames"])):
                    result[split].append(SyncFrame(
                        split, subject, sequence_path.name, frame_index, serials, mano_calib[0],
                        mano_sides[0] == "right",
                    ))
    return result


def all_sample_ids(frames: list[SyncFrame]) -> list[str]:
    return [frame.sample_id(serial) for frame in frames for serial in frame.serials]


def select_balanced_views(frames: list[SyncFrame], count: int, seed: int) -> list[tuple[SyncFrame, str]]:
    """Round-robin episodes, with one globally balanced camera per sync frame."""
    by_episode: dict[str, list[SyncFrame]] = defaultdict(list)
    for frame in frames:
        by_episode[frame.episode_id].append(frame)
    pools = {
        episode: sorted(rows, key=lambda row: stable_digest(seed, row.episode_id, row.frame_index))
        for episode, rows in by_episode.items()
    }
    episodes = sorted(pools)
    cursor = Counter()
    camera_counts = Counter()
    selected: list[tuple[SyncFrame, str]] = []
    while len(selected) < count:
        before = len(selected)
        for episode in episodes:
            if cursor[episode] >= len(pools[episode]):
                continue
            frame = pools[episode][cursor[episode]]
            cursor[episode] += 1
            minimum = min(camera_counts[serial] for serial in frame.serials)
            candidates = [serial for serial in frame.serials if camera_counts[serial] == minimum]
            serial = min(candidates, key=lambda value: stable_digest(seed, frame.episode_id, frame.frame_index, value))
            selected.append((frame, serial))
            camera_counts[serial] += 1
            if len(selected) == count:
                return selected
        if len(selected) == before:
            break
    raise ValueError(f"requested {count} samples but only selected {len(selected)}")


def valid_hand_label(raw_root: Path, frame: SyncFrame, serial: str) -> bool:
    _, label_path = paths_for(raw_root, frame, serial)
    with np.load(label_path) as label:
        joint_3d = np.asarray(label["joint_3d"], dtype=np.float32)
        pose_m = np.asarray(label["pose_m"], dtype=np.float32)
    return bool(
        np.isfinite(joint_3d).all()
        and np.isfinite(pose_m).all()
        and not np.all(joint_3d == -1)
        and not np.all(pose_m == 0)
    )


def select_balanced_valid_views(
    raw_root: Path, frames: list[SyncFrame], count: int, seed: int
) -> tuple[list[tuple[SyncFrame, str]], dict]:
    """Select exact valid rows while preserving round-robin episode balance."""
    by_episode: dict[str, list[SyncFrame]] = defaultdict(list)
    for frame in frames:
        by_episode[frame.episode_id].append(frame)
    pools = {
        episode: sorted(rows, key=lambda row: stable_digest(seed, row.episode_id, row.frame_index))
        for episode, rows in by_episode.items()
    }
    episodes = sorted(pools)
    cursor = Counter()
    camera_counts = Counter()
    valid_counts = Counter()
    rejected_invalid = 0
    selected: list[tuple[SyncFrame, str]] = []
    while len(selected) < count:
        before = len(selected)
        for episode in episodes:
            while cursor[episode] < len(pools[episode]):
                frame = pools[episode][cursor[episode]]
                cursor[episode] += 1
                minimum = min(camera_counts[serial] for serial in frame.serials)
                candidates = [serial for serial in frame.serials if camera_counts[serial] == minimum]
                serial = min(candidates, key=lambda value: stable_digest(
                    seed, frame.episode_id, frame.frame_index, value
                ))
                if not valid_hand_label(raw_root, frame, serial):
                    rejected_invalid += 1
                    continue
                selected.append((frame, serial))
                camera_counts[serial] += 1
                valid_counts[episode] += 1
                break
            if len(selected) == count:
                return selected, {
                    "invalid_candidates_skipped": rejected_invalid,
                    "episodes_with_selected_rows": len(valid_counts),
                    "episodes_without_selected_rows": sorted(set(episodes) - set(valid_counts)),
                }
        if len(selected) == before:
            break
    raise ValueError(f"requested {count} valid samples but only selected {len(selected)}")


def internal_episode_split(rows: list[tuple[SyncFrame, str]], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    episodes = np.asarray(sorted({frame.episode_id for frame, _ in rows}))
    rng = np.random.default_rng(seed)
    val_count = max(1, int(math.ceil(len(episodes) * val_fraction)))
    val = set(episodes[rng.permutation(len(episodes))[:val_count]].tolist())
    return sorted(set(episodes.tolist()) - val), sorted(val)


def bbox_from_joints(joint_2d: np.ndarray, margin: float = BBOX_MARGIN) -> list[float]:
    joints = np.asarray(joint_2d, dtype=np.float32).reshape(-1, 2)
    valid = np.isfinite(joints).all(axis=1)
    if int(valid.sum()) < 4:
        raise ValueError("fewer than four finite 2D joints")
    x1, y1 = joints[valid].min(axis=0)
    x2, y2 = joints[valid].max(axis=0)
    width, height = float(x2 - x1), float(y2 - y1)
    if width < 4.0 or height < 4.0:
        raise ValueError(f"degenerate bbox {width}x{height}")
    x1 = max(0.0, float(x1 - margin * width))
    y1 = max(0.0, float(y1 - margin * height))
    x2 = min(float(IMAGE_WIDTH - 1), float(x2 + margin * width))
    y2 = min(float(IMAGE_HEIGHT - 1), float(y2 + margin * height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox empty after clipping")
    return [x1, y1, x2, y2]


def mano_betas(raw_root: Path, calibration_id: str) -> np.ndarray:
    path = raw_root / "calibration" / f"mano_{calibration_id}" / "mano.yml"
    betas = np.asarray(load_yaml(path)["betas"], dtype=np.float32)
    if betas.shape != (10,):
        raise ValueError(f"invalid MANO betas in {path}: {betas.shape}")
    return betas


def paths_for(raw_root: Path, frame: SyncFrame, serial: str) -> tuple[Path, Path]:
    directory = raw_root / frame.subject_id / frame.sequence_id / serial
    return directory / f"color_{frame.frame_index:06d}.jpg", directory / f"labels_{frame.frame_index:06d}.npz"


def manifest_row(raw_root: Path, frame: SyncFrame, serial: str, k: list[list[float]]) -> tuple[dict, np.ndarray, np.ndarray]:
    image_path, label_path = paths_for(raw_root, frame, serial)
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"missing raw sample {frame.sample_id(serial)}")
    with np.load(label_path) as label:
        joint_2d = np.asarray(label["joint_2d"], dtype=np.float32).reshape(21, 2)
        joint_3d = np.asarray(label["joint_3d"], dtype=np.float32).reshape(21, 3)
        pose_m = np.asarray(label["pose_m"], dtype=np.float32).reshape(51)
        hand_pixels = int(np.count_nonzero(np.asarray(label["seg"]) == 255))
    if not np.isfinite(joint_3d).all() or not np.isfinite(pose_m).all():
        raise ValueError("non-finite 3D/MANO label")
    if np.all(joint_3d == -1) or np.all(pose_m == 0):
        raise ValueError("official invalid/no-visible-hand sentinel")
    row = {
        "sample_id": frame.sample_id(serial),
        "image_path": image_path.relative_to(raw_root).as_posix(),
        "bbox_xyxy": bbox_from_joints(joint_2d),
        "is_right": frame.is_right,
        "mano_side": "right" if frame.is_right else "left",
        "subject_id": frame.subject_id,
        "sequence_id": frame.sequence_id,
        "episode_id": frame.episode_id,
        "frame_index": frame.frame_index,
        "camera_serial": serial,
        "intrinsics": k,
        "label_path": label_path.relative_to(raw_root).as_posix(),
        "mano_calibration_id": frame.mano_calibration_id,
        "split": frame.split,
        "bbox_source": "official joint_2d; 25 percent margin; clipped to 640x480",
        "hand_segmentation_pixels": hand_pixels,
        "visibility_proxy": "official seg==255 visible hand pixel count",
        "root_depth_m": float(joint_3d[0, 2]),
    }
    return row, joint_3d, pose_m


def write_json_array(path: Path, rows: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for index, row in enumerate(rows):
            if index:
                handle.write(",")
            handle.write(json.dumps(np.asarray(row).tolist(), separators=(",", ":")))
        handle.write("]\n")


def export_subset(
    raw_root: Path,
    output_root: Path,
    selected: list[tuple[SyncFrame, str]],
    split_file: str,
    k_by_serial: dict[str, list[list[float]]],
    selection: dict,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / f"{split_file}.jsonl"
    xyz_path = output_root / f"{split_file}_xyz.json"
    pose_path = output_root / f"{split_file}_mano.npz"
    rejected_path = output_root / "rejected_samples.jsonl"
    rows, xyz_rows, pose_rows, beta_rows, rejected = [], [], [], [], []
    beta_cache: dict[str, np.ndarray] = {}
    for frame, serial in selected:
        try:
            row, xyz, pose = manifest_row(raw_root, frame, serial, k_by_serial[serial])
        except (FileNotFoundError, KeyError, ValueError) as error:
            rejected.append({"sample_id": frame.sample_id(serial), "error": str(error)})
            continue
        if frame.mano_calibration_id not in beta_cache:
            beta_cache[frame.mano_calibration_id] = mano_betas(raw_root, frame.mano_calibration_id)
        rows.append(row)
        xyz_rows.append(xyz)
        pose_rows.append(pose)
        beta_rows.append(beta_cache[frame.mano_calibration_id])
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    write_json_array(xyz_path, xyz_rows)
    pose = np.asarray(pose_rows, dtype=np.float32)
    np.savez(
        pose_path,
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        pose_m=pose,
        global_orient=pose[:, :3],
        articulated_pose_pca=pose[:, 3:48],
        translation=pose[:, 48:51],
        betas=np.asarray(beta_rows, dtype=np.float32),
        is_right=np.asarray([row["is_right"] for row in rows], dtype=bool),
    )
    protocol = {
        "dataset": "DexYCB",
        "official_setup": "s1",
        "split": rows[0]["split"] if rows else None,
        "raw_root": str(raw_root),
        "raw_data_policy": "read_only; no RGB, depth, labels, models, or BOP resources copied",
        "num_requested": len(selected),
        "num_samples": len(rows),
        "num_rejected": len(rejected),
        "evaluation_population": "valid hand-pose rows only, matching the official HPE evaluator's sentinel skip",
        "official_sentinel": "joint_3d all -1 and pose_m all 0 means no visible/annotated hand",
        "rejected_reason_counts": dict(sorted(Counter(row["error"] for row in rejected).items())),
        "selection": selection,
        "sample_id": "subject/sequence/camera_serial/frame_index",
        "episode_id": "subject/sequence; camera deliberately excluded",
        "units": "metres",
        "coordinate_frame": "per-camera OpenCV frame; +x right, +y down, +z forward",
        "joint_order": "wrist then thumb/index/middle/ring/little MCP,PIP,DIP,tip; RopeTrack OpenPose order",
        "mano": {
            "pose_m": "3 global axis-angle + 45 articulated PCA coefficients + 3 translation",
            "flat_hand_mean": False,
            "use_pca": True,
            "num_pca_comps": 45,
            "betas": "10 values from calibration/mano_<sequence mano_calib>/mano.yml",
            "tip_vertex_ids": {
                "right": [745, 317, 444, 556, 673],
                "left": [745, 317, 445, 556, 673],
            },
        },
        "bbox": {
            "source": "official joint_2d",
            "margin_fraction_per_side": BBOX_MARGIN,
            "clip_xyxy": [0, 0, IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1],
            "invalid": "fewer than 4 finite joints, span below 4 px, empty after clipping",
            "model_error_used": False,
        },
        "input": "original RGB only; natural object occlusion retained; no depth, mask, mask70, or synthetic occlusion",
        "rope": "GT-derived ideal five-rope observation; not a validated physical sensor",
        "sha256": {
            "manifest": file_sha256(manifest_path),
            "sample_ids": ids_sha256(row["sample_id"] for row in rows),
            "xyz": file_sha256(xyz_path),
            "mano": file_sha256(pose_path),
            "rejected": file_sha256(rejected_path),
        },
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def verify_toolkit(toolkit_root: Path) -> dict:
    dataset_source = toolkit_root / "dex_ycb_toolkit" / "dex_ycb.py"
    hpe_source = toolkit_root / "dex_ycb_toolkit" / "hpe_eval.py"
    grasp_source = toolkit_root / "dex_ycb_toolkit" / "grasp_eval.py"
    text = dataset_source.read_text(encoding="utf-8")
    required = [
        "subject_ind = [0, 1, 2, 3, 4, 5, 9]",
        "subject_ind = [6]",
        "subject_ind = [7, 8]",
        "serial_ind = [0, 1, 2, 3, 4, 5, 6, 7]",
        "sequence_ind = list(range(100))",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise ValueError(f"official toolkit S1 source changed or mismatched: {missing}")
    commit = subprocess.run(
        ["git", "-C", str(toolkit_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != TOOLKIT_COMMIT:
        raise ValueError(f"official toolkit commit {commit} != pinned {TOOLKIT_COMMIT}")
    return {
        "repository": "https://github.com/NVlabs/dex-ycb-toolkit",
        "commit": TOOLKIT_COMMIT,
        "dataset_source_sha256": file_sha256(dataset_source),
        "hpe_source_sha256": file_sha256(hpe_source),
        "grasp_source_sha256": file_sha256(grasp_source),
        "verified_literals": required,
    }


def expected_bop_keyframes(frames: list[SyncFrame]) -> set[tuple[int, int]]:
    sequence_indices = {}
    result = set()
    for frame in frames:
        sequence_index = sequence_indices.setdefault(frame.episode_id, len(sequence_indices))
        if frame.frame_index % 4 != 0:
            continue
        for camera_index, _ in enumerate(frame.serials):
            result.add((sequence_index * len(SERIALS) + camera_index, frame.frame_index))
    return result


def verify_bop(bop_manifest: Path, toolkit_keyframes: set[tuple[int, int]]) -> dict:
    targets = json.loads(bop_manifest.read_text(encoding="utf-8"))
    images = {(int(row["scene_id"]), int(row["im_id"])) for row in targets}
    extras = sorted(images - toolkit_keyframes)
    omitted = sorted(toolkit_keyframes - images)
    if extras:
        raise ValueError(
            f"BOP S1 target images are not a subset of official every-fourth keyframes: {extras[:5]}"
        )
    omission_details = []
    for scene_id, frame_index in omitted:
        info_path = bop_manifest.parent / "test" / f"{scene_id:06d}" / "scene_gt_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))[str(frame_index)]
        maximum = max(float(row["visib_fract"]) for row in info)
        omission_details.append({
            "scene_id": scene_id,
            "im_id": frame_index,
            "max_object_visible_fraction": maximum,
        })
    if not omission_details or any(row["max_object_visible_fraction"] >= 0.1 for row in omission_details):
        raise ValueError(f"BOP omissions are not explained by the BOP19 10 percent visibility gate: {omission_details}")
    return {
        "path": str(bop_manifest),
        "sha256": file_sha256(bop_manifest),
        "target_rows": len(targets),
        "unique_scene_image_targets": len(images),
        "official_toolkit_every_fourth_keyframes": len(toolkit_keyframes),
        "omitted_keyframes_without_10pct_visible_object": omission_details,
        "crosscheck": "BOP target images are a strict subset of official frame_index modulo 4 keyframes; every omitted image has all object visib_fract below 0.1",
        "boundary": "BOP target manifest independently checks test images only, not train/val",
    }


def audit_report(frames_by_split: dict[str, list[SyncFrame]], audit_summary: Path, toolkit_root: Path, bop_manifest: Path) -> dict:
    audit = json.loads(audit_summary.read_text(encoding="utf-8"))
    subjects_by_split = {split: set(subjects) for split, subjects in S1_SUBJECTS.items()}
    sample_ids = {split: all_sample_ids(frames) for split, frames in frames_by_split.items()}
    for split, expected in S1_COUNTS.items():
        if len(sample_ids[split]) != expected:
            raise ValueError(f"{split} count {len(sample_ids[split])} != official {expected}")
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    subject_intersections = {f"{a}_{b}": sorted(subjects_by_split[a] & subjects_by_split[b]) for a, b in split_pairs}
    sample_intersections = {f"{a}_{b}": len(set(sample_ids[a]) & set(sample_ids[b])) for a, b in split_pairs}
    episode_split = {}
    for split, frames in frames_by_split.items():
        for frame in frames:
            previous = episode_split.setdefault(frame.episode_id, split)
            if previous != split:
                raise ValueError(f"episode crosses official split: {frame.episode_id}")
    audit_subject_counts = {
        subject: int(row["complete_rgb_depth_label_samples"])
        for subject, row in audit["subject_stats"].items()
    }
    expected_subject_counts = Counter()
    for frames in frames_by_split.values():
        for frame in frames:
            expected_subject_counts[frame.subject_id] += len(frame.serials)
    if dict(expected_subject_counts) != audit_subject_counts:
        raise ValueError("existing copy audit subject counts disagree with meta.yml enumeration")
    report = {
        "dataset": "DexYCB",
        "protocol": "official S1 unseen-subject",
        "toolkit": verify_toolkit(toolkit_root),
        "bop_s1": verify_bop(bop_manifest, expected_bop_keyframes(frames_by_split["test"])),
        "existing_copy_audit": {
            "path": str(audit_summary),
            "sha256": file_sha256(audit_summary),
            "subject_counts_match": True,
            "zero_byte_counts": audit.get("zero_byte_counts", {}),
        },
        "splits": {},
        "subject_intersections": subject_intersections,
        "sample_id_intersections": sample_intersections,
        "episode_cross_split_count": 0,
        "synchronized_views_cross_split_count": 0,
    }
    for split, frames in frames_by_split.items():
        ids = sample_ids[split]
        report["splits"][split] = {
            "subjects": list(S1_SUBJECTS[split]),
            "subject_count": len(S1_SUBJECTS[split]),
            "sequence_count": len({frame.episode_id for frame in frames}),
            "camera_serials": list(SERIALS),
            "camera_count": len(SERIALS),
            "synchronized_frame_count": len(frames),
            "sample_count": len(ids),
            "sample_id_sha256": ids_sha256(ids),
            "per_subject": dict(sorted(Counter(frame.subject_id for frame in frames for _ in frame.serials).items())),
            "per_camera": dict(sorted(Counter(serial for frame in frames for serial in frame.serials).items())),
            "per_hand_side": dict(sorted(Counter(
                ("right" if frame.is_right else "left")
                for frame in frames for _ in frame.serials
            ).items())),
        }
    if any(subject_intersections.values()) or any(sample_intersections.values()):
        raise ValueError("official S1 overlap audit failed")
    return report


def require_test_freeze(path: Path | None) -> dict:
    if path is None or not path.is_file():
        raise PermissionError("DexYCB test export requires --test-freeze-file")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "frozen",
        "dataset": "dexycb_s1",
        "checkpoint_selection": "internal_validation_only",
        "test_score_reads_before_freeze": 0,
    }
    mismatched = {key: (freeze.get(key), value) for key, value in required.items() if freeze.get(key) != value}
    checkpoints = freeze.get("checkpoint_sha256", {})
    if mismatched or set(checkpoints) != {"rgb_only", "rgb_rope"}:
        raise PermissionError(f"invalid DexYCB test freeze: fields={mismatched} checkpoints={sorted(checkpoints)}")
    return {"path": str(path), "sha256": file_sha256(path), "checkpoint_sha256": checkpoints}


def export_all(args) -> Path:
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("derived output must not be inside the raw DexYCB root")
    frames_by_split = enumerate_s1(raw_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report = audit_report(frames_by_split, args.audit_summary, args.toolkit_root, args.bop_manifest)
    (output_root / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    k_by_serial = intrinsics(raw_root)

    requested = set(args.splits.split(","))
    unknown = requested - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    test_freeze = require_test_freeze(args.test_freeze_file) if "test" in requested else None

    exports = {}
    if "train" in requested:
        selected, validity_selection = select_balanced_valid_views(
            raw_root, frames_by_split["train"], args.train_count, args.seed
        )
        train_episodes, val_episodes = internal_episode_split(selected, 0.1, 0)
        sequence_counts = Counter(frame.episode_id for frame, _ in selected)
        camera_counts = Counter(serial for _, serial in selected)
        sync_ids = [f"{frame.episode_id}/{frame.frame_index:06d}" for frame, _ in selected]
        if len(sync_ids) != len(set(sync_ids)):
            raise ValueError("train27k selected more than one camera for a synchronized frame")
        selection = {
            "name": "dexycb_s1_train27k_v1",
            "rule": "episode round-robin; stable frame hash; globally least-used camera with stable hash tie-break",
            "seed": args.seed,
            "uses_val_or_test_error": False,
            "validity_rule": "skip official joint_3d=-1 or pose_m=0 no-visible-hand sentinels before fixed-budget selection",
            "validity_selection": validity_selection,
            "max_views_per_subject_sequence_frame": 1,
            "sequence_count": len(sequence_counts),
            "sequence_count_min_max": [min(sequence_counts.values()), max(sequence_counts.values())],
            "camera_counts": dict(sorted(camera_counts.items())),
            "camera_count_min_max": [min(camera_counts.values()), max(camera_counts.values())],
            "internal_validation": {
                "rule": "complete subject/sequence episodes; numpy RNG seed 0; ceil(10 percent)",
                "train_episodes": train_episodes,
                "validation_episodes": val_episodes,
                "episode_intersection": [],
            },
        }
        train_root = output_root / "train27k"
        exports["train27k"] = export_subset(raw_root, train_root, selected, "training", k_by_serial, selection)
        smoke_selected = sorted(selected, key=lambda row: stable_digest(args.seed, "smoke", row[0].sample_id(row[1])))[:args.smoke_count]
        exports["smoke"] = export_subset(
            raw_root,
            output_root / "smoke",
            smoke_selected,
            "training",
            k_by_serial,
            {"name": "dexycb_s1_smoke_v1", "source": "train27k only", "count": args.smoke_count},
        )
    for split in ("val", "test"):
        if split not in requested:
            continue
        selected = [(frame, serial) for frame in frames_by_split[split] for serial in frame.serials]
        selection = {"name": f"dexycb_s1_{split}", "rule": "all official S1 views in canonical order"}
        if split == "test":
            selection["test_access_freeze"] = test_freeze
        exports[split] = export_subset(raw_root, output_root / split, selected, "evaluation", k_by_serial, selection)

    top_protocol = {
        "dataset": "DexYCB",
        "official_setup": "s1",
        "raw_root": str(raw_root),
        "processed_root": str(output_root),
        "split_report": "split_report.json",
        "test_policy": "test labels exported only after frozen recipe/checkpoint guard; score once",
        "exports": {name: value["sha256"] for name, value in exports.items()},
    }
    (output_root / "protocol.json").write_text(json.dumps(top_protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "exports": {key: value["num_samples"] for key, value in exports.items()}}, indent=2))
    return output_root


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--bop-manifest", type=Path, required=True)
    parser.add_argument("--splits", default="train,val", help="Comma-separated train,val,test")
    parser.add_argument("--test-freeze-file", type=Path)
    parser.add_argument("--train-count", type=int, default=TRAIN27K_COUNT)
    parser.add_argument("--smoke-count", type=int, default=SMOKE_COUNT)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args(argv)
    if args.train_count < 2 or args.smoke_count < 2 or args.smoke_count > args.train_count:
        parser.error("require 2 <= smoke-count <= train-count")
    return args


if __name__ == "__main__":
    export_all(parse_args())
