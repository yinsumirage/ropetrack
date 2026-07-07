from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Ho3dSample:
    sample_id: str
    image_path: Path
    meta_path: Path


@dataclass(frozen=True)
class FreiHandSample:
    sample_id: str
    image_path: Path
    bbox_xyxy: np.ndarray


@dataclass(frozen=True)
class BBoxItem:
    sample_index: int
    bbox_index: int
    sample: object
    bbox_xyxy: np.ndarray
    is_right: bool
    score: float
    source: str


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(rgb_dir: Path, frame: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = rgb_dir / f"{frame}{suffix}"
        if path.exists():
            return path
    return rgb_dir / f"{frame}.png"


def project_points(points, K) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    intrinsics = np.asarray(K, dtype=np.float32)
    uvw = (intrinsics @ pts.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def bbox_from_projected_points(points, K, image_size: int = 224) -> np.ndarray:
    uv = project_points(points, K)
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        raise ValueError("no finite projected points for bbox")
    xy_min = uv[valid].min(axis=0)
    xy_max = uv[valid].max(axis=0)
    xyxy = np.concatenate([xy_min, xy_max]).astype(np.float32)
    return np.clip(xyxy, 0.0, float(image_size))


def iter_hand_pose_samples(adapter: str, root: Path, limit: int | None, split: str = "evaluation"):
    if adapter == "freihand":
        return iter_freihand_eval_samples(root, limit, split)
    if adapter == "ho3d":
        if split == "evaluation":
            return iter_ho3d_samples(root, limit)
        if split in {"training", "train"}:
            return iter_ho3d_train_samples(root, limit)
        raise ValueError(f"unsupported HO3D split: {split}")
    raise ValueError(f"unsupported eval adapter: {adapter}")


iter_eval_samples = iter_hand_pose_samples


def iter_freihand_eval_samples(root: Path, limit: int | None, split: str = "evaluation") -> Iterable[FreiHandSample]:
    Ks = read_json(root / f"{split}_K.json")
    verts = read_json(root / f"{split}_verts.json")
    if len(Ks) != len(verts):
        raise ValueError(f"FreiHAND {split} length mismatch: K={len(Ks)} verts={len(verts)}")
    if limit is not None and limit <= 0:
        limit = None

    for idx, (K, sample_verts) in enumerate(zip(Ks, verts)):
        if limit is not None and idx >= limit:
            break
        frame = f"{idx:08d}"
        yield FreiHandSample(
            sample_id=frame,
            image_path=resolve_image_path(root / split / "rgb", frame),
            bbox_xyxy=bbox_from_projected_points(sample_verts, K),
        )


def iter_ho3d_samples(root: Path, limit: int | None) -> Iterable[Ho3dSample]:
    eval_dir = root / "evaluation"
    list_path = root / "evaluation.txt"
    if list_path.exists():
        ids = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    elif (root / "evaluation_xyz.json").exists():
        ids = infer_ho3d_sample_order_from_gt_roots(root)
    else:
        ids = [
            f"{seq.name}/{img.stem}"
            for seq in sorted(eval_dir.iterdir())
            if seq.is_dir()
            for img in sorted((seq / "rgb").glob("*.png"))
        ]

    if limit is not None and limit <= 0:
        limit = None

    for sample_id in ids[:limit]:
        seq, frame = sample_id.split("/")
        yield Ho3dSample(
            sample_id=sample_id,
            image_path=resolve_image_path(eval_dir / seq / "rgb", frame),
            meta_path=eval_dir / seq / "meta" / f"{frame}.pkl",
        )


def read_ho3d_train_ids(root: Path, stride: int = 1, limit: int | None = None, split_file: str = "train.txt") -> list[str]:
    """Strided ids from train.txt.

    HO3D training sequences are 30 fps video, so adjacent frames are nearly
    duplicates; stride subsampling keeps the unique-pose content while
    keeping teacher-generation cost and train/val leakage in check.
    """
    list_path = root / split_file
    if not list_path.exists():
        raise FileNotFoundError(f"HO3D train split list missing: {list_path}")
    ids = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    ids = ids[::stride]
    if limit is not None and limit > 0:
        ids = ids[:limit]
    return ids


def iter_ho3d_train_samples(root: Path, limit: int | None, stride: int = 1, split_dir: str = "train") -> Iterable[Ho3dSample]:
    if limit is not None and limit <= 0:
        limit = None
    for sample_id in read_ho3d_train_ids(root, stride=stride, limit=limit):
        seq, frame = sample_id.split("/")
        yield Ho3dSample(
            sample_id=sample_id,
            image_path=resolve_image_path(root / split_dir / seq / "rgb", frame),
            meta_path=root / split_dir / seq / "meta" / f"{frame}.pkl",
        )


HO3D_TO_OPENCV_CAMERA = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
HO3D_IMAGE_SIZE = (640, 480)


def ho3d_projected_hand_bbox(
    meta: dict,
    image_size: tuple[int, int] = HO3D_IMAGE_SIZE,
    margin: float = 0.15,
) -> np.ndarray:
    """Hand bbox from projected 3D joints, for TRAIN metas without handBoundingBox.

    Joint projections hug the skeleton, so pad by `margin` of the larger bbox
    side to cover the hand surface (evaluation handBoundingBox boxes carry a
    similar margin).
    """
    joints = np.asarray(meta["handJoints3D"], dtype=np.float32) * HO3D_TO_OPENCV_CAMERA
    K = np.asarray(meta["camMat"], dtype=np.float32)
    uv = project_points(joints, K)
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        raise ValueError("no finite projected joints for HO3D train bbox")
    xy_min = uv[valid].min(axis=0)
    xy_max = uv[valid].max(axis=0)
    pad = margin * float(max(xy_max[0] - xy_min[0], xy_max[1] - xy_min[1], 1.0))
    width, height = image_size
    bbox = np.asarray([
        max(0.0, xy_min[0] - pad),
        max(0.0, xy_min[1] - pad),
        min(float(width), xy_max[0] + pad),
        min(float(height), xy_max[1] + pad),
    ], dtype=np.float32)
    return bbox.reshape(1, 4)


def infer_ho3d_sample_order_from_gt_roots(root: Path) -> list[str]:
    eval_dir = root / "evaluation"
    meta_items = []
    for meta_path in eval_dir.glob("*/meta/*.pkl"):
        with meta_path.open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        seq = meta_path.parents[1].name
        sample_id = f"{seq}/{meta_path.stem}"
        meta_items.append((sample_id, np.asarray(meta["handJoints3D"], dtype=np.float64).reshape(-1)[:3]))

    meta_ids = [item[0] for item in meta_items]
    meta_roots = np.stack([item[1] for item in meta_items], axis=0)
    gt_roots = np.asarray(read_json(root / "evaluation_xyz.json"), dtype=np.float64)[:, 0, :]
    ids = []
    used = set()
    max_dist = 0.0
    for root_xyz in gt_roots:
        dists = np.linalg.norm(meta_roots - root_xyz[None, :], axis=1)
        match = next(int(i) for i in np.argsort(dists) if int(i) not in used)
        used.add(match)
        max_dist = max(max_dist, float(dists[match]))
        ids.append(meta_ids[match])
    if max_dist > 1e-4:
        raise ValueError(f"HO3D order matching too loose: max root distance {max_dist}")
    return ids


def hand_bbox_from_meta(meta: dict) -> np.ndarray:
    return np.asarray(meta["handBoundingBox"], dtype=np.float32).reshape(-1, 4)


def bbox_candidates_from_sample(
    sample_index: int,
    sample,
    boxes: np.ndarray,
    is_right: np.ndarray,
    scores: np.ndarray,
    source: str,
) -> list[BBoxItem]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    is_right = np.asarray(is_right).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (len(boxes) == len(is_right) == len(scores)):
        raise ValueError("boxes, is_right, and scores must have matching lengths")
    return [
        BBoxItem(
            sample_index=sample_index,
            bbox_index=bbox_index,
            sample=sample,
            bbox_xyxy=boxes[bbox_index],
            is_right=bool(is_right[bbox_index]),
            score=float(scores[bbox_index]),
            source=source,
        )
        for bbox_index in range(len(boxes))
    ]


def load_gt_bbox_candidates(adapter: str, samples: list) -> list[BBoxItem]:
    if adapter == "freihand":
        return [
            BBoxItem(idx, 0, sample, sample.bbox_xyxy, True, 1.0, "gt_bbox")
            for idx, sample in enumerate(samples)
        ]
    if adapter == "ho3d":
        candidates = []
        for sample_index, sample in enumerate(samples):
            with sample.meta_path.open("rb") as f:
                meta = pickle.load(f, encoding="latin1")
            if meta.get("handBoundingBox") is not None:
                boxes = hand_bbox_from_meta(meta)
            else:
                # training metas carry no handBoundingBox (audit 0040):
                # fall back to a bbox projected from the GT 3D joints
                boxes = ho3d_projected_hand_bbox(meta)
            candidates.extend(bbox_candidates_from_sample(
                sample_index,
                sample,
                boxes,
                np.ones(len(boxes), dtype=np.float32),
                np.ones(len(boxes), dtype=np.float32),
                "gt_bbox",
            ))
        return candidates
    raise ValueError(f"unsupported eval adapter: {adapter}")


def write_eval_gt_subset(adapter: str, root: Path, eval_input: Path, count: int, split: str = "evaluation") -> None:
    _ = adapter
    eval_input.mkdir(parents=True, exist_ok=True)
    for gt_name in (f"{split}_xyz.json", f"{split}_verts.json"):
        gt_path = root / gt_name
        if gt_path.exists():
            values = read_json(gt_path)
            (eval_input / gt_name).write_text(json.dumps(values[:count]))


def validate_eval_protocol(adapter: str, root: Path, samples: list, count: int | None, tolerance_m: float | None) -> None:
    if count is None or count <= 0 or tolerance_m is None:
        return
    check_count = min(count, len(samples))
    xyz = read_json(root / "evaluation_xyz.json")
    verts = read_json(root / "evaluation_verts.json")
    if len(xyz) < check_count or len(verts) < check_count:
        raise ValueError(f"{adapter} GT shorter than protocol check count {check_count}")
    if adapter != "ho3d":
        return

    for idx, sample in enumerate(samples[:check_count]):
        with sample.meta_path.open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        meta_root = np.asarray(meta["handJoints3D"], dtype=np.float64)[0]
        gt_root = np.asarray(xyz[idx], dtype=np.float64)[0]
        dist = float(np.linalg.norm(meta_root - gt_root))
        if dist > tolerance_m:
            raise ValueError(
                f"HO3D protocol check failed at {idx} {sample.sample_id}: "
                f"root distance {dist:.6f}m > {tolerance_m:.6f}m"
            )
