from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ropetrack.io import load_pred_json, read_json
from ropetrack.refine.cache import load_sample_order


@dataclass(frozen=True)
class MeshTriplet:
    index: int
    sample_id: str
    gt: np.ndarray
    clean: np.ndarray
    hard: np.ndarray
    clean_error: float
    hard_error: float

    @property
    def degradation(self) -> float:
        return self.hard_error - self.clean_error


def read_predictions(run_dir: Path) -> tuple[list, list]:
    return load_pred_json(run_dir / "eval_input" / "pred.json")


def read_sample_order(run_dir: Path, count: int) -> list[str]:
    meta_path = run_dir / "run_meta.json"
    fallback = [str(i) for i in range(count)]
    if not meta_path.exists():
        return fallback
    return load_sample_order(meta_path, fallback)


def align_mesh_to_gt(gt, pred) -> np.ndarray:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    gt_mean = gt.mean(axis=0)
    pred_mean = pred.mean(axis=0)
    gt_centered = gt - gt_mean
    pred_centered = pred - pred_mean
    gt_norm = np.linalg.norm(gt_centered)
    pred_norm = np.linalg.norm(pred_centered)
    if gt_norm <= 1e-12 or pred_norm <= 1e-12:
        return pred + (gt_mean - pred_mean)
    gt_normed = gt_centered / gt_norm
    pred_normed = pred_centered / pred_norm
    u, singular_values, vt = np.linalg.svd(gt_normed.T @ pred_normed, full_matrices=False)
    rotation = u @ vt
    return pred_normed @ rotation.T * singular_values.sum() * gt_norm + gt_mean


def mesh_error(gt, pred) -> float:
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.linalg.norm(gt - pred, axis=1).mean())


def load_triplets(clean_run: Path, hard_run: Path, gt_root: Path, indices: list[int] | None = None) -> list[MeshTriplet]:
    _, clean_verts = read_predictions(clean_run)
    _, hard_verts = read_predictions(hard_run)
    gt_verts = read_json(gt_root / "evaluation_verts.json")
    if not (len(clean_verts) == len(hard_verts) == len(gt_verts)):
        raise ValueError(
            f"Length mismatch: clean={len(clean_verts)} hard={len(hard_verts)} gt={len(gt_verts)}"
        )
    order = read_sample_order(hard_run, len(gt_verts))
    wanted = indices if indices is not None else range(len(gt_verts))
    triplets = []
    for idx in wanted:
        gt, clean, hard = gt_verts[idx], clean_verts[idx], hard_verts[idx]
        gt_arr = np.asarray(gt, dtype=np.float64)
        clean_aligned = align_mesh_to_gt(gt_arr, clean)
        hard_aligned = align_mesh_to_gt(gt_arr, hard)
        triplets.append(MeshTriplet(
            index=idx,
            sample_id=order[idx],
            gt=gt_arr,
            clean=clean_aligned,
            hard=hard_aligned,
            clean_error=mesh_error(gt_arr, clean_aligned),
            hard_error=mesh_error(gt_arr, hard_aligned),
        ))
    return triplets


def select_triplets(triplets: list[MeshTriplet], count: int, mode: str) -> list[MeshTriplet]:
    if mode == "first":
        return triplets[:count]
    if mode == "worst":
        return sorted(triplets, key=lambda item: item.hard_error, reverse=True)[:count]
    if mode == "degradation":
        return sorted(triplets, key=lambda item: item.degradation, reverse=True)[:count]
    if mode == "middle_degradation":
        ordered = sorted(triplets, key=lambda item: item.degradation)
        start = max(0, (len(ordered) - count) // 2)
        return ordered[start:start + count]
    if mode == "low_degradation":
        return sorted(triplets, key=lambda item: abs(item.degradation))[:count]
    raise ValueError(f"unsupported selection mode: {mode}")


def load_mano_faces(mano_path: Path) -> np.ndarray:
    with mano_path.open("rb") as f:
        mano = pickle.load(f, encoding="latin1")
    return np.asarray(mano["f"], dtype=np.int64)
