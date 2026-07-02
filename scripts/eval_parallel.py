#!/usr/bin/env python3
"""Parallel mesh/joint eval compatible with the current benchmark scores."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


F_THRESHOLDS = (0.005, 0.015)
_GT_XYZ = None
_GT_VERTS = None
_PRED_XYZ = None
_PRED_VERTS = None


def calculate_fscore(gt, pred, threshold):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    d_gt = np.min(np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1), axis=1)
    d_pred = np.min(np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1), axis=1)
    recall = float(np.mean(d_pred < threshold)) if len(d_pred) else 0.0
    precision = float(np.mean(d_gt < threshold)) if len(d_gt) else 0.0
    if recall + precision == 0.0:
        return 0.0
    return 2.0 * recall * precision / (recall + precision)


def align_sc_tr(gt, pred):
    pred_aligned = pred.copy()
    pred_scale = np.linalg.norm(pred_aligned[4] - pred_aligned[0])
    if pred_scale > 0:
        pred_aligned = pred_aligned / pred_scale
    gt_scale = np.linalg.norm(gt[4] - gt[0])
    pred_aligned = pred_aligned * gt_scale
    return pred_aligned - pred_aligned[0:1, :] + gt[0:1, :]


def align_w_scale(gt, pred, return_trafo=False):
    gt_mean = gt.mean(0)
    pred_mean = pred.mean(0)
    gt_normed = gt - gt_mean
    pred_normed = pred - pred_mean
    gt_scale = np.linalg.norm(gt_normed) + 1e-8
    pred_scale = np.linalg.norm(pred_normed) + 1e-8
    gt_normed = gt_normed / gt_scale
    pred_normed = pred_normed / pred_scale

    u, singular_values, vt = np.linalg.svd(gt_normed.T @ pred_normed, full_matrices=False)
    rotation = u @ vt
    procrustes_scale = singular_values.sum()
    aligned = pred_normed @ rotation.T * procrustes_scale * gt_scale + gt_mean
    if return_trafo:
        return rotation, procrustes_scale, gt_scale, gt_mean - pred_mean
    return aligned


def align_by_trafo(points, trafo):
    pred_mean = points.mean(0)
    centered = points - pred_mean
    rotation, procrustes_scale, gt_scale, translation = trafo
    return centered @ rotation.T * procrustes_scale * gt_scale + translation + pred_mean


def point_distances(gt, pred):
    return np.linalg.norm(gt - pred, axis=1)


def evaluate_sample(sample):
    xyz, verts, xyz_pred, verts_pred = [np.asarray(x, dtype=np.float64) for x in sample]
    xyz_pred_sc_tr = align_sc_tr(xyz, xyz_pred)
    xyz_pred_pa = align_w_scale(xyz, xyz_pred)

    if verts_pred.shape[0] == verts.shape[0]:
        verts_pred_pa = align_w_scale(verts, verts_pred)
    else:
        verts_pred_pa = align_by_trafo(verts_pred, align_w_scale(xyz, xyz_pred, return_trafo=True))

    return {
        "xyz": point_distances(xyz, xyz_pred),
        "xyz_pa": point_distances(xyz, xyz_pred_pa),
        "xyz_st": point_distances(xyz, xyz_pred_sc_tr),
        "mesh": point_distances(verts, verts_pred) if verts_pred.shape[0] == verts.shape[0] else None,
        "mesh_pa": point_distances(verts, verts_pred_pa) if verts_pred.shape[0] == verts.shape[0] else None,
        "f_scores": [calculate_fscore(verts, verts_pred, t) for t in F_THRESHOLDS],
        "f_scores_aligned": [calculate_fscore(verts, verts_pred_pa, t) for t in F_THRESHOLDS],
    }


def evaluate_index(idx):
    return evaluate_sample((_GT_XYZ[idx], _GT_VERTS[idx], _PRED_XYZ[idx], _PRED_VERTS[idx]))


def measure_distances(distances, val_min=0.0, val_max=0.05, steps=100):
    distances = np.asarray(distances, dtype=np.float64)
    thresholds = np.linspace(val_min, val_max, steps)
    pck_curve = np.array([(distances <= t).mean(axis=0).mean() for t in thresholds])
    auc = np.trapz(pck_curve, thresholds) / np.trapz(np.ones_like(thresholds), thresholds)
    return float(distances.mean(axis=0).mean()), float(auc), pck_curve, thresholds


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_inputs(input_dir, pred_file_name, set_name):
    input_dir = Path(input_dir)
    xyz = read_json(input_dir / f"{set_name}_xyz.json")
    verts = read_json(input_dir / f"{set_name}_verts.json")
    pred = read_json(input_dir / pred_file_name)
    if len(pred) != 2:
        raise ValueError("Expected pred.json to contain [xyz_predictions, vertex_predictions].")
    if not (len(xyz) == len(verts) == len(pred[0]) == len(pred[1])):
        raise ValueError(
            "Length mismatch: "
            f"xyz={len(xyz)} verts={len(verts)} pred_xyz={len(pred[0])} pred_verts={len(pred[1])}"
        )
    return xyz, verts, pred[0], pred[1]


def evaluate_dataset(input_dir, pred_file_name="pred.json", set_name="evaluation", num_workers=1, chunksize=16):
    global _GT_XYZ, _GT_VERTS, _PRED_XYZ, _PRED_VERTS
    _GT_XYZ, _GT_VERTS, _PRED_XYZ, _PRED_VERTS = load_inputs(input_dir, pred_file_name, set_name)
    indices = range(len(_GT_XYZ))
    if num_workers <= 1:
        return [evaluate_index(i) for i in indices]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(evaluate_index, indices, chunksize=chunksize))


def stack_results(results, key):
    values = [result[key] for result in results if result[key] is not None]
    if not values:
        return None
    return np.stack(values, axis=0)


def summarize_results(results):
    xyz_mean, xyz_auc, _, _ = measure_distances(stack_results(results, "xyz"))
    xyz_pa_mean, xyz_pa_auc, _, _ = measure_distances(stack_results(results, "xyz_pa"))
    xyz_st_mean, xyz_st_auc, _, _ = measure_distances(stack_results(results, "xyz_st"))

    mesh = stack_results(results, "mesh")
    mesh_pa = stack_results(results, "mesh_pa")
    if mesh is None:
        mesh_mean = mesh_auc = mesh_pa_mean = mesh_pa_auc = -1.0
    else:
        mesh_mean, mesh_auc, _, _ = measure_distances(mesh)
        mesh_pa_mean, mesh_pa_auc, _, _ = measure_distances(mesh_pa)

    f_scores = np.array([result["f_scores"] for result in results], dtype=np.float64)
    f_scores_aligned = np.array([result["f_scores_aligned"] for result in results], dtype=np.float64)

    scores = {
        "xyz_mean3d": xyz_mean * 100.0,
        "xyz_auc3d": xyz_auc,
        "xyz_procrustes_al_mean3d": xyz_pa_mean * 100.0,
        "xyz_procrustes_al_auc3d": xyz_pa_auc,
        "xyz_scale_trans_al_mean3d": xyz_st_mean * 100.0,
        "xyz_scale_trans_al_auc3d": xyz_st_auc,
        "mesh_mean3d": mesh_mean * 100.0,
        "mesh_auc3d": mesh_auc,
        "mesh_al_mean3d": mesh_pa_mean * 100.0,
        "mesh_al_auc3d": mesh_pa_auc,
        "f_score_5": float(f_scores[:, 0].mean()),
        "f_al_score_5": float(f_scores_aligned[:, 0].mean()),
        "f_score_15": float(f_scores[:, 1].mean()),
        "f_al_score_15": float(f_scores_aligned[:, 1].mean()),
    }
    return scores


def write_scores(output_dir, scores):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [
        "xyz_mean3d",
        "xyz_auc3d",
        "xyz_procrustes_al_mean3d",
        "xyz_procrustes_al_auc3d",
        "xyz_scale_trans_al_mean3d",
        "xyz_scale_trans_al_auc3d",
        "mesh_mean3d",
        "mesh_auc3d",
        "mesh_al_mean3d",
        "mesh_al_auc3d",
        "f_score_5",
        "f_al_score_5",
        "f_score_15",
        "f_al_score_15",
    ]
    with (output_dir / "scores.txt").open("w", encoding="utf-8") as f:
        for key in order:
            f.write(f"{key}: {scores[key]:f}\n")
    with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel benchmark evaluation.")
    parser.add_argument("input_dir", help="Directory containing pred.json and evaluation_* ground truth JSON files.")
    parser.add_argument("output_dir", help="Directory where scores.txt should be written.")
    parser.add_argument("--pred_file_name", default="pred.json", help="Prediction JSON filename.")
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--chunksize", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    results = evaluate_dataset(
        args.input_dir,
        pred_file_name=args.pred_file_name,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )
    scores = summarize_results(results)
    write_scores(args.output_dir, scores)
    print(f"Evaluated {len(results)} samples with {args.num_workers} worker(s).")
    print(f"Scores written to: {Path(args.output_dir) / 'scores.txt'}")


if __name__ == "__main__":
    main()
