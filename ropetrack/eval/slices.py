#!/usr/bin/env python3
"""Occlusion-sliced scoring for base vs refined predictions.

The headline benchmark averages over all 21 joints and all samples, which
dilutes a correction that (by construction) can only act on occluded finger
curl. This script slices the same PA-aligned per-joint errors by:

- fingertips vs all joints;
- occluded vs clean fingers (derived from hard_manifest.jsonl);
- base rope-residual buckets (needs the refiner eval cache), including the
  residual-vs-improvement correlation that shows corrections concentrate
  where the rope disagreed with the prediction.

Alignment note: Procrustes alignment always uses all 21 joints (identical to
scripts/evaluation/score_predictions.py xyz_procrustes_al_*); slicing only selects which
joints' distances are averaged. The all_joints slice therefore reproduces
xyz_procrustes_al_mean3d.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ropetrack.datasets.hand_pose import clamp_bbox, centered_rect  # noqa: E402
from ropetrack.io import load_pred_json, read_json, read_jsonl  # noqa: E402
from ropetrack.refine.analysis import (  # noqa: E402
    bucket_indices,
    json_sanitize,
    pearson,
    quantile_bucket_edges,
    spearman,
)
from ropetrack.rope import FINGER_CHAINS, FINGER_ORDER, canonical_rope_dataset  # noqa: E402

from ropetrack.eval.scoring import align_w_scale  # noqa: E402

# Effects where every finger is directly targeted, so occlusion = all fingers.
ALL_FINGER_EFFECTS = {"tip_circle", "tip_square", "tip_blur", "tip_mixed", "finger_end", "blur"}
DEFAULT_IMAGE_SIZE = {"freihand": (224, 224), "ho3d": (640, 480)}


def occluded_fingers_for_row(row: dict, image_size: tuple[int, int]) -> list[bool] | None:
    """Per-finger occlusion flags in FINGER_ORDER, or None when undecidable."""
    effect = row.get("effect")
    if effect in ALL_FINGER_EFFECTS:
        return [True] * 5
    if effect == "mask":
        points = row.get("points_xy") or []
        if len(points) != 5:
            return None
        rect = centered_rect(*clamp_bbox(row["bbox_xyxy"], image_size[0], image_size[1]), row["severity"])
        rx1, ry1, rx2, ry2 = rect
        # PIL ImageDraw.rectangle paints both endpoints inclusive, so the
        # blacked-out pixel region is [rx1, rx2 + 1) x [ry1, ry2 + 1).
        return [bool(rx1 <= px < rx2 + 1 and ry1 <= py < ry2 + 1) for px, py in points]
    return None


def load_pred_xyz(path: Path) -> np.ndarray:
    xyz, _ = load_pred_json(path)
    return np.asarray(xyz, dtype=np.float64)


def per_joint_pa_distances(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> np.ndarray:
    """[N, 21] per-joint distances after all-joint Procrustes alignment, meters."""
    if gt_xyz.shape != pred_xyz.shape or gt_xyz.ndim != 3 or gt_xyz.shape[1:] != (21, 3):
        raise ValueError(f"expected matching [N, 21, 3] arrays, got gt={gt_xyz.shape} pred={pred_xyz.shape}")
    distances = np.zeros(gt_xyz.shape[:2], dtype=np.float64)
    for idx in range(gt_xyz.shape[0]):
        aligned = align_w_scale(gt_xyz[idx], pred_xyz[idx])
        distances[idx] = np.linalg.norm(gt_xyz[idx] - aligned, axis=1)
    return distances


def finger_joint_map(dataset: str) -> tuple[list[list[int]], list[int]]:
    """Per-finger non-wrist joint ids and tip ids, in FINGER_ORDER."""
    chains = FINGER_CHAINS[canonical_rope_dataset(dataset)]
    finger_joints = [list(chain[1:]) for chain in chains]
    tips = [chain[-1] for chain in chains]
    return finger_joints, tips


def slice_mean_cm(distances: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Mean distance in cm over selected (sample, joint) entries."""
    selected = distances[mask]
    if selected.size == 0:
        return float("nan"), 0
    return float(selected.mean() * 100.0), int(selected.size)


def build_joint_masks(
    num_samples: int,
    dataset: str,
    occluded: list[list[bool] | None],
) -> dict[str, np.ndarray]:
    finger_joints, tips = finger_joint_map(dataset)
    masks = {
        "all_joints": np.ones((num_samples, 21), dtype=bool),
        "fingertips": np.zeros((num_samples, 21), dtype=bool),
    }
    masks["fingertips"][:, tips] = True

    known = np.asarray([flags is not None for flags in occluded], dtype=bool)
    if known.any():
        for name in ("occluded_finger_joints", "occluded_fingertips", "clean_finger_joints", "clean_fingertips"):
            masks[name] = np.zeros((num_samples, 21), dtype=bool)
        for idx, flags in enumerate(occluded):
            if flags is None:
                continue
            for finger_idx, is_occluded in enumerate(flags):
                joint_key = "occluded_finger_joints" if is_occluded else "clean_finger_joints"
                tip_key = "occluded_fingertips" if is_occluded else "clean_fingertips"
                masks[joint_key][idx, finger_joints[finger_idx]] = True
                masks[tip_key][idx, tips[finger_idx]] = True
    return masks


def per_finger_tip_table(
    base_distances: np.ndarray,
    refined_distances: np.ndarray,
    dataset: str,
    occluded: list[list[bool] | None],
) -> dict:
    _, tips = finger_joint_map(dataset)
    table = {}
    for finger_idx, finger in enumerate(FINGER_ORDER):
        entry = {}
        for label, wanted in (("occluded", True), ("clean", False)):
            rows = [idx for idx, flags in enumerate(occluded) if flags is not None and flags[finger_idx] == wanted]
            if rows:
                base_vals = base_distances[rows, tips[finger_idx]]
                refined_vals = refined_distances[rows, tips[finger_idx]]
                entry[label] = {
                    "n": len(rows),
                    "tip_base_cm": float(base_vals.mean() * 100.0),
                    "tip_refined_cm": float(refined_vals.mean() * 100.0),
                    "tip_delta_cm": float((refined_vals - base_vals).mean() * 100.0),
                }
            else:
                entry[label] = {"n": 0}
        table[finger] = entry
    return table


def sample_rope_residual(cache: dict[str, np.ndarray]) -> np.ndarray:
    base = np.asarray(cache["base_rope_norm"], dtype=np.float64)
    target = np.asarray(cache["input_rope_norm"], dtype=np.float64)
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    residual = np.where(valid, np.abs(base - target), 0.0)
    counts = valid.sum(axis=1)
    # manual masked mean: nanmean warns on all-invalid rows
    return np.where(counts > 0, residual.sum(axis=1) / np.maximum(counts, 1), np.nan)


def residual_bucket_table(
    residual: np.ndarray,
    base_distances: np.ndarray,
    refined_distances: np.ndarray,
    masks: dict[str, np.ndarray],
    num_buckets: int,
) -> list[dict]:
    edges = quantile_bucket_edges(residual, num_buckets)
    bucket_of = bucket_indices(residual, edges)
    rows = []
    for bucket in range(num_buckets):
        selected = bucket_of == bucket
        if not selected.any():
            rows.append({"bucket": bucket, "n": 0})
            continue
        entry = {
            "bucket": bucket,
            "n": int(selected.sum()),
            "mean_residual": float(np.nanmean(residual[selected])),
        }
        for name in ("all_joints", "occluded_fingertips"):
            if name not in masks:
                continue
            mask = masks[name] & selected[:, None]
            base_cm, count = slice_mean_cm(base_distances, mask)
            refined_cm, _ = slice_mean_cm(refined_distances, mask)
            entry[name] = {
                "base_cm": base_cm,
                "refined_cm": refined_cm,
                "delta_cm": refined_cm - base_cm if count else float("nan"),
                "num_joint_obs": count,
            }
        rows.append(entry)
    return rows


def residual_correlations(
    residual: np.ndarray,
    base_distances: np.ndarray,
    refined_distances: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict:
    out = {}
    for name in ("all_joints", "occluded_fingertips"):
        if name not in masks:
            continue
        mask = masks[name]
        with np.errstate(invalid="ignore"):
            base_mean = np.where(mask.any(axis=1), (base_distances * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), np.nan)
            refined_mean = np.where(mask.any(axis=1), (refined_distances * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1), np.nan)
        improvement_cm = (base_mean - refined_mean) * 100.0
        out[name] = {
            "pearson_residual_vs_improvement": pearson(residual, improvement_cm),
            "spearman_residual_vs_improvement": spearman(residual, improvement_cm),
        }
    return out


def write_tsv(path: Path, report: dict) -> None:
    lines = ["section\tname\tbase_cm\trefined_cm\tdelta_cm\tn"]
    for name, entry in report["slices"].items():
        lines.append(
            f"slice\t{name}\t{entry['base_cm']:.4f}\t{entry['refined_cm']:.4f}\t{entry['delta_cm']:.4f}\t{entry['num_joint_obs']}"
        )
    for finger, entry in report.get("per_finger", {}).items():
        for label in ("occluded", "clean"):
            sub = entry.get(label, {})
            if sub.get("n"):
                lines.append(
                    f"finger_{label}\t{finger}\t{sub['tip_base_cm']:.4f}\t{sub['tip_refined_cm']:.4f}\t{sub['tip_delta_cm']:.4f}\t{sub['n']}"
                )
    for row in report.get("residual_buckets", []):
        if row.get("n") and "all_joints" in row:
            entry = row["all_joints"]
            lines.append(
                f"residual_bucket\tq{row['bucket']}(r={row['mean_residual']:.4f})\t{entry['base_cm']:.4f}\t{entry['refined_cm']:.4f}\t{entry['delta_cm']:.4f}\t{row['n']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    dataset: str,
    gt_xyz: np.ndarray,
    base_xyz: np.ndarray,
    refined_xyz: np.ndarray,
    manifest_rows: list[dict] | None,
    cache: dict[str, np.ndarray] | None,
    image_size: tuple[int, int],
    num_buckets: int,
) -> dict:
    num_samples = gt_xyz.shape[0]
    base_distances = per_joint_pa_distances(gt_xyz, base_xyz)
    refined_distances = per_joint_pa_distances(gt_xyz, refined_xyz)

    occluded: list[list[bool] | None] = [None] * num_samples
    occlusion_info = {}
    if manifest_rows is not None:
        if len(manifest_rows) != num_samples:
            raise ValueError(f"manifest rows ({len(manifest_rows)}) != samples ({num_samples})")
        occluded = [occluded_fingers_for_row(row, image_size) for row in manifest_rows]
        known = [flags for flags in occluded if flags is not None]
        occlusion_info = {
            "num_with_occlusion_info": len(known),
            "num_undecidable": num_samples - len(known),
            "mean_occluded_fingers": float(np.mean([sum(flags) for flags in known])) if known else float("nan"),
        }

    masks = build_joint_masks(num_samples, dataset, occluded)
    slices = {}
    for name, mask in masks.items():
        base_cm, count = slice_mean_cm(base_distances, mask)
        refined_cm, _ = slice_mean_cm(refined_distances, mask)
        slices[name] = {
            "base_cm": base_cm,
            "refined_cm": refined_cm,
            "delta_cm": refined_cm - base_cm if count else float("nan"),
            "num_joint_obs": count,
        }

    report = {
        "dataset": dataset,
        "num_samples": num_samples,
        "slices": slices,
        "occlusion": occlusion_info,
    }
    if manifest_rows is not None:
        report["per_finger"] = per_finger_tip_table(base_distances, refined_distances, dataset, occluded)
    if cache is not None:
        if len(cache["sample_id"]) != num_samples:
            raise ValueError(f"cache samples ({len(cache['sample_id'])}) != gt samples ({num_samples})")
        if manifest_rows is not None:
            cache_ids = [str(sid) for sid in cache["sample_id"]]
            manifest_ids = [str(row["sample_id"]) for row in manifest_rows]
            if cache_ids != manifest_ids:
                raise ValueError("cache/manifest sample_id order mismatch")
        residual = sample_rope_residual(cache)
        report["residual_buckets"] = residual_bucket_table(residual, base_distances, refined_distances, masks, num_buckets)
        report["correlations"] = residual_correlations(residual, base_distances, refined_distances, masks)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Occlusion-sliced base-vs-refined scoring.")
    parser.add_argument("pred_dir", type=Path, help="Directory containing base and refined prediction json files.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", choices=["freihand", "ho3d"], required=True)
    parser.add_argument("--gt-dir", type=Path, default=None, help="Directory with {set_name}_xyz.json. Defaults to pred_dir.")
    parser.add_argument("--set-name", default="evaluation")
    parser.add_argument("--base-file", default="base_pred.json")
    parser.add_argument("--pred-file", default="pred.json")
    parser.add_argument("--hard-manifest", type=Path, default=None, help="hard_manifest.jsonl from make_hard_images.py; enables occlusion slices.")
    parser.add_argument("--cache", type=Path, default=None, help="refiner_eval_cache.npz; enables residual buckets and correlations.")
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--residual-buckets", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    gt_dir = args.gt_dir if args.gt_dir is not None else args.pred_dir
    gt_xyz = np.asarray(read_json(gt_dir / f"{args.set_name}_xyz.json"), dtype=np.float64)
    base_xyz = load_pred_xyz(args.pred_dir / args.base_file)
    refined_xyz = load_pred_xyz(args.pred_dir / args.pred_file)

    manifest_rows = list(read_jsonl(args.hard_manifest)) if args.hard_manifest is not None else None
    cache = None
    if args.cache is not None:
        with np.load(args.cache) as loaded:
            cache = {key: loaded[key] for key in loaded.files}

    default_w, default_h = DEFAULT_IMAGE_SIZE[canonical_rope_dataset(args.dataset)]
    image_size = (
        args.image_width if args.image_width is not None else default_w,
        args.image_height if args.image_height is not None else default_h,
    )

    report = build_report(
        args.dataset,
        gt_xyz,
        base_xyz,
        refined_xyz,
        manifest_rows,
        cache,
        image_size,
        args.residual_buckets,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sliced_scores.json").write_text(json.dumps(json_sanitize(report), indent=2), encoding="utf-8")
    write_tsv(args.output_dir / "sliced_scores.tsv", report)
    print(f"Sliced scores written to: {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
