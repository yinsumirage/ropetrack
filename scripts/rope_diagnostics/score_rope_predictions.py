#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.io import load_pred_json, read_jsonl, write_jsonl
from ropetrack.refine.cache import load_sample_order as load_run_meta_order
from ropetrack.rope import FINGER_ORDER, canonical_rope_dataset, normalize_rope_distance, rope_distances_for_joints


def load_prediction_joints(pred_dir: Path, pred_file_name: str = "pred.json"):
    xyz, _ = load_pred_json(pred_dir / pred_file_name)
    return xyz


def load_sample_order(pred_dir: Path, run_meta: Path | None, rope_rows: list[dict]) -> list[str]:
    """Loud order resolution (rule 7): an explicit --run-meta must exist and
    carry sample_order; without one, the conventional sibling run_meta.json is
    used if present, else the rope-label row order (documented fallback)."""
    fallback = [row["sample_id"] for row in rope_rows]
    if run_meta is not None:
        return load_run_meta_order(run_meta, fallback)
    sibling = pred_dir.parent / "run_meta.json"
    if sibling.exists():
        return load_run_meta_order(sibling, fallback)
    return fallback


def abs_or_none(left, right):
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def mean_present(values) -> float | None:
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def summarize_errors(error_rows: list[dict]) -> dict:
    per_finger_norm = []
    per_finger_dist = []
    for idx in range(len(FINGER_ORDER)):
        per_finger_norm.append(mean_present(row["rope_norm_abs_error"][idx] for row in error_rows))
        per_finger_dist.append(mean_present(row["rope_dist_abs_error_m"][idx] for row in error_rows))
    all_norm = [v for row in error_rows for v in row["rope_norm_abs_error"] if v is not None]
    all_dist = [v for row in error_rows for v in row["rope_dist_abs_error_m"] if v is not None]
    return {
        "num_samples": len(error_rows),
        "num_valid_fingers": len(all_norm),
        "finger_order": FINGER_ORDER,
        "rope_norm_mae": mean_present(all_norm),
        "rope_dist_mae_m": mean_present(all_dist),
        "per_finger_rope_norm_mae": per_finger_norm,
        "per_finger_rope_dist_mae_m": per_finger_dist,
    }


def score_rope_predictions(
    pred_dir: Path,
    rope_labels: Path,
    output_dir: Path,
    dataset: str | None = None,
    run_meta: Path | None = None,
    pred_file_name: str = "pred.json",
) -> dict:
    rope_rows = list(read_jsonl(rope_labels))
    if not rope_rows:
        raise ValueError(f"no rope rows in {rope_labels}")
    dataset_name = canonical_rope_dataset(dataset or rope_rows[0]["dataset"])
    pred_joints = load_prediction_joints(pred_dir, pred_file_name)
    sample_order = load_sample_order(pred_dir, run_meta, rope_rows)
    rope_by_id = {row["sample_id"]: row for row in rope_rows}
    if len(pred_joints) != len(sample_order):
        raise ValueError(f"prediction/order length mismatch: pred={len(pred_joints)} order={len(sample_order)}")

    error_rows = []
    for sample_id, joints in zip(sample_order, pred_joints, strict=True):
        if sample_id not in rope_by_id:
            raise ValueError(f"missing rope label for sample {sample_id}")
        gt = rope_by_id[sample_id]
        pred_distances = rope_distances_for_joints(dataset_name, joints)
        pred_norms = []
        norm_err, dist_err, valid = [], [], []
        gt_fist_ratio = gt.get("normalization", {}).get("fist_ratio", 0.5)
        for idx in range(len(FINGER_ORDER)):
            is_valid = bool(gt["rope_valid"][idx] and pred_distances[idx] is not None)
            valid.append(is_valid)
            pred_norm = normalize_rope_distance(
                pred_distances[idx],
                gt.get("rope_chain_m", [None] * len(FINGER_ORDER))[idx],
                fist_ratio=gt_fist_ratio,
            )
            pred_norms.append(pred_norm)
            norm_err.append(abs_or_none(pred_norm, gt["rope_norm"][idx]) if is_valid else None)
            dist_err.append(abs_or_none(pred_distances[idx], gt["rope_dist_m"][idx]) if is_valid else None)
        error_rows.append({
            "sample_id": sample_id,
            "dataset": dataset_name,
            "finger_order": FINGER_ORDER,
            "rope_valid": valid,
            "pred_rope_norm": pred_norms,
            "gt_rope_norm": gt["rope_norm"],
            "rope_norm_abs_error": norm_err,
            "pred_rope_dist_m": pred_distances,
            "gt_rope_dist_m": gt["rope_dist_m"],
            "rope_dist_abs_error_m": dist_err,
            "rope_norm_mae": mean_present(norm_err),
        })

    scores = summarize_errors(error_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "rope_errors.jsonl", error_rows)
    (output_dir / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "scores.txt").open("w", encoding="utf-8") as f:
        for key in ("num_samples", "num_valid_fingers", "rope_norm_mae", "rope_dist_mae_m"):
            f.write(f"{key}: {scores[key]}\n")
        for finger, value in zip(FINGER_ORDER, scores["per_finger_rope_norm_mae"], strict=True):
            f.write(f"rope_norm_mae_{finger}: {value}\n")
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score predicted rope values against GT rope JSONL labels.")
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("rope_labels", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", choices=["freihand", "ho3d"], default=None)
    parser.add_argument("--run-meta", type=Path, default=None)
    parser.add_argument("--pred-file-name", default="pred.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = score_rope_predictions(
        args.pred_dir,
        args.rope_labels,
        args.output_dir,
        dataset=args.dataset,
        run_meta=args.run_meta,
        pred_file_name=args.pred_file_name,
    )
    print(f"Scored {scores['num_samples']} samples: {args.output_dir / 'scores.txt'}")


if __name__ == "__main__":
    main()
