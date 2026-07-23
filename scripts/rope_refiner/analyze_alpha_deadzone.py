#!/usr/bin/env python3
"""Multiplicative dead-zone probe for rope optimization outputs.

Hypothesis H2 (experience/0029_p0_oracle_slice_deadzone_tooling.md): multiplicative
curl scaling cannot correct fingers whose base pose is near zero (predicted
open), because the correction is proportional to the existing bend. If true,
residual closure should shrink with base finger pose magnitude under mult
action spaces, and the effect should disappear under flex15.

Inputs are the artifacts written by apply_rope_refinement.py --mode optimize:
refiner_eval_cache.npz (base pose), alpha.npy, rope_residuals.npz.
Numpy-only; safe on CPU scoring nodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.actions import (  # noqa: E402
    ACTION_SPACES,
    per_finger_alpha_abs,
    per_finger_pose_magnitude,
)
from ropetrack.refine.analysis import (  # noqa: E402
    bucket_indices,
    json_sanitize,
    pearson,
    quantile_bucket_edges,
    spearman,
)
from ropetrack.rope import FINGER_ORDER  # noqa: E402


def finger_stats(curl: np.ndarray, alpha_abs: np.ndarray, closure: np.ndarray) -> dict:
    return {
        "mean_curl_mag": float(np.nanmean(curl)),
        "mean_alpha_abs": float(np.nanmean(alpha_abs)),
        "mean_closure": float(np.nanmean(closure)),
        "pearson_curl_vs_alpha_abs": pearson(curl, alpha_abs),
        "spearman_curl_vs_alpha_abs": spearman(curl, alpha_abs),
        "pearson_curl_vs_closure": pearson(curl, closure),
        "spearman_curl_vs_closure": spearman(curl, closure),
    }


def curl_bucket_table(curl: np.ndarray, alpha_abs: np.ndarray, closure: np.ndarray, base_residual: np.ndarray, num_buckets: int) -> list[dict]:
    edges = quantile_bucket_edges(curl, num_buckets)
    bucket_of = bucket_indices(curl, edges)
    rows = []
    for bucket in range(num_buckets):
        selected = bucket_of == bucket
        if not selected.any():
            rows.append({"bucket": bucket, "n": 0})
            continue
        rows.append({
            "bucket": bucket,
            "n": int(selected.sum()),
            "mean_curl_mag": float(np.nanmean(curl[selected])),
            "mean_alpha_abs": float(np.nanmean(alpha_abs[selected])),
            "mean_base_residual": float(np.nanmean(base_residual[selected])),
            "mean_closure": float(np.nanmean(closure[selected])),
        })
    return rows


def build_report(
    base_hand_pose: np.ndarray,
    alpha: np.ndarray,
    base_residual: np.ndarray,
    refined_residual: np.ndarray,
    valid: np.ndarray,
    action_space: str,
    num_buckets: int,
) -> dict:
    curl = per_finger_pose_magnitude(base_hand_pose).astype(np.float64)
    alpha_abs = per_finger_alpha_abs(alpha, action_space).astype(np.float64)
    base_res = np.asarray(base_residual, dtype=np.float64)
    refined_res = np.asarray(refined_residual, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    closure = base_res - refined_res
    for array in (closure, base_res):
        array[~mask] = np.nan
    curl_masked = np.where(mask, curl, np.nan)
    alpha_masked = np.where(mask, alpha_abs, np.nan)

    per_finger = {
        finger: finger_stats(curl_masked[:, idx], alpha_masked[:, idx], closure[:, idx])
        for idx, finger in enumerate(FINGER_ORDER)
    }
    pooled_curl = curl_masked.reshape(-1)
    pooled_alpha = alpha_masked.reshape(-1)
    pooled_closure = closure.reshape(-1)
    pooled_base_res = base_res.reshape(-1)

    return {
        "action_space": action_space,
        "num_samples": int(base_hand_pose.shape[0]),
        "num_valid_fingers": int(mask.sum()),
        "per_finger": per_finger,
        "pooled": finger_stats(pooled_curl, pooled_alpha, pooled_closure),
        "curl_buckets": curl_bucket_table(pooled_curl, pooled_alpha, pooled_closure, pooled_base_res, num_buckets),
    }


def write_tsv(path: Path, report: dict) -> None:
    lines = ["section\tname\tmean_curl\tmean_alpha_abs\tmean_closure\tpearson_curl_closure\tn"]
    for finger, stats in report["per_finger"].items():
        lines.append(
            f"finger\t{finger}\t{stats['mean_curl_mag']:.4f}\t{stats['mean_alpha_abs']:.5f}\t"
            f"{stats['mean_closure']:.5f}\t{stats['pearson_curl_vs_closure']:.4f}\t-"
        )
    pooled = report["pooled"]
    lines.append(
        f"pooled\tall\t{pooled['mean_curl_mag']:.4f}\t{pooled['mean_alpha_abs']:.5f}\t"
        f"{pooled['mean_closure']:.5f}\t{pooled['pearson_curl_vs_closure']:.4f}\t{report['num_valid_fingers']}"
    )
    for row in report["curl_buckets"]:
        if row.get("n"):
            lines.append(
                f"curl_bucket\tq{row['bucket']}\t{row['mean_curl_mag']:.4f}\t{row['mean_alpha_abs']:.5f}\t"
                f"{row['mean_closure']:.5f}\t-\t{row['n']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate base finger curl with alpha magnitude and residual closure.")
    parser.add_argument("--cache", type=Path, required=True, help="refiner_eval_cache.npz")
    parser.add_argument("--alpha", type=Path, required=True, help="alpha.npy from optimize mode")
    parser.add_argument("--residuals", type=Path, required=True, help="rope_residuals.npz from apply_rope_refinement.py")
    parser.add_argument("--action-space", choices=list(ACTION_SPACES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--curl-buckets", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    with np.load(args.cache) as loaded:
        base_hand_pose = np.asarray(loaded["base_hand_pose"], dtype=np.float32)
        cache_ids = [str(sid) for sid in loaded["sample_id"]]
    alpha = np.load(args.alpha)
    with np.load(args.residuals) as loaded:
        base_residual = np.asarray(loaded["base_rope_residual"], dtype=np.float64)
        refined_residual = np.asarray(loaded["refined_rope_residual"], dtype=np.float64)
        valid = np.asarray(loaded["rope_valid"], dtype=bool)
        residual_ids = [str(sid) for sid in loaded["sample_id"]]

    if cache_ids != residual_ids:
        raise ValueError("cache/residuals sample_id order mismatch")
    if not (len(base_hand_pose) == len(alpha) == len(base_residual)):
        raise ValueError(
            f"length mismatch: pose={len(base_hand_pose)} alpha={len(alpha)} residuals={len(base_residual)}"
        )

    report = build_report(base_hand_pose, alpha, base_residual, refined_residual, valid, args.action_space, args.curl_buckets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "alpha_deadzone.json").write_text(json.dumps(json_sanitize(report), indent=2), encoding="utf-8")
    write_tsv(args.output_dir / "alpha_deadzone.tsv", report)
    print(f"Dead-zone analysis written to: {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
