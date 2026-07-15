#!/usr/bin/env python3
"""Evaluate a frozen visibility gate on an all-occluded feature cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ropetrack.refine.alpha_student import load_image_feature_cache  # noqa: E402
from scripts.probe_visibility_gate import Gate, probabilities  # noqa: E402


def longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def hard_metrics(sample_ids: list[str], scores: np.ndarray, threshold: float) -> dict:
    false_clean = scores < threshold
    sequences: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        sequences.setdefault(sample_id.split("/", 1)[0], []).append(index)
    runs = [longest_run(false_clean[rows]) for rows in sequences.values()]
    return {
        "threshold": float(threshold),
        "hard_false_clean_rate": float(false_clean.mean()),
        "hard_detected_rate": float((~false_clean).mean()),
        "sequences_any_false_clean": int(sum(run > 0 for run in runs)),
        "sequences_three_consecutive_false_clean": int(sum(run >= 3 for run in runs)),
        "max_consecutive_false_clean": int(max(runs, default=0)),
    }


def run(args: argparse.Namespace) -> dict:
    sample_ids, features = load_image_feature_cache(args.feature_cache)
    manifest = [json.loads(line) for line in args.hard_manifest.read_text().splitlines() if line.strip()]
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if sample_ids != manifest_ids:
        raise ValueError("feature cache and hard manifest sample order mismatch")
    if args.expected_count and len(sample_ids) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} rows, got {len(sample_ids)}")

    # Trusted checkpoint produced by scripts/probe_image_visibility_gate.py;
    # it contains NumPy normalization arrays in addition to tensor weights.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if features.shape[1] != len(mean) or mean.shape != std.shape:
        raise ValueError(f"feature/checkpoint dimension mismatch: {features.shape}, {mean.shape}, {std.shape}")
    model = Gate(features.shape[1], int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"])
    scores = probabilities(model.eval(), features, mean, std)
    thresholds = {str(key): float(value) for key, value in payload["thresholds"].items()}
    effects = sorted({str(row.get("effect")) for row in manifest})
    severities = sorted({float(row.get("severity")) for row in manifest})
    report = {
        "num_samples": len(sample_ids),
        "num_sequences": len({sample_id.split("/", 1)[0] for sample_id in sample_ids}),
        "effects": effects,
        "severities": severities,
        "score_quantiles": {
            str(q): float(np.quantile(scores, q)) for q in (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
        },
        "thresholds": {
            name: hard_metrics(sample_ids, scores, threshold) for name, threshold in thresholds.items()
        },
        "protocol": {"phase_is_input": False, "checkpoint_is_frozen": True, "all_rows_are_hard": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run(args)
    print(f"Visibility shift report written to: {args.output}")
    return report


if __name__ == "__main__":
    main()
