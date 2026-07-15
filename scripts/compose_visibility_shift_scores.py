#!/usr/bin/env python3
"""Compose clean base scores with frozen-gate scores from a shifted hard root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ropetrack.refine.alpha_student import load_image_feature_cache  # noqa: E402
from scripts.evaluate_visibility_shift import load_gate_checkpoint  # noqa: E402
from scripts.probe_visibility_gate import evaluate, probabilities  # noqa: E402
from scripts.score_temporal_predictions import _load_episode_manifest, _validate_order  # noqa: E402
from scripts.temporal_oracle_state import complete_episode_rows  # noqa: E402


def compose_scores(base: np.ndarray, shifted: np.ndarray, masked: np.ndarray) -> np.ndarray:
    if base.shape != shifted.shape or base.shape != masked.shape:
        raise ValueError("base, shifted, and masked rows must match")
    out = np.asarray(base, dtype=np.float64).copy()
    out[masked] = shifted[masked]
    return out


def run(args: argparse.Namespace) -> dict:
    order = _validate_order(args.run_meta)
    manifest = _load_episode_manifest(args.episode_manifest, order, raw_frame_step=1)
    episodes = complete_episode_rows(manifest)
    labels = np.asarray([row["episode_phase"] == "masked" for row in manifest])
    rows = np.sort(np.concatenate(episodes))

    with np.load(args.base_scores, allow_pickle=False) as loaded:
        base_ids = [str(value) for value in loaded["sample_id"]]
        base_scores = np.asarray(loaded["score"], dtype=np.float64)
    shifted_ids, features = load_image_feature_cache(args.shifted_features)
    if base_ids != order or shifted_ids != order:
        raise ValueError("base scores, shifted features, and run order must match")
    model, mean, std, thresholds = load_gate_checkpoint(args.checkpoint, features.shape[1])
    shifted_scores = probabilities(model, features, mean, std)
    scores = compose_scores(base_scores, shifted_scores, labels)
    report = {
        "evaluation": evaluate(labels, scores, rows, episodes, thresholds),
        "num_replaced_masked_rows": int(labels.sum()),
        "protocol": {"phase_is_model_input": False, "checkpoint_is_frozen": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_id=np.asarray(order),
        score=scores.astype(np.float32),
        thresholds_json=np.asarray(json.dumps(thresholds)),
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--shifted-features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run(args)
    print(f"Composed visibility scores written to: {args.output}")
    return report


if __name__ == "__main__":
    main()
