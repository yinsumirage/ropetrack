#!/usr/bin/env python3
"""Probe whether frozen image features can safely gate temporal state updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ropetrack.refine.alpha_student import load_image_feature_cache  # noqa: E402
from ropetrack.refine.cache import align_rows_by_sample_id  # noqa: E402
from scripts.legacy.temporal.probe_visibility_gate import (  # noqa: E402
    Gate,
    auc,
    build_features,
    evaluate,
    load_cache,
    probabilities,
    sequence_split,
)
from ropetrack.eval.temporal_metrics import _load_episode_manifest, _validate_order  # noqa: E402
from scripts.legacy.temporal.temporal_oracle_state import complete_episode_rows  # noqa: E402


def aligned_image_features(path: Path, order: list[str]) -> np.ndarray:
    sample_ids, features = load_image_feature_cache(path)
    features = features[align_rows_by_sample_id(order, sample_ids)]
    if features.ndim != 2 or len(features) != len(order) or not np.isfinite(features).all():
        raise ValueError(f"image features must be finite [N,C], got {features.shape}")
    return features


def fit_gate_minibatch(
    features: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    hidden: int,
    seed: int,
    steps: int,
    batch_size: int,
) -> tuple[Gate, np.ndarray, np.ndarray]:
    mean = features[rows].mean(axis=0)
    std = features[rows].std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = ((features - mean) / std).astype(np.float32)
    x = torch.from_numpy(normalized)
    y = torch.from_numpy(labels.astype(np.float32))
    train_rows = torch.from_numpy(rows.astype(np.int64))
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = Gate(features.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for _ in range(steps):
        batch = train_rows[torch.randint(len(train_rows), (min(batch_size, len(train_rows)),), generator=generator)]
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[batch]), y[batch])
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def run(args: argparse.Namespace) -> dict:
    train_order = _validate_order(args.train_run_meta)
    eval_order = _validate_order(args.eval_run_meta)
    train_manifest = _load_episode_manifest(args.train_manifest, train_order, 1)
    eval_manifest = _load_episode_manifest(args.eval_manifest, eval_order, 1)
    train_episodes = complete_episode_rows(train_manifest)
    eval_episodes = complete_episode_rows(eval_manifest)
    train_labels = np.asarray([row["episode_phase"] == "masked" for row in train_manifest])
    eval_labels = np.asarray([row["episode_phase"] == "masked" for row in eval_manifest])
    train_rows_all = np.sort(np.concatenate(train_episodes))
    eval_rows = np.sort(np.concatenate(eval_episodes))
    train_rows, val_rows = sequence_split(train_manifest, train_rows_all, 0.2, args.seed)

    train_image = aligned_image_features(args.train_image_features, train_order)
    eval_image = aligned_image_features(args.eval_image_features, eval_order)
    train_cache, train_betas = load_cache(args.train_cache, args.train_mano_cache, train_order)
    eval_cache, eval_betas = load_cache(args.eval_cache, args.eval_mano_cache, eval_order)
    train_cached = build_features(train_cache, train_betas)
    eval_cached = build_features(eval_cache, eval_betas)
    feature_sets = {
        "image": (train_image, eval_image),
        "image_cached": (
            np.concatenate([train_image, train_cached], axis=1),
            np.concatenate([eval_image, eval_cached], axis=1),
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "models": {},
        "protocol": {
            "phase_is_model_input": False,
            "train_val_split": "sequence_disjoint_complete_episodes",
            "image_features": "frozen_wilor_backbone_mean_pool",
        },
    }
    for feature_name, (train_features, eval_features) in feature_sets.items():
        report["models"][feature_name] = {}
        for model_name, hidden in (("linear", 0), ("mlp16", 16)):
            model, mean, std = fit_gate_minibatch(
                train_features, train_labels, train_rows, hidden, args.seed, args.steps, args.batch_size
            )
            train_scores = probabilities(model, train_features, mean, std)
            eval_scores = probabilities(model, eval_features, mean, std)
            masked_val = np.sort(train_scores[val_rows][train_labels[val_rows]])
            thresholds = {
                "default_050": 0.5,
                "val_zero_false_clean": float(np.nextafter(masked_val[0], -np.inf)),
                "val_0p1pct_false_clean": float(np.quantile(masked_val, 0.001)),
            }
            report["models"][feature_name][model_name] = {
                "feature_dim": int(train_features.shape[1]),
                "validation": evaluate(train_labels, train_scores, val_rows, train_episodes, thresholds),
                "evaluation": evaluate(eval_labels, eval_scores, eval_rows, eval_episodes, thresholds),
            }
            stem = f"{feature_name}_{model_name}"
            torch.save(
                {"state_dict": model.state_dict(), "hidden": hidden, "mean": mean, "std": std,
                 "thresholds": thresholds},
                args.output_dir / f"{stem}.pt",
            )
            np.savez(
                args.output_dir / f"{stem}_scores.npz",
                sample_id=np.asarray(eval_order),
                score=eval_scores.astype(np.float32),
                thresholds_json=np.asarray(json.dumps(thresholds)),
            )
    (args.output_dir / "image_visibility_probe.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-image-features", type=Path, required=True)
    parser.add_argument("--eval-image-features", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-mano-cache", type=Path, required=True)
    parser.add_argument("--train-run-meta", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--eval-mano-cache", type=Path, required=True)
    parser.add_argument("--eval-run-meta", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run(args)
    print(f"Image visibility probe written to: {args.output_dir / 'image_visibility_probe.json'}")
    return report


if __name__ == "__main__":
    main()
