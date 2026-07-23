#!/usr/bin/env python3
"""Probe whether cached pose/beta/rope consistency can safely gate state updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ropetrack.refine.cache import align_rows_by_sample_id, validate_cache  # noqa: E402
from ropetrack.eval.temporal_metrics import _load_episode_manifest, _segments, _validate_order  # noqa: E402
from scripts.legacy.temporal.temporal_oracle_state import complete_episode_rows  # noqa: E402


FEATURE_NAMES = (
    "rope_residual_mean",
    "rope_residual_max",
    "rope_valid_fraction",
    "pose_delta_mean",
    "pose_delta_max",
    "pose_acceleration_mean",
    "beta_delta_l2",
    "pose_ema_deviation",
    "beta_ema_deviation",
)


def load_cache(cache_path: Path, mano_cache_path: Path, order: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    validate_cache(cache)
    cache_ids = [str(value) for value in cache["sample_id"]]
    if cache_ids != order:
        raise ValueError("refiner cache sample order mismatch")
    with np.load(mano_cache_path, allow_pickle=False) as loaded:
        mano_ids = [str(value) for value in loaded["sample_id"]]
        betas = np.asarray(loaded["base_betas"], dtype=np.float32)
    betas = betas[align_rows_by_sample_id(order, mano_ids)]
    if betas.shape != (len(order), 10) or not np.isfinite(betas).all():
        raise ValueError(f"MANO betas must be finite {(len(order), 10)}, got {betas.shape}")
    return cache, betas


def build_features(cache: dict[str, np.ndarray], betas: np.ndarray) -> np.ndarray:
    pose = np.asarray(cache["base_hand_pose"], dtype=np.float32)
    base_rope = np.asarray(cache["base_rope_norm"], dtype=np.float32)
    input_rope = np.asarray(cache["input_rope_norm"], dtype=np.float32)
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    if betas.shape != (len(pose), 10):
        raise ValueError("betas must match cache rows")
    residual = np.where(valid, np.abs(base_rope - input_rope), 0.0)
    counts = valid.sum(axis=1)
    features = np.zeros((len(pose), len(FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = residual.sum(axis=1) / np.maximum(counts, 1)
    features[:, 1] = residual.max(axis=1)
    features[:, 2] = counts / 5.0

    for rows in _segments(cache["sample_id"], raw_frame_step=1):
        pose_joint = pose[rows].reshape(len(rows), 15, 3)
        pose_delta = np.diff(pose_joint, axis=0, prepend=pose_joint[:1])
        beta_delta = np.diff(betas[rows], axis=0, prepend=betas[rows[:1]])
        pose_speed = np.linalg.norm(pose_delta, axis=-1)
        pose_acceleration = np.diff(pose_delta, axis=0, prepend=pose_delta[:1])
        features[rows, 3] = pose_speed.mean(axis=1)
        features[rows, 4] = pose_speed.max(axis=1)
        features[rows, 5] = np.linalg.norm(pose_acceleration, axis=-1).mean(axis=1)
        features[rows, 6] = np.linalg.norm(beta_delta, axis=1)
        pose_ema = pose[rows[0]].copy()
        beta_ema = betas[rows[0]].copy()
        for row in rows:
            features[row, 7] = np.linalg.norm(pose[row] - pose_ema) / np.sqrt(45.0)
            features[row, 8] = np.linalg.norm(betas[row] - beta_ema) / np.sqrt(10.0)
            pose_ema = 0.95 * pose_ema + 0.05 * pose[row]
            beta_ema = 0.95 * beta_ema + 0.05 * betas[row]
    if not np.isfinite(features).all():
        raise ValueError("visibility features must be finite")
    return features


def sequence_split(manifest: list[dict], wanted: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    segments = sorted({str(manifest[index]["segment_id"]) for index in wanted})
    rng = np.random.default_rng(seed)
    val_count = max(1, int(round(len(segments) * val_fraction)))
    val_segments = set(rng.permutation(segments)[:val_count].tolist())
    is_val = np.asarray([str(manifest[index]["segment_id"]) in val_segments for index in wanted])
    return wanted[~is_val], wanted[is_val]


class Gate(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Linear(feature_dim, 1) if hidden == 0 else torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(1)


def fit_gate(features: np.ndarray, labels: np.ndarray, rows: np.ndarray, hidden: int, seed: int) -> tuple[Gate, np.ndarray, np.ndarray]:
    mean = features[rows].mean(axis=0)
    std = features[rows].std(axis=0)
    std[std < 1e-6] = 1.0
    x = torch.from_numpy(((features[rows] - mean) / std).astype(np.float32))
    y = torch.from_numpy(labels[rows].astype(np.float32))
    torch.manual_seed(seed)
    model = Gate(features.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def probabilities(model: Gate, features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(((features - mean) / std).astype(np.float32))
        return torch.sigmoid(model(x)).numpy().astype(np.float64)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(positive.sum())
    n_neg = len(scores) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    episodes: tuple[np.ndarray, ...],
    threshold: float,
) -> dict:
    masked = labels[rows]
    predicted_masked = scores[rows] >= threshold
    wanted = np.zeros(len(labels), dtype=bool)
    wanted[rows] = True
    selected_episodes = tuple(episode for episode in episodes if wanted[episode].all())
    first_rows = np.asarray([episode[30] for episode in selected_episodes], dtype=np.int64)
    any_false_clean = 0
    three_false_clean = 0
    for episode in selected_episodes:
        false_clean = scores[episode[30:90]] < threshold
        any_false_clean += int(false_clean.any())
        three_false_clean += int(np.convolve(false_clean.astype(np.int8), np.ones(3, dtype=np.int8), "valid").max() >= 3)
    return {
        "threshold": float(threshold),
        "masked_false_clean_rate": float((~predicted_masked[masked]).mean()),
        "clean_false_freeze_rate": float(predicted_masked[~masked].mean()),
        "first_mask_miss_rate": float((scores[first_rows] < threshold).mean()),
        "episodes_any_false_clean": int(any_false_clean),
        "episodes_three_consecutive_false_clean": int(three_false_clean),
    }


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    episodes: tuple[np.ndarray, ...],
    thresholds: dict[str, float],
) -> dict:
    return {
        "auc": auc(labels[rows], scores[rows]),
        "num_rows": int(len(rows)),
        "num_masked": int(labels[rows].sum()),
        "thresholds": {
            name: gate_metrics(labels, scores, rows, episodes, threshold)
            for name, threshold in thresholds.items()
        },
    }


def run(args: argparse.Namespace) -> dict:
    train_order = _validate_order(args.train_run_meta)
    eval_order = _validate_order(args.eval_run_meta)
    train_manifest = _load_episode_manifest(args.train_manifest, train_order, 1)
    eval_manifest = _load_episode_manifest(args.eval_manifest, eval_order, 1)
    train_episodes = complete_episode_rows(train_manifest)
    eval_episodes = complete_episode_rows(eval_manifest)
    train_cache, train_betas = load_cache(args.train_cache, args.train_mano_cache, train_order)
    eval_cache, eval_betas = load_cache(args.eval_cache, args.eval_mano_cache, eval_order)
    train_features = build_features(train_cache, train_betas)
    eval_features = build_features(eval_cache, eval_betas)
    train_labels = np.asarray([row["episode_phase"] == "masked" for row in train_manifest])
    eval_labels = np.asarray([row["episode_phase"] == "masked" for row in eval_manifest])
    train_rows_all = np.sort(np.concatenate(train_episodes))
    eval_rows = np.sort(np.concatenate(eval_episodes))
    train_rows, val_rows = sequence_split(train_manifest, train_rows_all, 0.2, args.seed)

    report = {"feature_names": FEATURE_NAMES, "models": {}, "protocol": {"phase_is_model_input": False}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, hidden in (("linear", 0), ("mlp16", 16)):
        model, mean, std = fit_gate(train_features, train_labels, train_rows, hidden, args.seed)
        train_scores = probabilities(model, train_features, mean, std)
        eval_scores = probabilities(model, eval_features, mean, std)
        masked_val = np.sort(train_scores[val_rows][train_labels[val_rows]])
        thresholds = {
            "default_050": 0.5,
            "val_zero_false_clean": float(np.nextafter(masked_val[0], -np.inf)),
            "val_0p1pct_false_clean": float(np.quantile(masked_val, 0.001)),
        }
        report["models"][name] = {
            "validation": evaluate(train_labels, train_scores, val_rows, train_episodes, thresholds),
            "evaluation": evaluate(eval_labels, eval_scores, eval_rows, eval_episodes, thresholds),
        }
        torch.save(
            {"state_dict": model.state_dict(), "hidden": hidden, "mean": mean, "std": std, "thresholds": thresholds},
            args.output_dir / f"{name}.pt",
        )
    (args.output_dir / "visibility_probe.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--train-mano-cache", type=Path, required=True)
    parser.add_argument("--train-run-meta", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--eval-mano-cache", type=Path, required=True)
    parser.add_argument("--eval-run-meta", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run(args)
    print(f"Visibility probe written to: {args.output_dir / 'visibility_probe.json'}")
    return report


if __name__ == "__main__":
    main()
