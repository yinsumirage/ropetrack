#!/usr/bin/env python3
"""Participant-disjoint image gate and causal state screen for natural HOT3D drops."""

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
from ropetrack.refine.cache import align_rows_by_sample_id, load_sample_order, validate_cache  # noqa: E402
from scripts.legacy.temporal.hot3d_natural_state import (  # noqa: E402
    natural_episode_rows,
    refine_low_student,
)
from scripts.legacy.temporal.probe_image_visibility_gate import fit_gate_minibatch  # noqa: E402
from scripts.legacy.temporal.probe_visibility_gate import Gate, auc, probabilities  # noqa: E402
from ropetrack.refine.apply import load_mano_globals, mano_layer  # noqa: E402
from scripts.legacy.temporal.temporal_oracle_state import _prepare_state, _write_method  # noqa: E402


def participant(sample_id: str) -> str:
    value = str(sample_id).replace("\\", "/").split("/", 1)[0].split("_", 1)[0]
    if not value.startswith("P"):
        raise ValueError(f"cannot parse HOT3D participant from {sample_id!r}")
    return value


def gate_metrics(labels: np.ndarray, scores: np.ndarray, rows: np.ndarray, threshold: float = 0.5) -> dict:
    truth = np.asarray(labels, dtype=bool)[rows]
    pred = np.asarray(scores, dtype=np.float64)[rows] >= threshold
    if not truth.any() or truth.all():
        raise ValueError("gate metrics require both context and low-visibility rows")
    return {
        "rows": int(len(rows)),
        "auc": float(auc(truth.astype(np.float64), np.asarray(scores)[rows])),
        "accuracy": float((truth == pred).mean()),
        "balanced_accuracy": float(0.5 * (pred[truth].mean() + (~pred[~truth]).mean())),
        "context_false_freeze": float(pred[~truth].mean()),
        "low_visibility_missed_freeze": float((~pred[truth]).mean()),
        "predicted_freeze_fraction": float(pred.mean()),
    }


def causal_episode_state(
    base_pose: np.ndarray, episodes: tuple[np.ndarray, ...], freeze: np.ndarray
) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float32)
    selected = np.asarray(freeze, dtype=bool)
    if pose.ndim != 2 or pose.shape[1] != 45 or selected.shape != (len(pose),):
        raise ValueError("base_pose must be [N,45] and freeze must be [N]")
    state = pose.copy()
    for rows in episodes:
        trusted = pose[rows[0]].copy()
        for row in rows:
            if not selected[row]:
                trusted = pose[row].copy()
            state[row] = trusted
    return state


def shuffle_selected_rope(
    cache: dict[str, np.ndarray], episodes: tuple[np.ndarray, ...], selected: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in cache.items()}
    rng = np.random.default_rng(seed)
    for rows in episodes:
        chosen = rows[np.asarray(selected, dtype=bool)[rows]]
        if len(chosen) > 1:
            perm = rng.permutation(chosen)
            for key in ("input_rope_norm", "rope_valid"):
                out[key][chosen] = np.asarray(cache[key])[perm]
    validate_cache(out)
    return out


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    order = load_sample_order(args.run_meta, [])
    rows_by_id = {
        str(row["sample_id"]): row
        for row in (json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip())
    }
    if set(rows_by_id) != set(order):
        raise ValueError("manifest and run_meta sample ids must match exactly")
    manifest = [rows_by_id[sample_id] for sample_id in order]
    episodes = natural_episode_rows(manifest)
    labels = np.asarray([row["phase"] == "low_visibility" for row in manifest])
    participants = np.asarray([participant(sample_id) for sample_id in order])
    validation = np.isin(participants, args.validation_participant)
    train_rows = np.flatnonzero(~validation)
    validation_rows = np.flatnonzero(validation)
    if not len(train_rows) or not len(validation_rows):
        raise ValueError("participant split must have non-empty train and validation rows")

    feature_ids, features = load_image_feature_cache(args.image_features)
    features = np.asarray(features, dtype=np.float32)[align_rows_by_sample_id(order, feature_ids)]
    with np.load(args.eval_cache, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    validate_cache(cache)
    if [str(value) for value in cache["sample_id"]] != order:
        raise ValueError("eval cache sample order mismatch")
    baseline_ids = [str(value) for value in np.load(args.k1_method_dir / "sample_id.npy", allow_pickle=False)]
    baseline_pose = np.asarray(np.load(args.k1_method_dir / "refined_hand_pose.npy"), dtype=np.float32)
    baseline_alpha = np.asarray(np.load(args.k1_method_dir / "alpha.npy"), dtype=np.float32)
    if baseline_ids != order or baseline_pose.shape != (len(order), 45) or baseline_alpha.shape != (len(order), 15):
        raise ValueError("K1 method does not align with HOT3D order")

    orient, betas, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
    mano = mano_layer(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    method_root = args.out_dir / "methods"
    report = {
        "train_participants": sorted(set(participants[~validation].tolist())),
        "validation_participants": sorted(set(participants[validation].tolist())),
        "phase_is_model_input": False,
        "models": {},
    }
    created = []
    for name, hidden in (("linear", 0), ("mlp16", 16)):
        model, mean, std = fit_gate_minibatch(
            features, labels, train_rows, hidden, args.seed, args.steps, args.batch_size
        )
        scores = probabilities(model, features, mean, std)
        freeze = scores >= 0.5
        report["models"][name] = {
            "train": gate_metrics(labels, scores, train_rows),
            "validation": gate_metrics(labels, scores, validation_rows),
        }
        torch.save(
            {"state_dict": model.state_dict(), "hidden": hidden, "mean": mean, "std": std, "threshold": 0.5},
            args.out_dir / f"{name}.pt",
        )
        np.savez(args.out_dir / f"{name}_scores.npz", sample_id=np.asarray(order), score=scores)

        state_pose = causal_episode_state(cache["base_hand_pose"], episodes, freeze)
        state_cache, state_xyz = _prepare_state(
            "hot3d", cache, state_pose, betas, args.mano_cache, args.device, args.refine_batch_size, mano
        )
        variants = [(name, state_cache, False)]
        if name == "linear":
            variants.append((name + "_rope_shuffled", shuffle_selected_rope(
                state_cache, episodes, freeze, args.seed
            ), True))
        for output_name, method_cache, shuffled in variants:
            refined, alpha, elapsed = refine_low_student(
                method_cache, state_pose, baseline_pose, baseline_alpha, np.flatnonzero(freeze),
                betas, orient, args.k1_checkpoint, args.device, args.refine_batch_size, mano,
            )
            _write_method(
                method_root / output_name, cache["sample_id"], "hot3d", state_xyz, refined, betas,
                method_cache, args.mano_cache, args.device, args.refine_batch_size, mano, elapsed,
                {
                    "state": "causal_last_predicted_clean",
                    "gate": name,
                    "threshold": 0.5,
                    "phase_is_input": False,
                    "predicted_freeze_fraction": float(freeze.mean()),
                    "rope_shuffled": shuffled,
                }, alpha,
            )
            created.append(output_name)

    report["methods"] = created
    (args.out_dir / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--k1-method-dir", type=Path, required=True)
    parser.add_argument("--k1-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-participant", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--refine-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    output = run(parse_args(argv))
    print(f"HOT3D natural gate screen written to: {output}")
    return output


if __name__ == "__main__":
    main()
