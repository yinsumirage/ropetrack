#!/usr/bin/env python3
"""Apply a learned visibility gate to a causal trusted-pose state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ropetrack.refine.cache import align_rows_by_sample_id, validate_cache  # noqa: E402
from ropetrack.refine.apply import load_mano_globals, mano_layer  # noqa: E402
from ropetrack.eval.temporal_metrics import _segments, _validate_order  # noqa: E402
from scripts.legacy.temporal.temporal_oracle_state import _prepare_state, _refine_masked, _write_method  # noqa: E402
from scripts.legacy.temporal.temporal_state_followups import load_pose  # noqa: E402


DEFAULT_GATE_SPECS = (
    ("image_linear_default", "image_linear_scores.npz", "default_050"),
    ("image_linear_zero", "image_linear_scores.npz", "val_zero_false_clean"),
    ("image_cached_linear_default", "image_cached_linear_scores.npz", "default_050"),
    ("image_cached_linear_zero", "image_cached_linear_scores.npz", "val_zero_false_clean"),
)


def gate_specs(values: list[list[str]] | None) -> tuple[tuple[str, str, str], ...]:
    return tuple(tuple(value) for value in values) if values else DEFAULT_GATE_SPECS


def load_gate(path: Path, order: list[str], threshold_name: str) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as loaded:
        ids = [str(value) for value in loaded["sample_id"]]
        score = np.asarray(loaded["score"], dtype=np.float64)
        thresholds = json.loads(str(loaded["thresholds_json"]))
    score = score[align_rows_by_sample_id(order, ids)]
    if score.shape != (len(order),) or not np.isfinite(score).all():
        raise ValueError(f"gate scores must be finite {(len(order),)}, got {score.shape}")
    if threshold_name not in thresholds:
        raise ValueError(f"threshold {threshold_name!r} missing from {path}")
    return score, float(thresholds[threshold_name])


def causal_trusted_pose(base_pose: np.ndarray, sample_ids: np.ndarray, freeze: np.ndarray) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float32)
    frozen = np.asarray(freeze, dtype=bool)
    if pose.ndim != 2 or pose.shape[1] != 45 or frozen.shape != (len(pose),):
        raise ValueError("base_pose must be [N,45] and freeze must be [N]")
    state = pose.copy()
    for rows in _segments(sample_ids, raw_frame_step=1):
        trusted = pose[rows[0]].copy()
        for row in rows:
            if not frozen[row]:
                trusted = pose[row].copy()
            state[row] = trusted
    return state


def shuffle_selected_rope(
    cache: dict[str, np.ndarray],
    selected: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in cache.items()}
    rng = np.random.default_rng(seed)
    for rows in _segments(cache["sample_id"], raw_frame_step=1):
        chosen = rows[selected[rows]]
        if len(chosen) > 1:
            permutation = rng.permutation(chosen)
            for key in ("input_rope_norm", "rope_valid"):
                out[key][chosen] = cache[key][permutation]
    validate_cache(out)
    return out


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    order = _validate_order(args.run_meta)
    with np.load(args.eval_cache, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    validate_cache(cache)
    if [str(value) for value in cache["sample_id"]] != order:
        raise ValueError("eval cache sample order mismatch")

    base_pose = np.asarray(cache["base_hand_pose"], dtype=np.float32)
    baseline_pose = load_pose(args.k1_method_dir, len(order))
    baseline_alpha = np.asarray(np.load(args.k1_method_dir / "alpha.npy", allow_pickle=False), dtype=np.float32)
    if baseline_alpha.shape != (len(order), 15):
        raise ValueError("K1 alpha must be [N,15]")
    orient, betas, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
    mano = mano_layer(args.device)
    method_root = args.out_dir / "methods"
    method_root.mkdir(parents=True)
    created = []
    for name, score_file, threshold_name in gate_specs(args.gate):
        score_path = Path(score_file)
        score, threshold = load_gate(score_path if score_path.is_absolute() else args.score_dir / score_path,
                                     order, threshold_name)
        freeze = score >= threshold
        state_pose = causal_trusted_pose(base_pose, cache["sample_id"], freeze)
        state_cache, state_base_xyz = _prepare_state(
            args.dataset, cache, state_pose, betas, args.mano_cache, args.device, args.batch_size, mano
        )
        variants = [(name, state_cache, False)]
        if name == args.shuffle_method:
            variants.append((name + "_rope_shuffled", shuffle_selected_rope(state_cache, freeze, args.shuffle_seed), True))
        for output_name, method_cache, shuffled in variants:
            refined, alpha, elapsed = _refine_masked(
                method_cache,
                state_pose,
                baseline_pose,
                baseline_alpha,
                np.flatnonzero(freeze),
                betas,
                orient,
                args.k1_checkpoint,
                args.device,
                args.batch_size,
                mano,
            )
            _write_method(
                method_root / output_name,
                cache["sample_id"],
                args.dataset,
                state_base_xyz,
                refined,
                betas,
                method_cache,
                args.mano_cache,
                args.device,
                args.batch_size,
                mano,
                elapsed,
                {
                    "state": "causal_last_predicted_clean",
                    "gate": score_file,
                    "threshold_name": threshold_name,
                    "threshold": threshold,
                    "frozen_fraction": float(freeze.mean()),
                    "phase_is_input": False,
                    "rope_shuffled": shuffled,
                },
                alpha,
            )
            created.append(output_name)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "methods": created,
        "num_samples": len(order),
        "protocol": {
            "state_update": "causal update when predicted clean; hold when predicted masked",
            "phase_is_model_input": False,
            "shuffle_seed": args.shuffle_seed,
        },
    }, indent=2, sort_keys=True) + "\n")
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ho3d",), default="ho3d")
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--k1-method-dir", type=Path, required=True)
    parser.add_argument("--k1-checkpoint", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--gate", nargs=3, action="append", metavar=("NAME", "SCORE_FILE", "THRESHOLD"),
                        help="Method spec; repeat to override the four default image-gate methods.")
    parser.add_argument("--shuffle-method", default="image_cached_linear_default",
                        help="Also emit a rope-shuffled control for this method name; empty disables it.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shuffle-seed", type=int, default=20260715)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    out = run(parse_args(argv))
    print(f"Learned visibility state methods written to: {out}")
    return out


if __name__ == "__main__":
    main()
