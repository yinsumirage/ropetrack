#!/usr/bin/env python3
"""Use current rope residuals to arbitrate current versus trusted finger state."""

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
from ropetrack.eval.temporal_metrics import _load_episode_manifest, _validate_order  # noqa: E402
from scripts.legacy.temporal.temporal_oracle_state import _prepare_state, _write_method, complete_episode_rows  # noqa: E402
from scripts.legacy.temporal.temporal_state_followups import load_pose, mix_finger_poses  # noqa: E402


def load_refined_residuals(path: Path, order: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as loaded:
        ids = [str(value) for value in loaded["sample_id"]]
        residual = np.asarray(loaded["refined_rope_residual"], dtype=np.float32)
        valid = np.asarray(loaded["rope_valid"], dtype=bool)
    index = align_rows_by_sample_id(order, ids)
    residual, valid = residual[index], valid[index]
    if residual.shape != (len(order), 5) or valid.shape != residual.shape:
        raise ValueError(f"rope residuals must be {(len(order), 5)}, got {residual.shape}")
    return np.where(valid & np.isfinite(residual), residual, np.inf)


def shuffled_episode_rows(values: np.ndarray, episodes: tuple[np.ndarray, ...], seed: int) -> np.ndarray:
    out = values.copy()
    rng = np.random.default_rng(seed)
    for rows in episodes:
        masked = rows[30:90]
        out[masked] = values[rng.permutation(masked)]
    return out


def arbitration_gate(
    current_residual: np.ndarray,
    state_residual: np.ndarray,
    masked_rows: np.ndarray,
    margin: float,
) -> np.ndarray:
    if current_residual.shape != state_residual.shape or current_residual.shape[1:] != (5,):
        raise ValueError("current and state rope residuals must match [N,5]")
    gate = np.zeros(current_residual.shape, dtype=bool)
    gate[masked_rows] = state_residual[masked_rows] + margin < current_residual[masked_rows]
    return gate


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    order = _validate_order(args.run_meta)
    manifest = _load_episode_manifest(args.hard_manifest, order, raw_frame_step=1)
    episodes = complete_episode_rows(manifest)
    masked_rows = np.sort(np.concatenate([rows[30:90] for rows in episodes]))
    with np.load(args.eval_cache, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    validate_cache(cache)
    if [str(value) for value in cache["sample_id"]] != order:
        raise ValueError("eval cache sample order mismatch")

    current_pose = load_pose(args.current_method_dir, len(order))
    state_pose = load_pose(args.state_method_dir, len(order))
    current_residual = load_refined_residuals(args.current_method_dir / "rope_residuals.npz", order)
    state_residual = load_refined_residuals(args.state_method_dir / "rope_residuals.npz", order)
    shuffled_residual = shuffled_episode_rows(state_residual, episodes, args.shuffle_seed)
    _, betas, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
    mano = mano_layer(args.device)
    current_cache, current_base_xyz = _prepare_state(
        args.dataset, cache, cache["base_hand_pose"], betas, args.mano_cache, args.device, args.batch_size, mano
    )

    method_root = args.out_dir / "methods"
    method_root.mkdir(parents=True)
    created = []
    for margin in args.margins:
        tag = str(margin).replace(".", "p")
        specs = [(f"rope_select_margin{tag}", state_residual, False)]
        if margin == 0.0:
            specs.append((f"rope_select_margin{tag}_shuffled_gate", shuffled_residual, True))
        for name, candidate_residual, shuffled in specs:
            gate = arbitration_gate(current_residual, candidate_residual, masked_rows, margin)
            mixed = mix_finger_poses(current_pose, state_pose, gate)
            _write_method(
                method_root / name,
                cache["sample_id"],
                args.dataset,
                current_base_xyz,
                mixed,
                betas,
                current_cache,
                args.mano_cache,
                args.device,
                args.batch_size,
                mano,
                0.0,
                {
                    "state": "last_clean_k1",
                    "gate": "per_finger_refined_rope_residual",
                    "margin": margin,
                    "shuffled_gate_control": shuffled,
                    "phase_gate": "known_analysis_only",
                    "state_fraction_masked": float(gate[masked_rows].mean()),
                },
            )
            created.append(name)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "methods": created,
        "num_complete_episodes": len(episodes),
        "num_masked_rows": len(masked_rows),
        "margins": args.margins,
        "protocol": {"phase_usage": "analysis-only masked rows", "shuffle_seed": args.shuffle_seed},
    }, indent=2, sort_keys=True) + "\n")
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ho3d",), default="ho3d")
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--current-method-dir", type=Path, required=True)
    parser.add_argument("--state-method-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--margins", type=float, nargs="+", default=(0.0, 0.02, 0.05))
    parser.add_argument("--shuffle-seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    out = run(parse_args(argv))
    print(f"Temporal rope state arbitration written to: {out}")
    return out


if __name__ == "__main__":
    main()
