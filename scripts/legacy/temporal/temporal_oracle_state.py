#!/usr/bin/env python3
"""Deterministic clean-prefix state oracles for dense HO3D episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ropetrack.backends.hand_predictor import _rotmat_to_aa  # noqa: E402
from ropetrack.refine.actions import apply_action_np  # noqa: E402
from ropetrack.refine.analysis import json_sanitize  # noqa: E402
from ropetrack.refine.cache import validate_cache  # noqa: E402
from ropetrack.refine.temporal import temporal_alpha  # noqa: E402
from ropetrack.refine.apply import (  # noqa: E402
    aa_to_rotmat,
    compute_flex_directions,
    decoded_rope_norm,
    expand_gate_to_alpha,
    gate_from_cache,
    load_mano_globals,
    mano_layer,
    mano_predictions,
    rope_residual_report,
    write_pred,
)
from ropetrack.eval.temporal_metrics import _load_episode_manifest, _validate_order  # noqa: E402


def complete_episode_rows(manifest: list[dict]) -> tuple[np.ndarray, ...]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for row_index, row in enumerate(manifest):
        if row["episode_phase"] != "tail":
            grouped.setdefault((str(row["segment_id"]), str(row["episode_id"])), []).append(row_index)
    episodes = []
    expected = ["context"] * 30 + ["masked"] * 60 + ["recovery"] * 30
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda index: manifest[index]["episode_offset"])
        if [manifest[index]["episode_offset"] for index in rows] != list(range(120)):
            raise ValueError(f"episode {key} must contain offsets 0..119")
        if [manifest[index]["episode_phase"] for index in rows] != expected:
            raise ValueError(f"episode {key} must use the registered 30/60/30 phases")
        episodes.append(np.asarray(rows, dtype=np.int64))
    if not episodes:
        raise ValueError("manifest contains no complete episodes")
    return tuple(episodes)


def extrapolate_pose(last_five_pose: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """SO(3) zero/finite-CV/damped extrapolation for one 45D MANO pose."""
    poses = np.asarray(last_five_pose, dtype=np.float32)
    if poses.shape != (5, 45) or not np.isfinite(poses).all():
        raise ValueError(f"last_five_pose must be finite [5, 45], got {poses.shape}")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if mode not in {"last_clean", "constant_velocity", "damped_velocity"}:
        raise ValueError(f"unsupported oracle mode: {mode}")
    rotations = aa_to_rotmat(poses.reshape(5, 15, 3))
    relative = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
    velocity = np.median(_rotmat_to_aa(relative).reshape(4, 15, 3), axis=0)
    if mode == "last_clean":
        scale = 0.0
    elif mode == "constant_velocity":
        scale = float(min(horizon, 5))
    else:
        scale = float((1.0 - 0.8**horizon) / 0.2)
    predicted = rotations[-1] @ aa_to_rotmat(velocity * scale)
    return _rotmat_to_aa(predicted).reshape(45).astype(np.float32)


def build_oracle_states(
    base_pose: np.ndarray,
    base_betas: np.ndarray,
    episodes: tuple[np.ndarray, ...],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    pose = np.asarray(base_pose, dtype=np.float32)
    betas = np.asarray(base_betas, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 45 or betas.shape != (len(pose), 10):
        raise ValueError(f"expected pose [N,45] and betas [N,10], got {pose.shape} and {betas.shape}")
    states = {name: pose.copy() for name in ("last_clean", "constant_velocity", "damped_velocity")}
    fixed_betas = betas.copy()
    masked_rows = []
    for rows in episodes:
        context = rows[:30]
        masked = rows[30:90]
        last_five = pose[context[-5:]]
        fixed_betas[masked] = np.median(betas[context], axis=0)
        masked_rows.append(masked)
        for horizon, row in enumerate(masked, start=1):
            for mode in states:
                states[mode][row] = extrapolate_pose(last_five, horizon, mode)
    masked = np.concatenate(masked_rows)
    if len(np.unique(masked)) != len(masked):
        raise ValueError("complete episodes overlap")
    return states, fixed_betas, np.sort(masked)


def shuffle_masked_rope(
    cache: dict[str, np.ndarray], episodes: tuple[np.ndarray, ...], seed: int
) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in cache.items()}
    rng = np.random.default_rng(seed)
    for rows in episodes:
        masked = rows[30:90]
        shuffled = rng.permutation(masked)
        for key in ("input_rope_norm", "rope_valid"):
            out[key][masked] = np.asarray(cache[key])[shuffled]
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_method(
    out_dir: Path,
    sample_ids: np.ndarray,
    dataset: str,
    base_xyz: list,
    refined_pose: np.ndarray,
    betas: np.ndarray,
    cache: dict[str, np.ndarray],
    mano_cache: Path,
    device: str,
    batch_size: int,
    mano,
    elapsed_before_decode: float,
    config: dict,
    alpha: np.ndarray | None = None,
) -> None:
    started = time.perf_counter()
    xyz, verts = mano_predictions(
        dataset,
        refined_pose,
        sample_ids,
        mano_cache,
        device,
        batch_size,
        mano_module=mano,
        betas_override=betas,
    )
    elapsed = elapsed_before_decode + time.perf_counter() - started
    out_dir.mkdir(parents=True, exist_ok=False)
    write_pred(out_dir / "pred.json", xyz, verts)
    np.save(out_dir / "sample_id.npy", sample_ids)
    np.save(out_dir / "refined_hand_pose.npy", refined_pose)
    if alpha is not None:
        np.save(out_dir / "alpha.npy", alpha)
    residual_summary, residual_arrays = rope_residual_report(dataset, base_xyz, xyz, cache)
    np.savez(out_dir / "rope_residuals.npz", **residual_arrays)
    summary = {
        "num_samples": int(len(sample_ids)),
        "mode": "temporal_oracle_state",
        "action_space": "flex15" if alpha is not None else None,
        "rope_residual": residual_summary,
        "timing": {
            "wall_seconds": float(elapsed),
            "per_sample_ms": float(1000.0 * elapsed / len(sample_ids)),
        },
        "oracle": config,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_sanitize(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_state(
    dataset: str,
    cache: dict[str, np.ndarray],
    state_pose: np.ndarray,
    betas: np.ndarray,
    mano_cache: Path,
    device: str,
    batch_size: int,
    mano,
) -> tuple[dict[str, np.ndarray], list]:
    state_cache = {key: np.asarray(value).copy() for key, value in cache.items()}
    state_cache["base_hand_pose"] = np.asarray(state_pose, dtype=np.float32)
    base_xyz, _ = mano_predictions(
        dataset,
        state_pose,
        state_cache["sample_id"],
        mano_cache,
        device,
        batch_size,
        mano_module=mano,
        betas_override=betas,
        keep_vertices=False,
    )
    state_cache["base_rope_norm"] = decoded_rope_norm(dataset, base_xyz, state_cache)
    validate_cache(state_cache)
    return state_cache, base_xyz


def _refine_masked(
    cache: dict[str, np.ndarray],
    state_pose: np.ndarray,
    baseline_pose: np.ndarray,
    baseline_alpha: np.ndarray,
    masked_rows: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    checkpoint: Path,
    device: str,
    batch_size: int,
    mano,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    alpha, config = temporal_alpha(cache, checkpoint, device)
    if config["action_space"] != "flex15" or config.get("gate_threshold") is None:
        raise ValueError("oracle requires the frozen flex15 K1 checkpoint with a hard gate")
    directions = compute_flex_directions(
        torch.from_numpy(state_pose[masked_rows]).to(device),
        torch.from_numpy(global_orient[masked_rows]).to(device),
        torch.from_numpy(betas[masked_rows]).to(device),
        mano,
        batch_size,
    ).cpu().numpy().astype(np.float32)
    alpha = alpha * expand_gate_to_alpha(gate_from_cache(cache, config["gate_threshold"]), "flex15")
    refined = np.asarray(baseline_pose, dtype=np.float32).copy()
    refined[masked_rows] = apply_action_np(
        state_pose[masked_rows], alpha[masked_rows], "flex15", directions
    )
    emitted_alpha = np.asarray(baseline_alpha, dtype=np.float32).copy()
    emitted_alpha[masked_rows] = alpha[masked_rows]
    return refined, emitted_alpha, time.perf_counter() - started


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    order = _validate_order(args.run_meta)
    manifest = _load_episode_manifest(args.hard_manifest, order, raw_frame_step=1)
    episodes = complete_episode_rows(manifest)
    with np.load(args.eval_cache, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    validate_cache(cache)
    if [str(value) for value in cache["sample_id"]] != order:
        raise ValueError("eval cache sample_id order mismatch")
    baseline_ids = [str(value) for value in np.load(args.k1_method_dir / "sample_id.npy", allow_pickle=False)]
    if baseline_ids != order:
        raise ValueError("K1 sample_id order mismatch")
    baseline_pose = np.asarray(
        np.load(args.k1_method_dir / "refined_hand_pose.npy", allow_pickle=False), dtype=np.float32
    )
    baseline_alpha = np.asarray(
        np.load(args.k1_method_dir / "alpha.npy", allow_pickle=False), dtype=np.float32
    )
    if baseline_pose.shape != (len(order), 45) or not np.isfinite(baseline_pose).all():
        raise ValueError("K1 refined pose must be finite [N,45]")
    if baseline_alpha.shape != (len(order), 15) or not np.isfinite(baseline_alpha).all():
        raise ValueError("K1 alpha must be finite [N,15]")

    global_orient, base_betas, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
    states, fixed_betas, masked_rows = build_oracle_states(
        cache["base_hand_pose"], base_betas, episodes
    )
    if len(masked_rows) != 60 * len(episodes):
        raise ValueError("masked row count does not match complete episodes")
    shuffled_template = shuffle_masked_rope(cache, episodes, args.shuffle_seed)
    mano = mano_layer(args.device)
    methods_root = args.out_dir / "methods"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    methods = []

    prepared: dict[tuple[str, bool], tuple[dict[str, np.ndarray], list, np.ndarray, np.ndarray]] = {}
    for state_name, state_pose, fixed in (
        ("last_clean", states["last_clean"], False),
        ("constant_velocity", states["constant_velocity"], False),
        ("damped_velocity", states["damped_velocity"], False),
        ("current", cache["base_hand_pose"], True),
        ("last_clean", states["last_clean"], True),
        ("damped_velocity", states["damped_velocity"], True),
    ):
        beta_values = fixed_betas if fixed else base_betas
        key = (state_name, fixed)
        state_cache, base_xyz = _prepare_state(
            args.dataset,
            cache,
            state_pose,
            beta_values,
            args.mano_cache,
            args.device,
            args.batch_size,
            mano,
        )
        prepared[key] = (state_cache, base_xyz, state_pose, beta_values)

    raw_cache, raw_base_xyz, raw_state, raw_betas = prepared[("last_clean", False)]
    raw_pose = baseline_pose.copy()
    raw_pose[masked_rows] = raw_state[masked_rows]
    raw_started = time.perf_counter()
    _write_method(
        methods_root / "last_clean",
        cache["sample_id"],
        args.dataset,
        raw_base_xyz,
        raw_pose,
        raw_betas,
        raw_cache,
        args.mano_cache,
        args.device,
        args.batch_size,
        mano,
        time.perf_counter() - raw_started,
        {"state": "last_clean", "rope": False, "fixed_beta": False, "phase_gate": "known"},
    )
    methods.append("last_clean")

    method_specs = (
        ("last_clean_k1", "last_clean", False),
        ("constant_velocity_k1", "constant_velocity", False),
        ("damped_velocity_k1", "damped_velocity", False),
        ("fixed_beta_k1", "current", True),
        ("fixed_beta_last_clean_k1", "last_clean", True),
        ("fixed_beta_damped_k1", "damped_velocity", True),
    )
    for method_name, state_name, fixed in method_specs:
        state_cache, base_xyz, state_pose, beta_values = prepared[(state_name, fixed)]
        for shuffled in (False, True):
            method_cache = {key: np.asarray(value).copy() for key, value in state_cache.items()}
            if shuffled:
                for key in ("input_rope_norm", "rope_valid"):
                    method_cache[key] = np.asarray(shuffled_template[key]).copy()
            refined, alpha, elapsed = _refine_masked(
                method_cache,
                state_pose,
                baseline_pose,
                baseline_alpha,
                masked_rows,
                beta_values,
                global_orient,
                args.k1_checkpoint,
                args.device,
                args.batch_size,
                mano,
            )
            output_name = method_name + ("_rope_shuffled" if shuffled else "")
            _write_method(
                methods_root / output_name,
                cache["sample_id"],
                args.dataset,
                base_xyz,
                refined,
                beta_values,
                method_cache,
                args.mano_cache,
                args.device,
                args.batch_size,
                mano,
                elapsed,
                {
                    "state": state_name,
                    "rope": True,
                    "rope_shuffled": shuffled,
                    "fixed_beta": fixed,
                    "phase_gate": "known",
                    "velocity_frames": 5,
                    "constant_velocity_cap_frames": 5,
                    "damping": 0.8,
                },
                alpha,
            )
            methods.append(output_name)

    manifest_payload = {
        "num_samples": len(order),
        "num_complete_episodes": len(episodes),
        "num_masked_rows": len(masked_rows),
        "methods": methods,
        "k1_checkpoint": str(args.k1_checkpoint),
        "k1_checkpoint_sha256": _sha256(args.k1_checkpoint),
        "protocol": {
            "oracle_phase_usage": "gate_only",
            "deploy_model_input": False,
            "state": "45D articulated MANO hand_pose only",
            "global_orient_and_cam_t": "current WiLoR for every method",
            "velocity_frames": 5,
            "constant_velocity_cap_frames": 5,
            "damping": 0.8,
            "fixed_beta": "componentwise median of each episode's 30 clean-prefix frames",
            "shuffle_seed": args.shuffle_seed,
            "shuffle_scope": "within each episode's 60 masked frames",
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--k1-checkpoint", type=Path, required=True)
    parser.add_argument("--k1-method-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["ho3d"], default="ho3d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shuffle-seed", type=int, default=20260710)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    out = run(parse_args(argv))
    print(f"Temporal oracle predictions written to: {out}")
    return out


if __name__ == "__main__":
    main()
