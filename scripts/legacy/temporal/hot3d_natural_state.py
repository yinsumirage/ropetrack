#!/usr/bin/env python3
"""Known-phase last-context state screen for natural HOT3D visibility drops."""

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

from ropetrack.io import read_jsonl  # noqa: E402
from ropetrack.refine.actions import apply_action_np  # noqa: E402
from ropetrack.refine.alpha_student import student_alpha  # noqa: E402
from ropetrack.refine.cache import load_sample_order, validate_cache  # noqa: E402
from ropetrack.refine.apply import (  # noqa: E402
    compute_flex_directions,
    expand_gate_to_alpha,
    gate_from_cache,
    load_mano_globals,
    mano_layer,
)
from scripts.legacy.temporal.temporal_oracle_state import (  # noqa: E402
    _prepare_state,
    _write_method,
)


def natural_episode_rows(manifest: list[dict]) -> tuple[np.ndarray, ...]:
    grouped: dict[str, list[int]] = {}
    for row_index, row in enumerate(manifest):
        grouped.setdefault(str(row["episode_id"]), []).append(row_index)
    episodes = []
    for episode_id, rows in sorted(grouped.items()):
        context = sorted(
            (idx for idx in rows if manifest[idx]["phase"] == "context"),
            key=lambda idx: int(manifest[idx]["phase_index"]),
        )
        low = sorted(
            (idx for idx in rows if manifest[idx]["phase"] == "low_visibility"),
            key=lambda idx: int(manifest[idx]["phase_index"]),
        )
        if len(context) != 30 or not (1 <= len(low) <= 60):
            raise ValueError(f"episode {episode_id} requires 30 context and 1..60 low rows")
        if [int(manifest[idx]["phase_index"]) for idx in context] != list(range(1, 31)):
            raise ValueError(f"episode {episode_id} context phase_index must be 1..30")
        if [int(manifest[idx]["phase_index"]) for idx in low] != list(range(1, len(low) + 1)):
            raise ValueError(f"episode {episode_id} low phase_index must be contiguous from one")
        if len(context) + len(low) != len(rows):
            raise ValueError(f"episode {episode_id} contains an unsupported phase")
        episodes.append(np.asarray(context + low, dtype=np.int64))
    if not episodes:
        raise ValueError("manifest contains no natural HOT3D episodes")
    return tuple(episodes)


def last_context_state(base_pose: np.ndarray, episodes: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(base_pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 45 or not np.isfinite(pose).all():
        raise ValueError(f"base_pose must be finite [N,45], got {pose.shape}")
    state = pose.copy()
    low_rows = []
    for rows in episodes:
        low = rows[30:]
        state[low] = pose[rows[29]]
        low_rows.append(low)
    selected = np.concatenate(low_rows)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("natural HOT3D episodes overlap")
    return state, np.sort(selected)


def shuffle_low_rope(
    cache: dict[str, np.ndarray], episodes: tuple[np.ndarray, ...], seed: int
) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in cache.items()}
    rng = np.random.default_rng(seed)
    for rows in episodes:
        low = rows[30:]
        perm = rng.permutation(low)
        for key in ("input_rope_norm", "rope_valid"):
            out[key][low] = np.asarray(cache[key])[perm]
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refine_low_student(
    cache: dict[str, np.ndarray],
    state_pose: np.ndarray,
    baseline_pose: np.ndarray,
    baseline_alpha: np.ndarray,
    low_rows: np.ndarray,
    betas: np.ndarray,
    global_orient: np.ndarray,
    checkpoint: Path,
    device: str,
    batch_size: int,
    mano,
) -> tuple[np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    alpha, config = student_alpha(cache, checkpoint, device)
    if config["action_space"] != "flex15" or config.get("gate_threshold") is None:
        raise ValueError("state screen requires the frozen gated Flex15 student")
    directions = compute_flex_directions(
        torch.from_numpy(state_pose[low_rows]).to(device),
        torch.from_numpy(global_orient[low_rows]).to(device),
        torch.from_numpy(betas[low_rows]).to(device),
        mano,
        batch_size,
    ).cpu().numpy().astype(np.float32)
    gate = gate_from_cache(cache, float(config["gate_threshold"]))
    alpha = alpha * expand_gate_to_alpha(gate, "flex15")
    refined = np.asarray(baseline_pose, dtype=np.float32).copy()
    refined[low_rows] = apply_action_np(state_pose[low_rows], alpha[low_rows], "flex15", directions)
    emitted_alpha = np.asarray(baseline_alpha, dtype=np.float32).copy()
    emitted_alpha[low_rows] = alpha[low_rows]
    return refined, emitted_alpha, time.perf_counter() - started


def run(args: argparse.Namespace) -> Path:
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out_dir}")
    order = load_sample_order(args.run_meta, [])
    rows_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.manifest)}
    if len(rows_by_id) != len(order) or set(rows_by_id) != set(order):
        raise ValueError("manifest and run_meta sample ids must match exactly")
    manifest = [rows_by_id[sample_id] for sample_id in order]
    episodes = natural_episode_rows(manifest)

    with np.load(args.eval_cache, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    validate_cache(cache)
    if [str(value) for value in cache["sample_id"]] != order:
        raise ValueError("eval cache sample_id order mismatch")
    baseline_ids = [str(value) for value in np.load(args.k1_method_dir / "sample_id.npy", allow_pickle=False)]
    if baseline_ids != order:
        raise ValueError("K1 sample_id order mismatch")
    baseline_pose = np.asarray(np.load(args.k1_method_dir / "refined_hand_pose.npy"), dtype=np.float32)
    baseline_alpha = np.asarray(np.load(args.k1_method_dir / "alpha.npy"), dtype=np.float32)
    if baseline_pose.shape != (len(order), 45) or baseline_alpha.shape != (len(order), 15):
        raise ValueError("K1 pose/alpha shapes must be [N,45] and [N,15]")

    global_orient, betas, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
    state_pose, low_rows = last_context_state(cache["base_hand_pose"], episodes)
    mano = mano_layer(args.device)
    state_cache, state_xyz = _prepare_state(
        "hot3d", cache, state_pose, betas, args.mano_cache, args.device, args.batch_size, mano
    )
    methods_root = args.out_dir / "methods"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_pose = baseline_pose.copy()
    raw_pose[low_rows] = state_pose[low_rows]
    _write_method(
        methods_root / "last_context",
        cache["sample_id"], "hot3d", state_xyz, raw_pose, betas, state_cache,
        args.mano_cache, args.device, args.batch_size, mano, 0.0,
        {"state": "last_context", "rope": False, "phase_gate": "known"},
    )

    methods = ["last_context"]
    shuffled = shuffle_low_rope(state_cache, episodes, args.shuffle_seed)
    for name, method_cache, rope_shuffled in (
        ("last_context_k1", state_cache, False),
        ("last_context_k1_rope_shuffled", shuffled, True),
    ):
        refined, alpha, elapsed = refine_low_student(
            method_cache, state_pose, baseline_pose, baseline_alpha, low_rows, betas,
            global_orient, args.k1_checkpoint, args.device, args.batch_size, mano,
        )
        _write_method(
            methods_root / name,
            cache["sample_id"], "hot3d", state_xyz, refined, betas, method_cache,
            args.mano_cache, args.device, args.batch_size, mano, elapsed,
            {"state": "last_context", "rope": True, "rope_shuffled": rope_shuffled, "phase_gate": "known"},
            alpha,
        )
        methods.append(name)

    payload = {
        "num_samples": len(order),
        "num_episodes": len(episodes),
        "num_low_visibility_rows": int(len(low_rows)),
        "methods": methods,
        "k1_checkpoint": str(args.k1_checkpoint),
        "k1_checkpoint_sha256": _sha256(args.k1_checkpoint),
        "protocol": {
            "state": "last context WiLoR 45D MANO hand pose",
            "phase_gate": "known from natural visibility selection",
            "global_orient_betas_cam_t": "current WiLoR values",
            "shuffle_scope": "within each episode low-visibility rows",
            "shuffle_seed": args.shuffle_seed,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--k1-checkpoint", type=Path, required=True)
    parser.add_argument("--k1-method-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shuffle-seed", type=int, default=20260718)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    output = run(parse_args(argv))
    print(f"HOT3D natural state predictions written to: {output}")
    return output


if __name__ == "__main__":
    main()
