#!/usr/bin/env python3
"""High-signal follow-ups for the clean-prefix state oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ropetrack.backends.hand_predictor import _rotmat_to_aa  # noqa: E402
from ropetrack.eval.pipeline import load_mano_j_regressor  # noqa: E402
from ropetrack.io import load_pred_json, read_json  # noqa: E402
from ropetrack.refine.actions import FINGER_POSE_GROUPS  # noqa: E402
from ropetrack.refine.cache import validate_cache  # noqa: E402
from ropetrack.refine.apply import (  # noqa: E402
    aa_to_rotmat,
    load_mano_globals,
    mano_layer,
    optimize_alpha,
)
from ropetrack.eval.slices import (  # noqa: E402
    finger_joint_map,
    occluded_fingers_for_row,
    per_joint_pa_distances,
)
from ropetrack.eval.temporal_metrics import _load_episode_manifest, _validate_order  # noqa: E402
from scripts.legacy.temporal.temporal_oracle_state import (  # noqa: E402
    _prepare_state,
    _refine_masked,
    _write_method,
    complete_episode_rows,
    shuffle_masked_rope,
)


def rotation_mean_pose(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
        raise ValueError(f"poses must be finite [K,45], got {values.shape}")
    mean = aa_to_rotmat(values.reshape(-1, 15, 3)).reshape(len(values), 15, 3, 3).mean(axis=0)
    u, _, vt = np.linalg.svd(mean)
    rotations = u @ vt
    negative = np.linalg.det(rotations) < 0
    if negative.any():
        u[negative, :, -1] *= -1
        rotations = u @ vt
    return _rotmat_to_aa(rotations).reshape(45).astype(np.float32)


def rotation_medoid_pose(poses: np.ndarray) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
        raise ValueError(f"poses must be finite [K,45], got {values.shape}")
    rotations = aa_to_rotmat(values.reshape(-1, 15, 3)).reshape(len(values), 15, 3, 3)
    relative = np.swapaxes(rotations[:, None], -1, -2) @ rotations[None, :]
    distances = np.linalg.norm(
        _rotmat_to_aa(relative.reshape(-1, 3, 3)).reshape(len(values), len(values), 15, 3),
        axis=-1,
    ).sum(axis=2)
    return values[int(np.argmin(distances.sum(axis=1)))].copy()


def build_followup_states(base_pose: np.ndarray, episodes: tuple[np.ndarray, ...]) -> dict[str, np.ndarray]:
    pose = np.asarray(base_pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 45 or not np.isfinite(pose).all():
        raise ValueError(f"base_pose must be finite [N,45], got {pose.shape}")
    names = (
        "state_age5",
        "state_age15",
        "state_age30",
        "prefix_mean5",
        "prefix_medoid5",
        "delayed_freeze3",
        "false_update_h1",
        "false_update_h15",
        "false_update_h30",
    )
    states = {name: pose.copy() for name in names}
    for rows in episodes:
        context, masked = rows[:30], rows[30:90]
        anchors = {
            "state_age5": pose[context[-5]],
            "state_age15": pose[context[-15]],
            "state_age30": pose[context[-30]],
            "prefix_mean5": rotation_mean_pose(pose[context[-5:]]),
            "prefix_medoid5": rotation_medoid_pose(pose[context[-5:]]),
        }
        for name, anchor in anchors.items():
            states[name][masked] = anchor
        states["delayed_freeze3"][masked[:3]] = pose[masked[:3]]
        states["delayed_freeze3"][masked[3:]] = pose[masked[2]]
        for horizon in (1, 15, 30):
            name = f"false_update_h{horizon}"
            split = horizon - 1
            states[name][masked[:split]] = pose[context[-1]]
            states[name][masked[split:]] = pose[masked[split]]
    return states


def mix_finger_poses(current: np.ndarray, state: np.ndarray, use_state: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    gate = np.asarray(use_state, dtype=bool)
    if current.shape != state.shape or current.ndim != 2 or current.shape[1] != 45:
        raise ValueError("current and state poses must match [N,45]")
    if gate.shape != (len(current), 5):
        raise ValueError(f"use_state must be {(len(current), 5)}, got {gate.shape}")
    out = current.copy()
    for finger, joints in enumerate(FINGER_POSE_GROUPS):
        dims = [3 * joint + axis for joint in joints for axis in range(3)]
        rows = gate[:, finger]
        out[np.ix_(rows, dims)] = state[np.ix_(rows, dims)]
    return out


def known_visibility_gate(manifest: list[dict]) -> np.ndarray:
    gate = np.zeros((len(manifest), 5), dtype=bool)
    for index, row in enumerate(manifest):
        if row["episode_phase"] != "masked":
            continue
        flags = occluded_fingers_for_row(row, (640, 480))
        if flags is None:
            raise ValueError(f"masked row has undecidable visibility: {row['sample_id']}")
        gate[index] = flags
    return gate


def ideal_gate(
    gt_xyz: np.ndarray,
    current_xyz: np.ndarray,
    state_xyz: np.ndarray,
    masked_rows: np.ndarray,
    per_finger: bool,
) -> np.ndarray:
    current_error = per_joint_pa_distances(gt_xyz, current_xyz)
    state_error = per_joint_pa_distances(gt_xyz, state_xyz)
    gate = np.zeros((len(gt_xyz), 5), dtype=bool)
    if per_finger:
        finger_joints, _ = finger_joint_map("ho3d")
        for finger, joints in enumerate(finger_joints):
            gate[masked_rows, finger] = (
                state_error[np.ix_(masked_rows, joints)].mean(axis=1)
                < current_error[np.ix_(masked_rows, joints)].mean(axis=1)
            )
    else:
        choose = state_error[masked_rows].mean(axis=1) < current_error[masked_rows].mean(axis=1)
        gate[masked_rows] = choose[:, None]
    return gate


def slice_cache(cache: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for key, value in cache.items():
        array = np.asarray(value)
        out[key] = array[rows].copy() if array.ndim and len(array) == len(cache["sample_id"]) else array.copy()
    validate_cache(out)
    return out


def load_pose(directory: Path, count: int) -> np.ndarray:
    pose = np.asarray(np.load(directory / "refined_hand_pose.npy", allow_pickle=False), dtype=np.float32)
    if pose.shape != (count, 45) or not np.isfinite(pose).all():
        raise ValueError(f"invalid pose in {directory}: {pose.shape}")
    return pose


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

    method_root = args.out_dir / "methods"
    method_root.mkdir(parents=True)
    baseline_pose = load_pose(args.k1_method_dir, len(order))
    baseline_alpha = np.asarray(np.load(args.k1_method_dir / "alpha.npy", allow_pickle=False), dtype=np.float32)
    if baseline_alpha.shape != (len(order), 15):
        raise ValueError("K1 alpha must be [N,15]")
    orient, betas, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
    mano = mano_layer(args.device)
    current_cache, current_base_xyz = _prepare_state(
        args.dataset, cache, cache["base_hand_pose"], betas, args.mano_cache, args.device, args.batch_size, mano
    )
    shuffled_template = shuffle_masked_rope(cache, episodes, args.shuffle_seed)
    created = []

    for name, state_pose in build_followup_states(cache["base_hand_pose"], episodes).items():
        state_cache, state_base_xyz = _prepare_state(
            args.dataset, cache, state_pose, betas, args.mano_cache, args.device, args.batch_size, mano
        )
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
                betas,
                orient,
                args.k1_checkpoint,
                args.device,
                args.batch_size,
                mano,
            )
            output_name = name + ("_rope_shuffled" if shuffled else "")
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
                {"state": name, "rope": True, "rope_shuffled": shuffled, "phase_gate": "known"},
                alpha,
            )
            created.append(output_name)

    existing = args.oracle_method_root
    state_sources = {
        "last_clean": existing / "last_clean",
        "last_clean_k1": existing / "last_clean_k1",
        "last_clean_k1_rope_shuffled": existing / "last_clean_k1_rope_shuffled",
    }
    source_pose = {name: load_pose(path, len(order)) for name, path in state_sources.items()}
    gt_xyz = np.asarray(read_json(args.gt_xyz), dtype=np.float64)
    if gt_xyz.shape != (len(order), 21, 3):
        raise ValueError(f"GT xyz must be {(len(order), 21, 3)}, got {gt_xyz.shape}")
    current_xyz = np.asarray(load_pred_json(args.k1_method_dir / "pred.json")[0], dtype=np.float64)
    source_xyz = {
        name: np.asarray(load_pred_json(path / "pred.json")[0], dtype=np.float64)
        for name, path in state_sources.items()
    }
    gates = {
        "known_finger_last_clean": (known_visibility_gate(manifest), "last_clean"),
        "known_finger_last_clean_k1": (known_visibility_gate(manifest), "last_clean_k1"),
        "known_finger_last_clean_k1_rope_shuffled": (
            known_visibility_gate(manifest),
            "last_clean_k1_rope_shuffled",
        ),
        "ideal_finger_last_clean": (
            ideal_gate(gt_xyz, current_xyz, source_xyz["last_clean"], masked_rows, True),
            "last_clean",
        ),
        "ideal_finger_last_clean_k1": (
            ideal_gate(gt_xyz, current_xyz, source_xyz["last_clean_k1"], masked_rows, True),
            "last_clean_k1",
        ),
        "ideal_finger_last_clean_k1_rope_shuffled": (
            ideal_gate(gt_xyz, current_xyz, source_xyz["last_clean_k1_rope_shuffled"], masked_rows, True),
            "last_clean_k1_rope_shuffled",
        ),
        "ideal_frame_last_clean_k1": (
            ideal_gate(gt_xyz, current_xyz, source_xyz["last_clean_k1"], masked_rows, False),
            "last_clean_k1",
        ),
        "ideal_frame_last_clean_k1_rope_shuffled": (
            ideal_gate(gt_xyz, current_xyz, source_xyz["last_clean_k1_rope_shuffled"], masked_rows, False),
            "last_clean_k1_rope_shuffled",
        ),
    }
    for name, (gate, source) in gates.items():
        mixed = mix_finger_poses(baseline_pose, source_pose[source], gate)
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
            {"state": source, "finger_gate": name, "oracle_gt": name.startswith("ideal_")},
        )
        created.append(name)

    last_state = source_pose["last_clean"]
    state_cache, state_base_xyz = _prepare_state(
        args.dataset, cache, last_state, betas, args.mano_cache, args.device, args.batch_size, mano
    )
    j_regressor = load_mano_j_regressor(REPO / "mano_data" / "MANO_RIGHT.pkl")
    tto_specs = (
        ("last_clean_tto", "rope", False),
        ("last_clean_tto_rope_shuffled", "rope", True),
        ("last_clean_oracle_tip", "oracle_tip", False),
        ("last_clean_oracle_chain", "oracle_chain", False),
    )
    for name, objective, shuffled in tto_specs:
        method_cache = {key: np.asarray(value).copy() for key, value in state_cache.items()}
        if shuffled:
            for key in ("input_rope_norm", "rope_valid"):
                method_cache[key] = np.asarray(shuffled_template[key]).copy()
        masked_cache = slice_cache(method_cache, masked_rows)
        started = time.perf_counter()
        refined_masked, alpha_masked, _ = optimize_alpha(
            masked_cache,
            args.mano_cache,
            args.device,
            steps=400,
            lr=32.0,
            alpha_l2=0.001,
            max_alpha=0.5,
            batch_size=args.batch_size,
            dataset=args.dataset,
            action_space="flex15",
            objective=objective,
            gt_xyz=gt_xyz[masked_rows] if objective != "rope" else None,
            j_regressor=j_regressor if objective != "rope" else None,
            gate_threshold=0.1 if objective == "rope" else None,
            mano_module=mano,
        )
        refined = baseline_pose.copy()
        refined[masked_rows] = refined_masked
        alpha = baseline_alpha.copy()
        alpha[masked_rows] = alpha_masked
        _write_method(
            method_root / name,
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
            time.perf_counter() - started,
            {"state": "last_clean", "objective": objective, "rope_shuffled": shuffled, "phase_gate": "known"},
            alpha,
        )
        created.append(name)

    payload = {
        "methods": created,
        "num_methods": len(created),
        "num_samples": len(order),
        "num_complete_episodes": len(episodes),
        "num_masked_rows": len(masked_rows),
        "protocol": {
            "phase_usage": "oracle gate and analysis only",
            "state": "45D MANO hand_pose; current global orient/cam_t",
            "teacher": {"steps": 400, "lr": 32.0, "alpha_l2": 0.001, "max_alpha": 0.5, "gate": 0.1},
            "shuffle_seed": args.shuffle_seed,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return args.out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ho3d",), default="ho3d")
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--gt-xyz", type=Path, required=True)
    parser.add_argument("--k1-method-dir", type=Path, required=True)
    parser.add_argument("--k1-checkpoint", type=Path, required=True)
    parser.add_argument("--oracle-method-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shuffle-seed", type=int, default=20260710)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    out = run(parse_args(argv))
    print(f"Temporal state follow-ups written to: {out}")
    return out


if __name__ == "__main__":
    main()
