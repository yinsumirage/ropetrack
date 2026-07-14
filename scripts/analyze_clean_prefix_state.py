#!/usr/bin/env python3
"""CPU-only upper bounds from a perfect clean GT prefix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ropetrack.io import load_pred_json, read_json, read_jsonl  # noqa: E402
from ropetrack.refine.analysis import json_sanitize  # noqa: E402
from ropetrack.refine.cache import align_rows_by_sample_id  # noqa: E402
from scripts.score_sliced_predictions import per_joint_pa_distances  # noqa: E402
from scripts.score_temporal_predictions import (  # noqa: E402
    BOOTSTRAP_METRICS,
    _lag_metrics,
    _load_episode_manifest,
    _masked_motion_metrics,
    _masked_occluded_tip_mask,
    _phase_metrics,
    _sequence_metrics,
    _sequence_rows,
    _validate_order,
    sequence_bootstrap_ci,
    temporal_motion_metrics,
)
from scripts.temporal_oracle_state import complete_episode_rows  # noqa: E402


def perfect_prefix_predictions(
    gt_xyz: np.ndarray,
    episodes: tuple[np.ndarray, ...],
    baseline: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    gt = np.asarray(gt_xyz, dtype=np.float64)
    if gt.ndim != 3 or gt.shape[1:] != (21, 3) or not np.isfinite(gt).all():
        raise ValueError(f"gt_xyz must be finite [N,21,3], got {gt.shape}")
    outside = gt if baseline is None else np.asarray(baseline, dtype=np.float64)
    if outside.shape != gt.shape or not np.isfinite(outside).all():
        raise ValueError("baseline must match finite gt_xyz")
    predictions = {
        name: outside.copy()
        for name in ("perfect_prefix_last_clean", "perfect_prefix_constant_velocity", "perfect_prefix_damped_velocity")
    }
    root_relative = gt - gt[:, :1]
    for rows in episodes:
        context = rows[:30]
        last_five = root_relative[context[-5:]]
        velocity = np.median(np.diff(last_five, axis=0), axis=0)
        last_clean = last_five[-1]
        for horizon, row in enumerate(rows[30:90], start=1):
            scales = {
                "perfect_prefix_last_clean": 0.0,
                "perfect_prefix_constant_velocity": float(min(horizon, 5)),
                "perfect_prefix_damped_velocity": float((1.0 - 0.8**horizon) / 0.2),
            }
            for name, scale in scales.items():
                predictions[name][row] = gt[row, :1] + last_clean + scale * velocity
    return predictions


def ideal_gate(
    gt_xyz: np.ndarray,
    candidates: dict[str, np.ndarray],
    masked_rows: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    names = list(candidates)
    errors = np.stack(
        [per_joint_pa_distances(gt_xyz, candidates[name]).mean(axis=1) for name in names], axis=1
    )
    choice = np.argmin(errors[masked_rows], axis=1)
    output = np.asarray(baseline, dtype=np.float64).copy()
    for candidate_index, name in enumerate(names):
        selected = masked_rows[choice == candidate_index]
        output[selected] = candidates[name][selected]
    counts = {name: int(np.sum(choice == index)) for index, name in enumerate(names)}
    return output, counts


def beta_diagnostics(
    mano_cache: Path,
    order: list[str],
    manifest: list[dict],
    episodes: tuple[np.ndarray, ...],
) -> dict:
    with np.load(mano_cache, allow_pickle=False) as loaded:
        perm = align_rows_by_sample_id(order, loaded["sample_id"])
        betas = np.asarray(loaded["base_betas"], dtype=np.float64)[perm]
    distances = np.full(len(order), np.nan, dtype=np.float64)
    for rows in episodes:
        prefix = np.median(betas[rows[:30]], axis=0)
        distances[rows] = np.linalg.norm(betas[rows] - prefix, axis=1)
    phases = np.asarray([row["episode_phase"] for row in manifest])
    offsets = np.asarray([row["episode_offset"] for row in manifest])
    report = {}
    for phase in ("context", "masked", "recovery"):
        values = distances[(phases == phase) & np.isfinite(distances)]
        report[f"{phase}_beta_l2_to_prefix_median"] = float(values.mean()) if values.size else None
    for horizon in (1, 5, 15, 30, 60):
        values = distances[(phases == "masked") & (offsets == 29 + horizon)]
        report[f"masked_h{horizon}_beta_l2_to_prefix_median"] = float(values.mean())
    return report


def method_metrics(
    order: list[str],
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    manifest: list[dict],
    occluded_tip_mask: np.ndarray,
    fps: float,
) -> tuple[dict, dict]:
    joint_pa = per_joint_pa_distances(gt_xyz, pred_xyz) * 1000.0
    frame_pa = joint_pa.mean(axis=1)
    metrics = {
        "num_samples": len(order),
        "pa_mpjpe_mm": float(frame_pa.mean()),
        **temporal_motion_metrics(order, gt_xyz, pred_xyz, fps, raw_frame_step=1),
        **_lag_metrics(order, gt_xyz, pred_xyz, fps, raw_frame_step=1),
        **_phase_metrics(frame_pa, manifest, order, raw_frame_step=1),
        **_masked_motion_metrics(order, gt_xyz, pred_xyz, manifest, fps, raw_frame_step=1),
        "masked_occluded_tip_pa_mpjpe_mm": float(joint_pa[occluded_tip_mask].mean()),
    }
    sequence = _sequence_metrics(
        order,
        gt_xyz,
        pred_xyz,
        frame_pa,
        frame_pa,
        joint_pa,
        occluded_tip_mask,
        fps,
        1,
        manifest,
    )
    sequence.pop("pa_mpvpe_mm", None)
    return metrics, sequence


def run(args: argparse.Namespace) -> dict:
    manifest_rows = list(read_jsonl(args.hard_manifest))
    manifest_order = [str(row["sample_id"]) for row in manifest_rows]
    order = _validate_order(args.run_meta) if args.run_meta is not None else manifest_order
    manifest = _load_episode_manifest(args.hard_manifest, order, raw_frame_step=1)
    episodes = complete_episode_rows(manifest)
    gt_xyz = np.asarray(read_json(args.gt_xyz), dtype=np.float64)
    if gt_xyz.shape != (len(order), 21, 3) or not np.isfinite(gt_xyz).all():
        raise ValueError(f"GT xyz must be finite [{len(order)},21,3], got {gt_xyz.shape}")

    baseline = None
    if args.k1_method_dir is not None:
        xyz_rows, vertex_rows = load_pred_json(args.k1_method_dir / "pred.json")
        baseline = np.asarray(xyz_rows, dtype=np.float64)
        del xyz_rows, vertex_rows
        if baseline.shape != gt_xyz.shape or not np.isfinite(baseline).all():
            raise ValueError("K1 xyz must match finite GT xyz")
    predictions = perfect_prefix_predictions(gt_xyz, episodes, baseline)
    masked_rows = np.sort(np.concatenate([rows[30:90] for rows in episodes]))
    gate_counts = {}
    if baseline is not None:
        predictions = {"k1": baseline, **predictions}
        predictions["ideal_k1_last_clean_gate"], gate_counts["ideal_k1_last_clean_gate"] = ideal_gate(
            gt_xyz,
            {"k1": baseline, "perfect_prefix_last_clean": predictions["perfect_prefix_last_clean"]},
            masked_rows,
            baseline,
        )
        predictions["ideal_k1_motion_gate"], gate_counts["ideal_k1_motion_gate"] = ideal_gate(
            gt_xyz,
            {
                name: predictions[name]
                for name in (
                    "k1",
                    "perfect_prefix_last_clean",
                    "perfect_prefix_constant_velocity",
                    "perfect_prefix_damped_velocity",
                )
            },
            masked_rows,
            baseline,
        )

    occluded_tip_mask = _masked_occluded_tip_mask(manifest)
    methods = {}
    sequence_values = {}
    for name, prediction in predictions.items():
        methods[name], sequence_values[name] = method_metrics(
            order, gt_xyz, prediction, manifest, occluded_tip_mask, args.fps
        )
    reference = "k1" if "k1" in methods else "perfect_prefix_last_clean"
    paired = {
        name: {
            metric: values[metric] - methods[reference][metric]
            for metric in BOOTSTRAP_METRICS
            if values.get(metric) is not None and methods[reference].get(metric) is not None
        }
        for name, values in methods.items()
    }
    report = {
        "methods": methods,
        "reference": reference,
        "paired_deltas": paired,
        "bootstrap_ci": sequence_bootstrap_ci(list(_sequence_rows(order)), sequence_values, reference),
        "gate_choice_counts": gate_counts,
        "beta_diagnostics": beta_diagnostics(args.mano_cache, order, manifest, episodes),
        "protocol": {
            "oracle_only": True,
            "perfect_clean_prefix_gt": True,
            "num_complete_episodes": len(episodes),
            "num_masked_rows": len(masked_rows),
            "velocity_frames": 5,
            "constant_velocity_cap_frames": 5,
            "damping": 0.8,
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260710,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_sanitize(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-xyz", type=Path, required=True)
    parser.add_argument("--hard-manifest", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, default=None)
    parser.add_argument("--k1-method-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run(args)
    print(f"Clean-prefix state analysis written to: {args.output}")
    return args.output


if __name__ == "__main__":
    main()
