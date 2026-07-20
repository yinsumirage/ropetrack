#!/usr/bin/env python3
"""Decompose existing hand-joint errors into nested T/RT/Sim3 oracles."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ropetrack.io import load_pred_json, read_json, read_jsonl  # noqa: E402
from ropetrack.refine.cache import align_rows_by_sample_id  # noqa: E402
from scripts.score_dexycb import palm_orientation_error_deg, procrustes  # noqa: E402
from scripts.score_predictions import align_w_scale  # noqa: E402


TIP_IDS = np.asarray([4, 8, 12, 16, 20])
NON_TIP_IDS = np.asarray([index for index in range(21) if index not in TIP_IDS])
PALM_IDS = np.asarray([0, 5, 9, 17])
JOINT_NAMES = [
    "wrist", "thumb1", "thumb2", "thumb3", "thumb4", "index1", "index2",
    "index3", "index4", "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4", "little1", "little2", "little3",
    "little4",
]
CHAIN = ("E_camera", "E_T", "E_RT", "E_Sim3")
SUMMARY_METRICS = (
    "E_camera", "E_root", "E_T", "E_RT", "E_Sim3", "H_T", "H_R", "H_S",
    "H_local", "root_translation", "palm_orientation_deg", "tip_camera",
    "tip_root", "tip_pa", "non_tip_camera", "non_tip_root", "non_tip_pa",
)
RAW_METRICS = ("raw_E_T", "raw_E_RT", "raw_E_Sim3")
ADJUSTMENT_METRICS = ("T_envelope_adjustment", "RT_envelope_adjustment", "Sim3_envelope_adjustment")


def file_record(path: str | Path) -> dict:
    value = Path(path)
    info = value.stat()
    return {
        "path": str(value),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "mtime_utc": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
    }


def load_order(path: str | Path) -> list[str]:
    value = Path(path)
    if value.suffix == ".npy":
        return np.load(value).astype(str).tolist()
    payload = read_json(value)
    return list(map(str, payload.get("sample_order", payload)))


def align_prediction_rows(target_ids, source_ids, values):
    if len(source_ids) != len(set(map(str, source_ids))):
        raise ValueError("prediction sample IDs contain duplicates")
    return np.asarray(values)[align_rows_by_sample_id(target_ids, source_ids)]


def _proper_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(target.T @ source, full_matrices=False)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = u @ vt
    return rotation


def _mean_valid(values: np.ndarray, valid: np.ndarray) -> float:
    chosen = values[valid]
    return float(chosen.mean()) if len(chosen) else float("nan")


def decompose_sample(predicted, target, valid=None) -> dict:
    """Return errors in input units; raise on fewer than 3 non-degenerate joints."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 3:
        raise ValueError(f"joint arrays must share [J,3], got {predicted.shape} and {target.shape}")
    common = np.isfinite(predicted).all(axis=1) & np.isfinite(target).all(axis=1)
    if valid is not None:
        common &= np.asarray(valid, dtype=bool)
    if int(common.sum()) < 3:
        raise ValueError("fewer than three common valid joints")
    pred, gt = predicted[common], target[common]
    pred_centered, gt_centered = pred - pred.mean(axis=0), gt - gt.mean(axis=0)
    if np.linalg.matrix_rank(pred_centered, tol=1e-10) < 2 or np.linalg.matrix_rank(gt_centered, tol=1e-10) < 2:
        raise ValueError("fewer than three non-degenerate joints")

    translated = pred_centered + gt.mean(axis=0)
    rotation = _proper_rotation(pred_centered, gt_centered)
    rigid = pred_centered @ rotation.T + gt.mean(axis=0)
    similarity = procrustes(pred, gt)
    per_camera = np.linalg.norm(predicted - target, axis=1)
    per_t = np.linalg.norm(translated - gt, axis=1)
    per_rt = np.linalg.norm(rigid - gt, axis=1)
    per_pa_valid = np.linalg.norm(similarity - gt, axis=1)
    per_pa = np.full(len(predicted), np.nan, dtype=np.float64)
    per_pa[common] = per_pa_valid

    raw_errors = {
        "E_camera": float(per_camera[common].mean()),
        "raw_E_T": float(per_t.mean()),
        "raw_E_RT": float(per_rt.mean()),
        "raw_E_Sim3": float(per_pa_valid.mean()),
    }
    selected_t = min(raw_errors["E_camera"], raw_errors["raw_E_T"])
    selected_rt = min(selected_t, raw_errors["raw_E_RT"])
    selected_sim3 = min(selected_rt, raw_errors["raw_E_Sim3"])
    adjustments = {
        "T_envelope_adjustment": raw_errors["raw_E_T"] - selected_t,
        "RT_envelope_adjustment": raw_errors["raw_E_RT"] - selected_rt,
        "Sim3_envelope_adjustment": raw_errors["raw_E_Sim3"] - selected_sim3,
    }
    errors = {
        "E_camera": raw_errors["E_camera"], "E_T": selected_t,
        "E_RT": selected_rt, "E_Sim3": selected_sim3,
        **{name: raw_errors[name] for name in RAW_METRICS}, **adjustments,
    }
    tolerance = 1e-7
    for left, right in zip(CHAIN, CHAIN[1:]):
        if errors[left] + tolerance < errors[right]:
            raise ArithmeticError(f"nested oracle violation: {left}={errors[left]} < {right}={errors[right]}")

    root_valid = bool(common[0])
    per_root = np.full(len(predicted), np.nan, dtype=np.float64)
    if root_valid:
        per_root = np.linalg.norm(
            (predicted - predicted[:1]) - (target - target[:1]), axis=1
        )
    tip_valid, non_tip_valid = common.copy(), common.copy()
    tip_valid[NON_TIP_IDS[NON_TIP_IDS < len(common)]] = False
    non_tip_valid[TIP_IDS[TIP_IDS < len(common)]] = False
    palm_valid = len(common) >= 18 and common[PALM_IDS].all()
    errors.update({
        "E_root": _mean_valid(per_root, common) if root_valid else float("nan"),
        "H_T": errors["E_camera"] - errors["E_T"],
        "H_R": errors["E_T"] - errors["E_RT"],
        "H_S": errors["E_RT"] - errors["E_Sim3"],
        "H_local": errors["E_Sim3"],
        "root_translation": float(per_camera[0]) if root_valid else float("nan"),
        "palm_orientation_deg": float(palm_orientation_error_deg(predicted[None], target[None])[0]) if palm_valid else float("nan"),
        "tip_camera": _mean_valid(per_camera, tip_valid),
        "tip_root": _mean_valid(per_root, tip_valid) if root_valid else float("nan"),
        "tip_pa": _mean_valid(per_pa, tip_valid),
        "non_tip_camera": _mean_valid(per_camera, non_tip_valid),
        "non_tip_root": _mean_valid(per_root, non_tip_valid) if root_valid else float("nan"),
        "non_tip_pa": _mean_valid(per_pa, non_tip_valid),
        "valid_joint_count": int(common.sum()),
        "per_joint_camera": np.where(common, per_camera, np.nan),
        "per_joint_root": np.where(common, per_root, np.nan),
        "per_joint_pa": per_pa,
    })
    return errors


def compute_method(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[dict[str, np.ndarray], Counter]:
    count, joints = target.shape[:2]
    values = {name: np.full(count, np.nan, dtype=np.float64) for name in SUMMARY_METRICS}
    values["valid_joint_count"] = np.zeros(count, dtype=np.int16)
    for name in RAW_METRICS + ADJUSTMENT_METRICS:
        values[name] = np.full(count, np.nan, dtype=np.float64)
    for name in ("per_joint_camera", "per_joint_root", "per_joint_pa"):
        values[name] = np.full((count, joints), np.nan, dtype=np.float64)
    skipped = Counter()
    for index in range(count):
        try:
            result = decompose_sample(predicted[index], target[index], valid[index])
        except ValueError as error:
            skipped[str(error)] += 1
            continue
        except ArithmeticError as error:
            raise ArithmeticError(f"sample {index}: {error}") from error
        for name, value in result.items():
            values[name][index] = value
    return values, skipped


def compute_parity_method(predicted: np.ndarray, target: np.ndarray, valid: np.ndarray, legacy_alignment=False) -> dict[str, np.ndarray]:
    values = {name: np.full(len(target), np.nan, dtype=np.float64) for name in ("E_camera", "E_root", "raw_E_Sim3")}
    for index in range(len(target)):
        common = valid[index] & np.isfinite(predicted[index]).all(axis=1)
        if int(common.sum()) < 3:
            continue
        pred, gt = predicted[index, common], target[index, common]
        aligned = align_w_scale(gt, pred) if legacy_alignment else procrustes(pred, gt)
        values["E_camera"][index] = np.linalg.norm(pred - gt, axis=1).mean()
        if common[0]:
            rooted = (predicted[index] - predicted[index, :1]) - (target[index] - target[index, :1])
            values["E_root"][index] = np.linalg.norm(rooted[common], axis=1).mean()
        values["raw_E_Sim3"][index] = np.linalg.norm(aligned - gt, axis=1).mean()
    return values


def metric_means(values: dict[str, np.ndarray], scale=1000.0) -> dict:
    result = {}
    for name in SUMMARY_METRICS:
        finite = np.asarray(values[name])[np.isfinite(values[name])]
        factor = 1.0 if name == "palm_orientation_deg" else scale
        result[name] = float(finite.mean() * factor) if len(finite) else None
        result[f"{name}_count"] = int(len(finite))
    return result


def group_bootstrap(values, groups, iterations, seed) -> list[float] | None:
    values, groups = np.asarray(values), np.asarray(groups)
    finite = np.isfinite(values)
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse[finite], weights=values[finite], minlength=inverse.max() + 1)
    counts = np.bincount(inverse[finite], minlength=inverse.max() + 1)
    keep = counts > 0
    sums, counts = sums[keep], counts[keep]
    if not len(sums):
        return None
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(sums), size=(iterations, len(sums)))
    means = sums[chosen].sum(axis=1) / counts[chosen].sum(axis=1)
    return (np.percentile(means, [2.5, 97.5]) * 1000.0).tolist()


def group_bootstrap_delta(candidate, reference, groups, iterations, seed) -> list[float] | None:
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    delta = candidate - reference
    delta[~(np.isfinite(candidate) & np.isfinite(reference))] = np.nan
    return group_bootstrap(delta, groups, iterations, seed)


def read_gt(spec: dict, parity=False):
    prefix = "parity_" if parity else ""
    gt = np.asarray(read_json(spec[f"{prefix}gt"]), dtype=np.float64)
    manifest_path = spec.get(f"{prefix}manifest")
    rows = list(read_jsonl(manifest_path)) if manifest_path else None
    ids = ([str(row["sample_id"]) for row in rows] if rows is not None
           else load_order(spec[f"{prefix}target_order"]))
    if gt.shape != (len(ids), 21, 3):
        raise ValueError(f"{spec['name']} GT/order mismatch: {gt.shape} vs {len(ids)}")
    valid = np.isfinite(gt).all(axis=2)
    if rows is not None and all("joint_valid" in row for row in rows):
        valid &= np.asarray([row["joint_valid"] for row in rows], dtype=bool)
    return rows, ids, gt, valid


def gt_mesh_inventory(spec: dict, rows) -> tuple[dict | None, int]:
    path = spec.get("gt_mesh")
    if not path or not Path(path).is_file():
        return None, 0
    value = Path(path)
    if value.suffix == ".json":
        payload = read_json(value)
        count = sum(row is not None for row in payload)
    elif rows is not None and all("mano_valid" in row for row in rows):
        count = sum(bool(row["mano_valid"]) for row in rows)
    else:
        with np.load(value) as payload:
            count = int(np.asarray(payload["mano_valid"], dtype=bool).sum()) if "mano_valid" in payload else len(payload["sample_id"])
    return file_record(value), count


def group_ids(spec: dict, rows, ids) -> np.ndarray:
    if spec.get("group_field"):
        return np.asarray([str(row[spec["group_field"]]) for row in rows])
    if spec.get("group_rule") == "sample_prefix":
        return np.asarray([sample_id.split("/")[0] for sample_id in ids])
    raise ValueError(f"group rule missing for {spec['name']}")


def metadata_columns(spec: dict, rows, count: int) -> dict[str, np.ndarray]:
    empty = np.full(count, "", dtype=str)
    result = {name: empty.copy() for name in ("subject", "camera", "visibility", "side", "hand_type", "mano_population", "capture")}
    if rows is None:
        return result
    name = spec["name"]
    if name == "hot3d":
        result["visibility"] = np.asarray([str(row["phase"]) for row in rows])
    elif name == "dexycb_s1_val":
        low, high = spec["visibility_thresholds"]
        result["subject"] = np.asarray([str(row["subject_id"]) for row in rows])
        result["camera"] = np.asarray([str(row["camera_serial"]) for row in rows])
        result["visibility"] = np.asarray([
            "low_visible" if row["hand_segmentation_pixels"] <= low else
            "high_visible" if row["hand_segmentation_pixels"] >= high else "mid_visible"
            for row in rows
        ])
    elif name == "interhand26m_val":
        result["side"] = np.asarray([str(row["mano_side"]) for row in rows])
        result["hand_type"] = np.asarray(["interacting" if row["is_interacting"] else "single" for row in rows])
        result["mano_population"] = np.asarray(["mano_valid" if row["mano_valid"] else "joint_only" for row in rows])
        result["capture"] = np.asarray([str(row["capture_id"]) for row in rows])
        result["camera"] = np.asarray([str(row["camera_id"]) for row in rows])
    return result


def subgroup_summary(values: dict, metadata: dict[str, np.ndarray]) -> dict:
    report = {}
    for column, labels in metadata.items():
        if not np.any(labels != ""):
            continue
        report[column] = {}
        for label in sorted(set(labels.tolist())):
            chosen = labels == label
            report[column][label] = metric_means({key: value[chosen] for key, value in values.items() if key in SUMMARY_METRICS})
    return report


def summarize_values(values: dict, groups, iterations, seed) -> dict:
    result = metric_means(values)
    result["count"] = result["E_camera_count"]
    result["conventional_alignment_candidates_mm"] = {
        name: float(np.nanmean(values[name]) * 1000.0) for name in RAW_METRICS
    }
    result["monotone_envelope"] = {
        name: {
            "adjusted_sample_count": int(np.count_nonzero(np.asarray(values[name]) > 0)),
            "max_adjustment_mm": float(np.nanmax(values[name]) * 1000.0),
            "mean_adjustment_mm": float(np.nanmean(values[name]) * 1000.0),
        }
        for name in ADJUSTMENT_METRICS
    }
    result["bootstrap_95ci_mm"] = {
        name: group_bootstrap(values[name], groups, iterations, seed)
        for name in ("E_camera", "E_root", "E_T", "E_RT", "E_Sim3", "H_T", "H_R", "H_S", "H_local")
    }
    camera = np.asarray(values["E_camera"])
    result["mean_fraction_of_E_camera"] = {}
    result["ratio_of_means"] = {}
    for name in ("H_T", "H_R", "H_S", "H_local"):
        component = np.asarray(values[name])
        finite = np.isfinite(component) & np.isfinite(camera) & (camera > 0)
        result["mean_fraction_of_E_camera"][name] = float(np.mean(component[finite] / camera[finite])) if finite.any() else None
        result["ratio_of_means"][name] = result[name] / result["E_camera"] if result[name] is not None and result["E_camera"] else None
    result["per_joint_camera_mm"] = (np.nanmean(values["per_joint_camera"], axis=0) * 1000.0).tolist()
    result["per_joint_root_mm"] = (np.nanmean(values["per_joint_root"], axis=0) * 1000.0).tolist()
    result["per_joint_pa_mm"] = (np.nanmean(values["per_joint_pa"], axis=0) * 1000.0).tolist()
    return result


def mean_methods(methods: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.nanmean(np.stack([method[name] for method in methods]), axis=0) for name in methods[0]}


def comparison(candidate, reference, groups, iterations, seed) -> dict:
    names = ("E_camera", "E_root", "E_T", "E_RT", "E_Sim3", "H_T", "H_R", "H_S", "H_local")
    return {
        "delta_sign": "candidate minus reference; negative is improvement for E metrics",
        "means_mm": {
            name: float(np.nanmean(candidate[name] - reference[name]) * 1000.0)
            for name in names
        },
        "bootstrap_95ci_mm": {
            name: group_bootstrap_delta(candidate[name], reference[name], groups, iterations, seed)
            for name in names
        },
    }


def parity_rows(spec, observed, tolerance) -> list[dict]:
    rows = []
    expected = spec.get("parity_expected", {})
    mapping = {"camera_mm": "E_camera", "root_mm": "E_root", "pa_mm": "raw_E_Sim3"}
    for expected_name, metric in mapping.items():
        target = expected.get(expected_name)
        if target is None:
            continue
        values = np.asarray(observed[metric])
        value = float(np.nanmean(values) * 1000.0)
        difference = abs(value - target)
        rows.append({
            "metric": expected_name,
            "expected": target,
            "observed": value,
            "absolute_difference_mm": difference,
            "tolerance_mm": tolerance,
            "status": "PASS" if difference <= tolerance else "FAIL",
        })
    return rows


def write_per_sample(path: Path, chunks: dict[str, list[np.ndarray]]) -> None:
    arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    np.savez_compressed(path, **arrays)


def report_rows(summary: dict) -> list[tuple[str, str, dict]]:
    rows = []
    for dataset, entry in summary["datasets"].items():
        for method in entry["display_methods"]:
            rows.append((dataset, method, entry["methods"][method]))
    return rows


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# DirectPose existing-prediction error decomposition", "",
        "> T/RT/Sim3 values are nested per-sample oracle metric reductions, not causal error components or learnable ceilings.", "",
        "## A. Absolute decomposition", "",
        "| Dataset | Method | N | Camera | Root | T-aligned | RT-aligned | Sim3/PA |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, method, values in report_rows(summary):
        ranges = values.get("member_range_mm", {})
        cell = lambda key: (f"{fmt(values[key])} [{fmt(ranges[key][0])},{fmt(ranges[key][1])}]" if key in ranges else fmt(values[key]))
        lines.append(f"| {dataset} | {method} | {values['count']} | {cell('E_camera')} | {cell('E_root')} | {cell('E_T')} | {cell('E_RT')} | {cell('E_Sim3')} |")
    lines += ["", "## B. Oracle headroom", "", "| Dataset | Method | Translation gain | Rotation gain | Scale gain | Local residual |", "|---|---|---:|---:|---:|---:|"]
    for dataset, method, values in report_rows(summary):
        fraction = values["mean_fraction_of_E_camera"]
        cell = lambda key: f"{values[key]:.3f} ({100*fraction[key]:.1f}%)"
        lines.append(f"| {dataset} | {method} | {cell('H_T')} | {cell('H_R')} | {cell('H_S')} | {cell('H_local')} |")
    lines += ["", "## C. DirectPose signed delta versus matched base", "", "Negative ΔE means improvement; headroom deltas describe changed oracle room and are not themselves accuracy improvements.", "", "| Dataset | Method | ΔCamera | ΔRoot | ΔRT | ΔPA | ΔTranslation headroom | ΔRotation headroom |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, entry in summary["datasets"].items():
        for method, values in entry["versus_base"].items():
            delta = values["means_mm"]
            lines.append(f"| {dataset} | {method} | {fmt(delta['E_camera'])} | {fmt(delta['E_root'])} | {fmt(delta['E_RT'])} | {fmt(delta['E_Sim3'])} | {fmt(delta['H_T'])} | {fmt(delta['H_R'])} |")
    lines += ["", "## D. Dataset-level decision", "", "| Dataset | Main remaining bottleneck | DirectPose effect | Global branch evidence | Caveat |", "|---|---|---|---|---|"]
    for dataset, value in summary["dataset_decisions"].items():
        lines.append(f"| {dataset} | {value['main_remaining_bottleneck']} | {value['direct_pose_effect']} | {value['global_branch_evidence']} | {value['caveat']} |")
    judgment = summary["final_judgment"]
    lines += ["", "## Final judgment", "", f"**{judgment['decision']}**", "", judgment["reason"], "", f"LoRA: **{judgment['lora']}**", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(summary: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, rows = [], []
    for dataset, entry in summary["datasets"].items():
        for method in ("wilor_base", entry["representative"]):
            labels.append(f"{dataset}\n{method}")
            rows.append([entry["methods"][method][key] for key in ("H_T", "H_R", "H_S", "H_local")])
    values = np.asarray(rows)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    fig, axis = plt.subplots(figsize=(12, 5.4))
    bottom = np.zeros(len(values))
    for index, (name, color) in enumerate(zip(("Translation", "Rotation", "Scale", "Local/Sim3"), colors)):
        axis.bar(np.arange(len(values)), values[:, index], bottom=bottom, label=name, color=color)
        bottom += values[:, index]
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    axis.set_ylabel("Mean oracle metric reduction / residual (mm)")
    axis.legend(ncol=4, frameon=False, loc="upper center")
    axis.set_title("Nested oracle headroom (not causal proportions or learnable ceilings)")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_judgment(config: dict, summary: dict, results: dict, groups_by_dataset: dict) -> tuple[dict, dict]:
    decisions, qualifying = {}, []
    translation_support = rotation_support = 0
    for spec in config["datasets"]:
        dataset, representative = spec["name"], spec["representative"]
        base, candidate = results[(dataset, "wilor_base")], results[(dataset, representative)]
        groups = groups_by_dataset[dataset]
        delta = comparison(candidate, base, groups, config["bootstrap_iterations"], config["bootstrap_seed"])
        local_delta, camera_delta = delta["means_mm"]["E_Sim3"], delta["means_mm"]["E_camera"]
        local_ci = delta["bootstrap_95ci_mm"]["E_Sim3"]
        clear_local = local_delta < 0 and local_ci is not None and local_ci[1] < 0
        camera_not_corresponding = camera_delta >= 0 or abs(camera_delta) <= 0.5 * abs(local_delta)
        means = summary["datasets"][dataset]["methods"][representative]
        global_headroom = means["H_T"] + means["H_R"]
        local_scale = means["H_S"] + means["H_local"]
        cross_layer = global_headroom > local_scale
        qualifies = clear_local and camera_not_corresponding and cross_layer
        if qualifies:
            qualifying.append(dataset)
        if means["H_T"] > max(means["H_S"], means["H_local"]):
            translation_support += 1
        if means["H_R"] > max(means["H_S"], means["H_local"]):
            rotation_support += 1
        components = {name: means[name] for name in ("H_T", "H_R", "H_S", "H_local")}
        bottleneck = max(components, key=components.get)
        decisions[dataset] = {
            "main_remaining_bottleneck": bottleneck,
            "direct_pose_effect": f"ΔPA {local_delta:+.3f} mm; Δcamera {camera_delta:+.3f} mm",
            "global_branch_evidence": f"global headroom {global_headroom:.3f} vs scale+local {local_scale:.3f} mm; rule={'PASS' if qualifies else 'FAIL'}",
            "caveat": spec["caveat"],
        }
    if len(qualifying) >= 3:
        if translation_support >= 3 and rotation_support >= 3:
            decision = "CONTINUE ΔR+Δt structured branch"
        elif rotation_support >= 3:
            decision = "CONTINUE ΔR only"
        else:
            decision = "CONTINUE Δt only"
    else:
        decision = "STOP universal global branch; keep local-only DirectPose"
    return decisions, {
        "decision": decision,
        "qualifying_datasets": qualifying,
        "translation_support_dataset_count": translation_support,
        "rotation_support_dataset_count": rotation_support,
        "reason": f"{len(qualifying)} datasets pass the predeclared local-gain/camera-mismatch/cross-layer headroom rule; oracle consistency does not establish learnability.",
        "lora": "NO; existing-prediction geometry adds no new evidence for backbone adaptation",
        "learnability_boundary": "This analysis proves oracle headroom and cross-dataset consistency only, not that ΔR or Δt can be learned from the available inputs.",
    }


def analyze(config_path: Path, output_root: Path) -> None:
    config = read_json(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    iterations, seed = config["bootstrap_iterations"], config["bootstrap_seed"]
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    inventory = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "script_sha256": script_hash,
        "units": {"input": "metres", "output": "millimetres"},
        "datasets": {},
        "forbidden_test_paths_read": False,
    }
    summary = {
        "metric_definition": "per-sample mean Euclidean joint distance; T/RT/Sim3 use a monotone envelope over the conventional centroid/Kabsch/proper-Procrustes candidates",
        "conventional_alignment_boundary": "centroid/Kabsch/Procrustes minimize squared error while MPJPE is mean Euclidean; raw candidates and every envelope adjustment are retained for audit, and any violation of the resulting nested envelope fails loudly",
        "oracle_boundary": "metric reductions, not causal additive components or learnable ceilings",
        "bootstrap": {"iterations": iterations, "seed": seed},
        "parity": [], "datasets": {},
    }
    results, groups_by_dataset = {}, {}
    chunks = defaultdict(list)
    input_paths = {str(config_path)}

    for spec in config["datasets"]:
        if spec["name"] in {"dexycb_s1_val", "interhand26m_val"}:
            forbidden = [value for key, value in spec.items() if isinstance(value, str) and "/test/" in value]
            if forbidden:
                raise PermissionError(f"forbidden test path in config: {forbidden}")
        rows, target_ids, gt, valid = read_gt(spec)
        parity_rows_data, parity_ids, parity_gt, parity_valid = (
            read_gt(spec, parity=True) if spec.get("parity_gt") else (rows, target_ids, gt, valid)
        )
        groups = group_ids(spec, rows, target_ids)
        metadata = metadata_columns(spec, rows, len(target_ids))
        gt_mesh, gt_mesh_count = gt_mesh_inventory(spec, rows)
        groups_by_dataset[spec["name"]] = groups
        dataset_inventory = {
            "gt": file_record(spec["gt"]),
            "manifest": file_record(spec["manifest"]) if spec.get("manifest") else None,
            "parity_gt": file_record(spec["parity_gt"]) if spec.get("parity_gt") else None,
            "parity_manifest": file_record(spec["parity_manifest"]) if spec.get("parity_manifest") else None,
            "parity_source": file_record(spec["parity_source"]),
            "gt_sample_count": len(target_ids), "gt_joint_count": 21,
            "valid_joint_count_min_max": [int(valid.sum(axis=1).min()), int(valid.sum(axis=1).max())],
            "gt_mesh": gt_mesh, "gt_mesh_valid_count": gt_mesh_count,
            "bootstrap_unit": spec["bootstrap_unit"], "methods": {}, "sample_id_pairwise": {},
        }
        input_paths.update(path for path in (spec["gt"], spec.get("manifest"), spec.get("parity_gt"), spec.get("parity_manifest"), spec.get("gt_mesh"), spec["parity_source"]) if path)
        source_sets = {}
        for method in spec["methods"]:
            pred_path, order_path = method["prediction"], method["order"]
            prediction_record, order_record = file_record(pred_path), file_record(order_path)
            source_ids = load_order(order_path)
            xyz_rows, vertex_rows = load_pred_json(pred_path)
            if len(source_ids) != len(xyz_rows):
                raise ValueError(f"{spec['name']}/{method['name']} prediction/order length mismatch")
            target_set, source_set = set(target_ids), set(source_ids)
            audit = {
                "intersection_with_gt": len(target_set & source_set),
                "missing_from_prediction": len(target_set - source_set),
                "extra_in_prediction": len(source_set - target_set),
                "duplicate_prediction_ids": len(source_ids) - len(source_set),
                "exact_order_match": source_ids == target_ids,
            }
            allow_superset = bool(method.get("allow_prediction_superset"))
            audit["status"] = "PASS" if audit["missing_from_prediction"] == 0 and audit["duplicate_prediction_ids"] == 0 and (allow_superset or audit["extra_in_prediction"] == 0) else "FAIL"
            mesh_count = sum(value is not None for value in vertex_rows)
            dataset_inventory["methods"][method["name"]] = {
                "prediction": prediction_record, "sample_id_source": order_record,
                "sample_count": len(source_ids), "joint_count": len(xyz_rows[0]),
                "prediction_mesh_valid_count": mesh_count,
                "prediction_mesh_vertex_count": len(next((value for value in vertex_rows if value is not None), [])),
                "analyze": bool(method.get("analyze", True)), "sample_id_audit": audit,
            }
            source_sets[method["name"]] = source_set
            input_paths.update((pred_path, order_path))
            del vertex_rows
            if not method.get("analyze", True):
                del xyz_rows
                gc.collect()
                continue
            predicted_full = np.asarray(xyz_rows, dtype=np.float64)
            predicted = align_prediction_rows(target_ids, source_ids, predicted_full)
            values, skipped = compute_method(predicted, gt, valid)
            results[(spec["name"], method["name"])] = values
            parity_pred = (predicted if parity_ids == target_ids else align_prediction_rows(parity_ids, source_ids, predicted_full))
            parity_values = compute_parity_method(
                parity_pred, parity_gt, parity_valid,
                legacy_alignment=spec.get("parity_alignment") == "legacy_score_predictions",
            )
            checks = parity_rows(method, parity_values, config["parity_tolerance_mm"])
            for check in checks:
                check.update({"dataset": spec["name"], "method": method["name"], "source": spec["parity_source"], "alignment": spec.get("parity_alignment", "proper_procrustes")})
            summary["parity"].extend(checks)
            method["skipped"] = dict(skipped)
            n = len(target_ids)
            for name, array in {
                "dataset": np.full(n, spec["name"]), "method": np.full(n, method["name"]),
                "sample_id": np.asarray(target_ids), "group_id": groups,
                **metadata, **{key: values[key] for key in SUMMARY_METRICS + RAW_METRICS + ADJUSTMENT_METRICS},
                "valid_joint_count": values["valid_joint_count"],
                "per_joint_camera": values["per_joint_camera"], "per_joint_root": values["per_joint_root"],
                "per_joint_pa": values["per_joint_pa"],
            }.items():
                chunks[name].append(np.asarray(array, dtype=np.float32) if np.issubdtype(np.asarray(array).dtype, np.floating) else np.asarray(array))
            del xyz_rows, predicted_full, predicted, parity_pred
            gc.collect()
        names = list(source_sets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                dataset_inventory["sample_id_pairwise"][f"{left}|{right}"] = {
                    "intersection": len(source_sets[left] & source_sets[right]),
                    "left_only": len(source_sets[left] - source_sets[right]),
                    "right_only": len(source_sets[right] - source_sets[left]),
                }
        inventory["datasets"][spec["name"]] = dataset_inventory

        for family, members in spec.get("fold_families", {}).items():
            results[(spec["name"], family)] = mean_methods([results[(spec["name"], member)] for member in members])
        all_methods = [method["name"] for method in spec["methods"] if method.get("analyze", True)] + list(spec.get("fold_families", {}))
        entry = {
            "count": len(target_ids), "bootstrap_unit": spec["bootstrap_unit"],
            "representative": spec["representative"], "display_methods": spec["display_methods"],
            "methods": {}, "versus_base": {}, "comparisons": {}, "subgroups": {}, "skipped": {},
        }
        for method in all_methods:
            values = results[(spec["name"], method)]
            entry["methods"][method] = summarize_values(values, groups, iterations, seed)
            if method in spec.get("fold_families", {}):
                members = spec["fold_families"][method]
                entry["methods"][method]["aggregation"] = "mean of independently scored checkpoints; joints were not averaged before scoring"
                entry["methods"][method]["member_range_mm"] = {
                    metric: [min(entry["methods"][member][metric] for member in members), max(entry["methods"][member][metric] for member in members)]
                    for metric in ("E_camera", "E_root", "E_T", "E_RT", "E_Sim3", "H_T", "H_R", "H_S", "H_local")
                }
            raw_spec = next((method_spec for method_spec in spec["methods"] if method_spec["name"] == method), None)
            entry["skipped"][method] = raw_spec.get("skipped", {}) if raw_spec else {}
            entry["subgroups"][method] = subgroup_summary(values, metadata)
        base = results[(spec["name"], "wilor_base")]
        for method in spec["delta_methods"]:
            entry["versus_base"][method] = comparison(results[(spec["name"], method)], base, groups, iterations, seed)
        for label, candidate, reference in spec.get("comparisons", []):
            entry["comparisons"][label] = comparison(results[(spec["name"], candidate)], results[(spec["name"], reference)], groups, iterations, seed)
        summary["datasets"][spec["name"]] = entry

    failed_parity = [row for row in summary["parity"] if row["status"] != "PASS"]
    failed_ids = [f"{dataset}/{method}" for dataset, value in inventory["datasets"].items() for method, details in value["methods"].items() if details["sample_id_audit"]["status"] != "PASS"]
    inventory["sample_id_audit"] = {"status": "PASS" if not failed_ids else "FAIL", "failures": failed_ids}
    inventory["input_records_after"] = {path: file_record(path) for path in sorted(input_paths)}
    (output_root / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    if failed_ids or failed_parity:
        summary["status"] = "FAIL"
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"gate failure: sample_ids={failed_ids}, parity={failed_parity[:3]}")

    summary["status"] = "PASS"
    summary["joint_names"] = JOINT_NAMES
    decisions, judgment = make_judgment(config, summary, results, groups_by_dataset)
    summary["dataset_decisions"], summary["final_judgment"] = decisions, judgment
    write_per_sample(output_root / "per_sample.npz", chunks)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, output_root / "summary.md")
    write_figure(summary, output_root / "headroom.png")
    (output_root / "analysis_status.json").write_text(json.dumps({
        "status": "PASS", "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
        "git_commit": git_commit, "script_sha256": script_hash,
    }, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "judgment": judgment, "output_root": str(output_root)}, indent=2))


def verify(config_path: Path, output_root: Path, analysis_job: str) -> None:
    config, inventory, summary = read_json(config_path), read_json(output_root / "inventory.json"), read_json(output_root / "summary.json")
    aggregation_tolerance_mm = 1e-3  # per-sample metrics are intentionally stored as float32
    with np.load(output_root / "per_sample.npz") as loaded:
        data = {key: loaded[key] for key in loaded.files}
    aggregation = []
    for dataset, entry in summary["datasets"].items():
        for method, expected in entry["methods"].items():
            chosen = (data["dataset"] == dataset) & (data["method"] == method)
            if not chosen.any():
                continue  # fold-family summaries are means of member metric arrays.
            for metric in ("E_camera", "E_root", "E_T", "E_RT", "E_Sim3"):
                observed = float(np.nanmean(data[metric][chosen]) * 1000.0)
                aggregation.append({"dataset": dataset, "method": method, "metric": metric, "difference_mm": abs(observed - expected[metric])})
    state_lines = subprocess.check_output(["sacct", "-j", analysis_job, "--format=JobIDRaw,State,ExitCode", "-n", "-P"], text=True).splitlines()
    analysis_state = next((line for line in state_lines if line.split("|")[0] == analysis_job), "")
    required = {spec["name"] for spec in config["datasets"]}
    checks = {
        "required_datasets_complete": required == set(summary["datasets"]),
        "parity_all_pass": bool(summary["parity"]) and all(row["status"] == "PASS" for row in summary["parity"]),
        "sample_id_audit_pass": inventory["sample_id_audit"]["status"] == "PASS",
        "summary_matches_per_sample": max((row["difference_mm"] for row in aggregation), default=0.0) <= aggregation_tolerance_mm,
        "bootstrap_units_correct": all(summary["datasets"][spec["name"]]["bootstrap_unit"] == spec["bootstrap_unit"] for spec in config["datasets"]),
        "dexycb_or_interhand_test_read": bool(inventory["forbidden_test_paths_read"]),
        "raw_predictions_or_gt_modified": any(
            file_record(path)["size"] != record["size"] or file_record(path)["mtime_ns"] != record["mtime_ns"]
            for path, record in inventory["input_records_after"].items()
        ),
        "analysis_slurm_completed": "COMPLETED" in analysis_state and "0:0" in analysis_state,
        "required_outputs_exist": all((output_root / name).is_file() for name in ("inventory.json", "per_sample.npz", "summary.json", "summary.md", "headroom.png")),
    }
    verification = {
        "status": "PASS" if all(value if not key.endswith(("_read", "_modified")) else not value for key, value in checks.items()) else "FAIL",
        "checks": checks, "aggregation_tolerance_mm": aggregation_tolerance_mm, "aggregation_checks": aggregation,
        "bootstrap": summary["bootstrap"], "analysis_slurm": analysis_state,
        "git_provenance": {"analysis_commit": inventory["git_commit"], "script_sha256": inventory["script_sha256"]},
        "forbidden_boundary": "DexYCB test and InterHand test were not configured or read",
        "verifier_slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    }
    (output_root / "artifact_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))
    if verification["status"] != "PASS":
        raise RuntimeError("artifact verification failed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-analysis-job")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify_analysis_job:
        verify(args.config, args.output_root, args.verify_analysis_job)
    else:
        analyze(args.config, args.output_root)
