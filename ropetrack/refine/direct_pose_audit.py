"""Deterministic DirectPose gradient-conflict and one-step transfer audit.

The audit consumes only fixed training-split caches.  It keeps the historical
DirectPose implementation as the numerical source of truth and adds the exact
per-finger missing-sensor fallback required by later adaptation experiments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

try:  # Current package layout (0085 consolidation).
    from ropetrack.refine.direct_pose import (
        DirectPoseHead,
        decoded_batch,
        load_arrays,
        pa_align,
        read_npz,
        tensor_batch,
    )
except ImportError:  # Compatibility with the last committed pre-0085 layout.
    from scripts.rope_refiner.direct_pose_head import (  # type: ignore[no-redef]
        DirectPoseHead,
        decoded_batch,
        load_arrays,
        pa_align,
        read_npz,
        tensor_batch,
    )

from ropetrack.eval.pipeline import load_mano_j_regressor
from ropetrack.eval.protocols import canonical_dataset
from ropetrack.refine.actions import FINGER_POSE_GROUPS
from ropetrack.rope import FINGER_CHAINS, FINGER_ORDER

try:
    from ropetrack.refine.apply import mano_layer
except ImportError:
    from scripts.rope_refiner.apply_rope_refinement import mano_layer  # type: ignore[no-redef]


DATASETS = ("arctic", "hot3d", "ho3d_v3", "dexycb", "interhand26m")
CORE_DATASETS = DATASETS[:4]
COMPONENTS = ("pa", "root", "rope", "delta")
PARAMETER_GROUPS = ("condition_query", "rgb_attention", "residual_output")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ExactFallbackDirectPoseHead(DirectPoseHead):
    """Checkpoint-compatible head with structural per-finger fallback."""

    def forward(self, base_pose, base_rope, input_rope, rope_valid, tokens=None):
        refined = super().forward(base_pose, base_rope, input_rope, rope_valid, tokens)
        valid_dims = (rope_valid > 0.5)[:, :, None].expand(-1, -1, 9).reshape(len(base_pose), 45)
        mask = torch.zeros_like(base_pose, dtype=torch.bool).scatter(
            1, self.pose_dims.reshape(1, -1).expand(len(base_pose), -1), valid_dims
        )
        return torch.where(mask, refined, base_pose)


def load_checkpoint_models(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    args = (config["token_dim"], config["hidden_dim"], config["max_delta"])
    raw = DirectPoseHead(*args).to(device)
    safe = ExactFallbackDirectPoseHead(*args).to(device)
    raw.load_state_dict(payload["model_state"])
    safe.load_state_dict(payload["model_state"])
    raw.eval()
    safe.eval()
    return raw, safe, config


def exact_fallback_gate(checkpoint: Path, arrays: dict[str, np.ndarray], device: str) -> dict:
    raw, safe, config = load_checkpoint_models(checkpoint, device)
    rows = np.arange(min(4, len(arrays["sample_id"])))
    batch = tensor_batch(arrays, rows, device)
    base = batch["base_hand_pose"]
    rope = batch["base_rope_norm"]
    measured = batch["input_rope_norm"]
    tokens = batch.get("tokens")
    all_valid = torch.ones((len(rows), 5), device=device)
    with torch.no_grad():
        legacy = raw(base, rope, measured, all_valid, tokens)
        clean = safe(base, rope, measured, all_valid, tokens)
        clean_equal = torch.equal(legacy, clean)
        single_max = []
        valid_path_max = []
        dims = safe.pose_dims
        for finger in range(5):
            valid = all_valid.clone()
            valid[:, finger] = 0
            output = safe(base, rope, measured, valid, tokens)
            single_max.append(float((output[:, dims[finger]] - base[:, dims[finger]]).abs().max()))
            keep = [idx for idx in range(5) if idx != finger]
            valid_path_max.append(float((output[:, dims[keep]] - legacy[:, dims[keep]]).abs().max()))
        all_missing = safe(base, rope, measured, torch.zeros_like(all_valid), tokens)
        all_missing_max = float((all_missing - base).abs().max())
    passed = clean_equal and max(single_max) == 0.0 and max(valid_path_max) == 0.0 and all_missing_max == 0.0
    return {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": {key: config[key] for key in ("token_dim", "hidden_dim", "max_delta")},
        "all_valid_bitwise_equal_to_legacy": clean_equal,
        "single_invalid_finger_pose_max_abs": single_max,
        "other_valid_fingers_vs_legacy_max_abs": valid_path_max,
        "all_invalid_pose_vs_base_max_abs": all_missing_max,
    }


def _load_manifest(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
    return rows


def _default_episode(sample_id: str) -> str:
    return sample_id.replace("\\", "/").rsplit("/", 1)[0]


def _default_subject(dataset: str, sample_id: str) -> str:
    first = sample_id.replace("\\", "/").split("/", 1)[0]
    return first.split("_", 1)[0] if dataset == "hot3d" else first


def _identity_rows(dataset: str, sample_ids: np.ndarray, manifest: dict[str, dict]):
    episodes, subjects, split_values = [], [], []
    for value in sample_ids.astype(str):
        row = manifest.get(value, {})
        episodes.append(str(row.get("episode_id", row.get("sequence", _default_episode(value)))))
        subjects.append(str(row.get("subject_id", row.get("participant_id", _default_subject(dataset, value)))))
        split_values.append(str(row.get("split", row.get("phase", row.get("source_split", "training")))))
    return np.asarray(episodes), np.asarray(subjects), sorted(set(split_values))


def select_update_probe_batches(
    sample_ids: np.ndarray,
    episodes: np.ndarray,
    *,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic sample- and episode-disjoint update/probe rows."""
    required = batch_size * num_batches
    unique = np.asarray(sorted(set(episodes.astype(str).tolist())))
    if len(unique) < 2:
        raise ValueError("audit needs at least two episodes")
    for attempt in range(100):
        rng = np.random.default_rng(seed + attempt)
        shuffled = unique[rng.permutation(len(unique))]
        left, right = np.array_split(shuffled, 2)
        update_pool = np.flatnonzero(np.isin(episodes, left))
        probe_pool = np.flatnonzero(np.isin(episodes, right))
        if len(update_pool) >= required and len(probe_pool) >= required:
            update = update_pool[rng.permutation(len(update_pool))[:required]].reshape(num_batches, batch_size)
            probe = probe_pool[rng.permutation(len(probe_pool))[:required]].reshape(num_batches, batch_size)
            if set(sample_ids[update].reshape(-1)) & set(sample_ids[probe].reshape(-1)):
                raise AssertionError("update/probe sample overlap")
            return update, probe
    raise ValueError(f"cannot form {num_batches}x{batch_size} episode-disjoint update/probe batches")


def _load_dataset_arrays(spec: dict) -> dict[str, np.ndarray]:
    if spec.get("bundle"):
        arrays = read_npz(Path(spec["bundle"]))
    else:
        source = spec["source"]
        arrays = load_arrays(
            Path(source["cache"]),
            Path(source["mano_cache"]),
            Path(source["gt_xyz"]),
            Path(source["run_meta"]),
            Path(source["feature_cache"]),
        )
    required = {
        "sample_id", "base_hand_pose", "base_rope_norm", "input_rope_norm", "rope_valid",
        "rope_chain_m", "fist_ratio", "global_orient", "betas", "is_right", "gt_xyz", "tokens",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"dataset bundle missing {missing}")
    return arrays


def verify_input_hashes(protocol: dict) -> dict[str, dict[str, str]]:
    verified = {}
    for name in DATASETS:
        spec = protocol["datasets"][name]
        expected = spec.get("sha256", {})
        paths = {"episode_manifest": spec.get("episode_manifest")}
        if spec.get("bundle"):
            paths["bundle"] = spec["bundle"]
        else:
            paths.update(spec["source"])
        if set(expected) != {key for key, value in paths.items() if value}:
            raise ValueError(f"{name} protocol hash keys do not match declared inputs")
        verified[name] = {}
        for key, value in paths.items():
            if value:
                actual = sha256_file(Path(value))
                if actual != expected[key]:
                    raise ValueError(f"{name} {key} hash changed after protocol freeze")
                verified[name][key] = actual
    return verified


def prepare_selected_data(protocol: dict, run_root: Path):
    selected = {}
    selection_summary = {}
    for dataset_index, name in enumerate(DATASETS):
        spec = protocol["datasets"][name]
        if spec.get("split_role") != "training":
            raise ValueError(f"{name} is not declared training-only")
        arrays = _load_dataset_arrays(spec)
        sample_ids = np.asarray(arrays["sample_id"]).astype(str)
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError(f"{name} has duplicate sample ids")
        manifest_path = Path(spec["episode_manifest"]) if spec.get("episode_manifest") else None
        manifest = _load_manifest(manifest_path)
        episodes, subjects, splits = _identity_rows(name, sample_ids, manifest)
        forbidden = {"test", "val", "validation", "evaluation"} & {value.lower() for value in splits}
        if forbidden:
            raise ValueError(f"{name} selected non-training split labels: {sorted(forbidden)}")
        update, probe = select_update_probe_batches(
            sample_ids,
            episodes,
            batch_size=int(protocol["sampling"]["batch_size"]),
            num_batches=int(protocol["sampling"]["num_batches"]),
            seed=int(protocol["sampling"]["seed"]) + 1009 * dataset_index,
        )
        ordered = np.concatenate((update.reshape(-1), probe.reshape(-1)))
        subset = {key: np.asarray(value)[ordered] for key, value in arrays.items()}
        count = update.size
        subset_update = np.arange(count).reshape(update.shape)
        subset_probe = np.arange(count, 2 * count).reshape(probe.shape)
        selected[name] = {
            "arrays": subset,
            "update": subset_update,
            "probe": subset_probe,
            "episodes": episodes[ordered],
            "subjects": subjects[ordered],
            "dataset": spec["decoder_dataset"],
        }
        rows = []
        for role, indices in (("update", update), ("probe", probe)):
            for batch_index, batch_rows in enumerate(indices):
                for position, row_index in enumerate(batch_rows):
                    rows.append({
                        "dataset": name,
                        "role": role,
                        "batch_index": batch_index,
                        "position": position,
                        "sample_id": str(sample_ids[row_index]),
                        "episode_id": str(episodes[row_index]),
                        "subject_id": str(subjects[row_index]),
                    })
        manifest_out = run_root / "sample_manifests" / f"{name}.jsonl"
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        update_eps = set(episodes[update].reshape(-1).tolist())
        probe_eps = set(episodes[probe].reshape(-1).tolist())
        update_subjects = set(subjects[update].reshape(-1).tolist())
        probe_subjects = set(subjects[probe].reshape(-1).tolist())
        selection_summary[name] = {
            "source_rows": len(sample_ids),
            "source_sample_id_sha256": sha256_lines(sample_ids),
            "split_values": splits,
            "update_rows": int(update.size),
            "probe_rows": int(probe.size),
            "update_sample_id_sha256": sha256_lines(sample_ids[update].reshape(-1)),
            "probe_sample_id_sha256": sha256_lines(sample_ids[probe].reshape(-1)),
            "manifest_sha256": sha256_file(manifest_out),
            "update_episode_count": len(update_eps),
            "probe_episode_count": len(probe_eps),
            "episode_overlap": sorted(update_eps & probe_eps),
            "update_subjects": sorted(update_subjects),
            "probe_subjects": sorted(probe_subjects),
            "subject_overlap": sorted(update_subjects & probe_subjects),
        }
        del arrays
    write_json(run_root / "sample_manifests" / "summary.json", selection_summary)
    return selected, selection_summary


def _mano_resources(dataset: str, device: str):
    mano = mano_layer(device)
    mano.requires_grad_(False)
    regressor = None
    if canonical_dataset(dataset) == "dexycb":
        repo = Path(__file__).resolve().parents[2]
        regressor = torch.from_numpy(load_mano_j_regressor(repo / "mano_data" / "MANO_RIGHT.pkl")).to(device)
    return mano, regressor


def _losses(model, mano, batch, dataset: str, weights: dict, regressor=None):
    refined, joints, pred_rope = decoded_batch(model, mano, batch, dataset, regressor)
    gt = batch["gt_xyz"]
    aligned = pa_align(joints, gt)
    pred_root = joints - joints[:, :1]
    gt_root = gt - gt[:, :1]
    valid = batch["rope_valid"]
    components = {
        "pa": torch.nn.functional.l1_loss(aligned, gt),
        "root": torch.nn.functional.l1_loss(pred_root, gt_root),
        "rope": (((pred_rope - batch["input_rope_norm"]) * valid) ** 2).sum() / valid.sum().clamp_min(1.0),
        "delta": ((refined - batch["base_hand_pose"]) ** 2).mean(),
    }
    total = components["pa"] + weights["root"] * components["root"] + weights["rope"] * components["rope"] + weights["delta"] * components["delta"]
    finger_losses = {}
    pose_dims = model.pose_dims
    for finger, chain in enumerate(FINGER_CHAINS["freihand"]):
        joints_idx = list(chain[1:])
        pa = torch.nn.functional.l1_loss(aligned[:, joints_idx], gt[:, joints_idx])
        root = torch.nn.functional.l1_loss(pred_root[:, joints_idx], gt_root[:, joints_idx])
        denom = valid[:, finger].sum().clamp_min(1.0)
        rope = (((pred_rope[:, finger] - batch["input_rope_norm"][:, finger]) * valid[:, finger]) ** 2).sum() / denom
        delta = ((refined[:, pose_dims[finger]] - batch["base_hand_pose"][:, pose_dims[finger]]) ** 2).mean()
        finger_losses[FINGER_ORDER[finger]] = pa + weights["root"] * root + weights["rope"] * rope + weights["delta"] * delta
    return total, finger_losses, components


def _flat_gradient(loss, parameters, *, retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
        for parameter, grad in zip(parameters, grads, strict=True)
    ])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denom.clamp_min(1e-20))


def _parameter_group(name: str) -> str:
    if name.startswith("query."):
        return "condition_query"
    if name.startswith(("token_proj.", "attention.")):
        return "rgb_attention"
    if name.startswith("output."):
        return "residual_output"
    raise ValueError(f"unassigned DirectPose parameter: {name}")


def _grouped_gradients(loss, named_parameters, *, retain_graph: bool) -> dict[str, torch.Tensor]:
    parameters = [parameter for _, parameter in named_parameters]
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    resolved = [torch.zeros_like(parameter) if grad is None else grad for parameter, grad in zip(parameters, grads, strict=True)]
    result = {"all": torch.cat([grad.reshape(-1) for grad in resolved])}
    for group in PARAMETER_GROUPS:
        values = [
            grad.reshape(-1)
            for (name, _), grad in zip(named_parameters, resolved, strict=True)
            if _parameter_group(name) == group
        ]
        result[group] = torch.cat(values) if values else torch.zeros(1, device=loss.device)
    return result


def _scenario_batch(arrays: dict[str, np.ndarray], rows: np.ndarray, device: str, mode: str, seed: int):
    batch = tensor_batch(arrays, rows, device)
    if mode == "correct":
        return batch
    batch = dict(batch)
    if mode == "shuffle":
        permutation = torch.as_tensor(np.random.default_rng(seed).permutation(len(rows)), device=device)
        batch["input_rope_norm"] = batch["input_rope_norm"][permutation]
        batch["rope_valid"] = batch["rope_valid"][permutation]
        return batch
    if mode == "zero":
        batch["input_rope_norm"] = torch.zeros_like(batch["input_rope_norm"])
        batch["rope_valid"] = torch.zeros_like(batch["rope_valid"])
        return batch
    raise ValueError(f"unknown attribution rope mode: {mode}")


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(replicates, len(values)))
    boot = values[samples].mean(axis=1)
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def _matrix_summary(values: np.ndarray, seed: int, replicates: int) -> dict:
    low, high = _bootstrap(values, seed=seed, replicates=replicates)
    return {"mean": values.mean(axis=0).tolist(), "ci95_low": low.tolist(), "ci95_high": high.tolist()}


def run_gradient_audit(protocol: dict, selected: dict, model, device: str, run_root: Path) -> dict:
    num_batches = int(protocol["sampling"]["num_batches"])
    weights = protocol["loss_weights"]
    replicates = int(protocol["bootstrap"]["replicates"])
    seed = int(protocol["bootstrap"]["seed"])
    keys = ("overall", *FINGER_ORDER)
    cross = {key: np.empty((num_batches, 5, 5), dtype=np.float64) for key in keys}
    overall_norm = np.empty((num_batches, 5), dtype=np.float64)
    component_norm = np.empty((num_batches, 5, 4), dtype=np.float64)
    component_cos = np.empty((num_batches, 5, 4, 4), dtype=np.float64)
    resources = {name: _mano_resources(selected[name]["dataset"], device) for name in DATASETS}
    for batch_index in range(num_batches):
        gradients = {}
        for dataset_index, name in enumerate(DATASETS):
            entry = selected[name]
            batch = tensor_batch(entry["arrays"], entry["update"][batch_index], device)
            model.zero_grad(set_to_none=True)
            total, fingers, components = _losses(
                model, resources[name][0], batch, entry["dataset"], weights, resources[name][1]
            )
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            requested = [("overall", total), *fingers.items(), *components.items()]
            gradients[name] = {}
            for index, (key, loss) in enumerate(requested):
                gradients[name][key] = _flat_gradient(loss, parameters, retain_graph=index + 1 < len(requested)).detach()
            overall_norm[batch_index, dataset_index] = float(torch.linalg.vector_norm(gradients[name]["overall"]))
            for left, component_left in enumerate(COMPONENTS):
                component_norm[batch_index, dataset_index, left] = float(torch.linalg.vector_norm(gradients[name][component_left]))
                for right, component_right in enumerate(COMPONENTS):
                    component_cos[batch_index, dataset_index, left, right] = _cosine(
                        gradients[name][component_left], gradients[name][component_right]
                    )
        for key in keys:
            for left, left_name in enumerate(DATASETS):
                for right, right_name in enumerate(DATASETS):
                    cross[key][batch_index, left, right] = _cosine(gradients[left_name][key], gradients[right_name][key])
    raw_path = run_root / "gradient_raw.npz"
    np.savez(
        raw_path,
        dataset_order=np.asarray(DATASETS),
        finger_order=np.asarray(FINGER_ORDER),
        component_order=np.asarray(COMPONENTS),
        overall_gradient_norm=overall_norm,
        component_gradient_norm=component_norm,
        component_cosine=component_cos,
        **{f"cosine_{key}": value for key, value in cross.items()},
    )
    component_low, component_high = _bootstrap(component_cos, seed=seed + 700, replicates=replicates)
    norm_low, norm_high = _bootstrap(component_norm, seed=seed + 701, replicates=replicates)
    summary = {
        "dataset_order": list(DATASETS),
        "finger_order": list(FINGER_ORDER),
        "component_order": list(COMPONENTS),
        "cross_dataset": {
            key: _matrix_summary(value, seed + 100 + index, replicates)
            for index, (key, value) in enumerate(cross.items())
        },
        "overall_gradient_norm": {
            "mean": overall_norm.mean(axis=0).tolist(),
            "per_batch": overall_norm.tolist(),
        },
        "component_gradient_norm": {
            "mean": component_norm.mean(axis=0).tolist(),
            "ci95_low": norm_low.tolist(),
            "ci95_high": norm_high.tolist(),
        },
        "within_dataset_component_cosine": {
            "mean": component_cos.mean(axis=0).tolist(),
            "ci95_low": component_low.tolist(),
            "ci95_high": component_high.tolist(),
        },
        "raw_sha256": sha256_file(raw_path),
    }
    write_json(run_root / "gradient_summary.json", summary)
    return summary


def _matched_rows(left: dict, right: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match unique update rows by base/measured normalized rope state."""
    left_rows = left["update"].reshape(-1)
    right_rows = right["update"].reshape(-1)
    left_features = np.concatenate((left["arrays"]["base_rope_norm"][left_rows], left["arrays"]["input_rope_norm"][left_rows]), axis=1)
    right_features = np.concatenate((right["arrays"]["base_rope_norm"][right_rows], right["arrays"]["input_rope_norm"][right_rows]), axis=1)
    pooled = np.concatenate((left_features, right_features), axis=0).astype(np.float64)
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    left_features = (left_features - mean) / np.maximum(std, 1e-6)
    right_features = (right_features - mean) / np.maximum(std, 1e-6)
    available = np.ones(len(right_rows), dtype=bool)
    chosen_left, chosen_right, distances = [], [], []
    # ponytail: greedy unique matching is enough for 384-row attribution; use
    # optimal assignment only if its recorded distance tail changes a decision.
    for left_index in np.random.default_rng(seed).permutation(len(left_rows)):
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        delta = right_features[candidates] - left_features[left_index]
        nearest_at = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
        right_index = int(candidates[nearest_at])
        available[right_index] = False
        chosen_left.append(int(left_rows[left_index]))
        chosen_right.append(int(right_rows[right_index]))
        distances.append(float(np.linalg.norm(delta[nearest_at])))
    return np.asarray(chosen_left), np.asarray(chosen_right), np.asarray(distances)


def _dataset_attribution_stats(selected: dict, model, device: str) -> dict:
    stats = {}
    resources = {name: _mano_resources(selected[name]["dataset"], device) for name in DATASETS}
    for name in DATASETS:
        entry = selected[name]
        arrays = entry["arrays"]
        probe_metrics, base_metrics = [], []
        for batch_index, rows in enumerate(entry["probe"]):
            batch = _scenario_batch(arrays, rows, device, "correct", 0)
            probe_metrics.append(_error_metrics(model, resources[name][0], batch, entry["dataset"], resources[name][1]))
            base_batch = dict(batch)
            base_batch["rope_valid"] = torch.zeros_like(batch["rope_valid"])
            base_metrics.append(_error_metrics(model, resources[name][0], base_batch, entry["dataset"], resources[name][1]))
        input_rope = np.asarray(arrays["input_rope_norm"], dtype=np.float64)
        base_rope = np.asarray(arrays["base_rope_norm"], dtype=np.float64)
        stats[name] = {
            "rows": len(input_rope),
            "valid_fraction": float(np.asarray(arrays["rope_valid"]).mean()),
            "input_rope_mean": input_rope.mean(axis=0).tolist(),
            "input_rope_std": input_rope.std(axis=0).tolist(),
            "input_minus_base_mean": (input_rope - base_rope).mean(axis=0).tolist(),
            "input_minus_base_std": (input_rope - base_rope).std(axis=0).tolist(),
            "token_rms": float(np.sqrt(np.mean(np.asarray(arrays["tokens"], dtype=np.float64) ** 2))),
            "wilor_base_metrics": np.mean(base_metrics, axis=0).tolist(),
            "current_head_metrics": np.mean(probe_metrics, axis=0).tolist(),
        }
    return stats


def run_attribution_audit(protocol: dict, selected: dict, model, device: str, run_root: Path) -> dict:
    config = protocol["attribution"]
    scenarios = config["scenarios"]
    scenario_names = list(scenarios)
    keys = ("overall", *FINGER_ORDER)
    num_batches = int(protocol["sampling"]["num_batches"])
    replicates = int(protocol["bootstrap"]["replicates"])
    bootstrap_seed = int(protocol["bootstrap"]["seed"])
    cross = np.empty((len(scenario_names), len(keys), num_batches, len(DATASETS), len(DATASETS)), dtype=np.float64)
    grouped = np.empty((len(scenario_names), len(PARAMETER_GROUPS), num_batches, len(DATASETS), len(DATASETS)), dtype=np.float64)
    resources = {name: _mano_resources(selected[name]["dataset"], device) for name in DATASETS}
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    for scenario_index, scenario_name in enumerate(scenario_names):
        scenario = scenarios[scenario_name]
        weights = scenario["loss_weights"]
        for batch_index in range(num_batches):
            gradients = {}
            for dataset_index, name in enumerate(DATASETS):
                entry = selected[name]
                batch = _scenario_batch(
                    entry["arrays"], entry["update"][batch_index], device, scenario["rope_mode"],
                    int(config["seed"]) + 100003 * scenario_index + 1009 * dataset_index + batch_index,
                )
                total, fingers, _ = _losses(model, resources[name][0], batch, entry["dataset"], weights, resources[name][1])
                requested = [("overall", total), *fingers.items()]
                gradients[name] = {}
                for loss_index, (key, loss) in enumerate(requested):
                    gradients[name][key] = _grouped_gradients(
                        loss, named_parameters, retain_graph=loss_index + 1 < len(requested)
                    )
            for key_index, key in enumerate(keys):
                for left, left_name in enumerate(DATASETS):
                    for right, right_name in enumerate(DATASETS):
                        cross[scenario_index, key_index, batch_index, left, right] = _cosine(
                            gradients[left_name][key]["all"], gradients[right_name][key]["all"]
                        )
            for group_index, group in enumerate(PARAMETER_GROUPS):
                for left, left_name in enumerate(DATASETS):
                    for right, right_name in enumerate(DATASETS):
                        grouped[scenario_index, group_index, batch_index, left, right] = _cosine(
                            gradients[left_name]["overall"][group], gradients[right_name]["overall"][group]
                        )

    raw = {
        "scenario_order": np.asarray(scenario_names),
        "key_order": np.asarray(keys),
        "parameter_group_order": np.asarray(PARAMETER_GROUPS),
        "dataset_order": np.asarray(DATASETS),
        "cross_dataset_cosine": cross,
        "parameter_group_cosine": grouped,
    }
    summary = {
        "scenario_order": scenario_names,
        "key_order": list(keys),
        "parameter_group_order": list(PARAMETER_GROUPS),
        "dataset_order": list(DATASETS),
        "scenarios": {},
        "dataset_stats": _dataset_attribution_stats(selected, model, device),
    }
    for scenario_index, scenario_name in enumerate(scenario_names):
        summary["scenarios"][scenario_name] = {
            "cross_dataset": {
                key: _matrix_summary(cross[scenario_index, key_index], bootstrap_seed + 1000 * scenario_index + key_index, replicates)
                for key_index, key in enumerate(keys)
            },
            "parameter_groups": {
                group: _matrix_summary(grouped[scenario_index, group_index], bootstrap_seed + 10000 + 1000 * scenario_index + group_index, replicates)
                for group_index, group in enumerate(PARAMETER_GROUPS)
            },
        }
    reference = scenario_names[0]
    summary["scenario_effect_vs_reference"] = {}
    for scenario_index, scenario_name in enumerate(scenario_names[1:], start=1):
        summary["scenario_effect_vs_reference"][scenario_name] = {
            key: _matrix_summary(
                cross[scenario_index, key_index] - cross[0, key_index],
                bootstrap_seed + 20000 + 1000 * scenario_index + key_index,
                replicates,
            )
            for key_index, key in enumerate(keys)
        }

    matched = {}
    for pair_index, (left_name, right_name) in enumerate(config["match_pairs"]):
        left_rows, right_rows, distances = _matched_rows(
            selected[left_name], selected[right_name], int(config["seed"]) + 50000 + pair_index
        )
        usable = min(len(left_rows), num_batches * int(protocol["sampling"]["batch_size"]))
        left_rows = left_rows[:usable].reshape(num_batches, -1)
        right_rows = right_rows[:usable].reshape(num_batches, -1)
        pair_result = {
            "rows": usable,
            "feature": "standardized [base_rope_norm,input_rope_norm]",
            "match_distance_mean": float(distances[:usable].mean()),
            "match_distance_p95": float(np.percentile(distances[:usable], 95)),
            "scenarios": {},
        }
        for scenario_index, scenario_name in enumerate(config["matched_scenarios"]):
            scenario = scenarios[scenario_name]
            values = np.empty((num_batches, len(keys)), dtype=np.float64)
            group_values = np.empty((num_batches, len(PARAMETER_GROUPS)), dtype=np.float64)
            for batch_index in range(num_batches):
                pair_gradients = []
                for side_index, (name, rows) in enumerate(((left_name, left_rows[batch_index]), (right_name, right_rows[batch_index]))):
                    entry = selected[name]
                    batch = _scenario_batch(
                        entry["arrays"], rows, device, scenario["rope_mode"],
                        int(config["seed"]) + 70000 + 1000 * pair_index + 100 * scenario_index + 10 * batch_index + side_index,
                    )
                    total, fingers, _ = _losses(
                        model, resources[name][0], batch, entry["dataset"], scenario["loss_weights"], resources[name][1]
                    )
                    requested = [("overall", total), *fingers.items()]
                    pair_gradients.append({
                        key: _grouped_gradients(loss, named_parameters, retain_graph=index + 1 < len(requested))
                        for index, (key, loss) in enumerate(requested)
                    })
                for key_index, key in enumerate(keys):
                    values[batch_index, key_index] = _cosine(pair_gradients[0][key]["all"], pair_gradients[1][key]["all"])
                for group_index, group in enumerate(PARAMETER_GROUPS):
                    group_values[batch_index, group_index] = _cosine(
                        pair_gradients[0]["overall"][group], pair_gradients[1]["overall"][group]
                    )
            low, high = _bootstrap(values, seed=bootstrap_seed + 30000 + 100 * pair_index + scenario_index, replicates=replicates)
            group_low, group_high = _bootstrap(group_values, seed=bootstrap_seed + 31000 + 100 * pair_index + scenario_index, replicates=replicates)
            pair_result["scenarios"][scenario_name] = {
                "mean": dict(zip(keys, values.mean(axis=0).tolist(), strict=True)),
                "ci95_low": dict(zip(keys, low.tolist(), strict=True)),
                "ci95_high": dict(zip(keys, high.tolist(), strict=True)),
                "parameter_group_mean": dict(zip(PARAMETER_GROUPS, group_values.mean(axis=0).tolist(), strict=True)),
                "parameter_group_ci95_low": dict(zip(PARAMETER_GROUPS, group_low.tolist(), strict=True)),
                "parameter_group_ci95_high": dict(zip(PARAMETER_GROUPS, group_high.tolist(), strict=True)),
            }
            raw[f"matched_{left_name}_{right_name}_{scenario_name}"] = values
            raw[f"matched_groups_{left_name}_{right_name}_{scenario_name}"] = group_values
        matched[f"{left_name}__{right_name}"] = pair_result
    summary["rope_state_matched_pairs"] = matched
    raw_path = run_root / "attribution_raw.npz"
    np.savez(raw_path, **raw)
    summary["raw_sha256"] = sha256_file(raw_path)
    write_json(run_root / "attribution_summary.json", summary)
    return summary


def _error_metrics(model, mano, batch, dataset: str, regressor=None) -> np.ndarray:
    with torch.no_grad():
        _, joints, _ = decoded_batch(model, mano, batch, dataset, regressor)
        gt = batch["gt_xyz"]
        aligned = pa_align(joints, gt)
        root = joints - joints[:, :1]
        gt_root = gt - gt[:, :1]
        metrics = [
            torch.linalg.vector_norm(aligned - gt, dim=-1).mean() * 1000.0,
            torch.linalg.vector_norm(root - gt_root, dim=-1).mean() * 1000.0,
        ]
        for chain in FINGER_CHAINS["freihand"]:
            indices = list(chain[1:])
            metrics.append(torch.linalg.vector_norm(aligned[:, indices] - gt[:, indices], dim=-1).mean() * 1000.0)
    return np.asarray([float(value) for value in metrics], dtype=np.float64)


def _one_step(model, mano, batch, dataset: str, regressor, protocol: dict, mode: str) -> float:
    model.train()
    weights = protocol["loss_weights"]
    total, _, _ = _losses(model, mano, batch, dataset, weights, regressor)
    if mode == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(protocol["one_step"]["lr"]),
            weight_decay=float(protocol["one_step"]["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["one_step"]["grad_clip_norm"]))
        before = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        optimizer.step()
        after = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
        step_norm = float(torch.linalg.vector_norm(after - before))
    elif mode == "normalized_gradient":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grads = torch.autograd.grad(total, parameters, allow_unused=True)
        flat = torch.cat([
            (torch.zeros_like(parameter) if grad is None else grad).reshape(-1)
            for parameter, grad in zip(parameters, grads, strict=True)
        ])
        step_norm = float(protocol["one_step"]["normalized_gradient_l2"])
        scale = step_norm / float(torch.linalg.vector_norm(flat).clamp_min(1e-20))
        with torch.no_grad():
            for parameter, grad in zip(parameters, grads, strict=True):
                if grad is not None:
                    parameter.add_(grad, alpha=-scale)
    else:
        raise ValueError(mode)
    model.eval()
    return step_norm


def run_transfer_audit(protocol: dict, selected: dict, base_model, device: str, run_root: Path) -> dict:
    num_batches = int(protocol["sampling"]["num_batches"])
    replicates = int(protocol["bootstrap"]["replicates"])
    seed = int(protocol["bootstrap"]["seed"])
    metric_names = ("pa", "root", *FINGER_ORDER)
    resources = {name: _mano_resources(selected[name]["dataset"], device) for name in DATASETS}
    before = np.empty((num_batches, 5, 7), dtype=np.float64)
    for batch_index in range(num_batches):
        for target_index, target in enumerate(DATASETS):
            entry = selected[target]
            batch = tensor_batch(entry["arrays"], entry["probe"][batch_index], device)
            before[batch_index, target_index] = _error_metrics(
                base_model, resources[target][0], batch, entry["dataset"], resources[target][1]
            )
    raw = {}
    summary = {"dataset_order": list(DATASETS), "metric_order": list(metric_names), "modes": {}}
    for mode_index, mode in enumerate(("adamw", "normalized_gradient")):
        after = np.empty((num_batches, 5, 5, 7), dtype=np.float64)
        step_norm = np.empty((num_batches, 5), dtype=np.float64)
        for batch_index in range(num_batches):
            for source_index, source in enumerate(DATASETS):
                model = copy.deepcopy(base_model)
                source_entry = selected[source]
                source_batch = tensor_batch(source_entry["arrays"], source_entry["update"][batch_index], device)
                step_norm[batch_index, source_index] = _one_step(
                    model,
                    resources[source][0],
                    source_batch,
                    source_entry["dataset"],
                    resources[source][1],
                    protocol,
                    mode,
                )
                for target_index, target in enumerate(DATASETS):
                    target_entry = selected[target]
                    target_batch = tensor_batch(target_entry["arrays"], target_entry["probe"][batch_index], device)
                    after[batch_index, source_index, target_index] = _error_metrics(
                        model,
                        resources[target][0],
                        target_batch,
                        target_entry["dataset"],
                        resources[target][1],
                    )
        delta = after - before[:, None, :, :]
        raw[f"{mode}_before"] = before
        raw[f"{mode}_after"] = after
        raw[f"{mode}_delta"] = delta
        raw[f"{mode}_step_l2"] = step_norm
        mode_summary = {"step_l2_mean": step_norm.mean(axis=0).tolist(), "metrics": {}}
        for metric_index, metric in enumerate(metric_names):
            low, high = _bootstrap(delta[:, :, :, metric_index], seed=seed + 1000 + mode_index * 100 + metric_index, replicates=replicates)
            mode_summary["metrics"][metric] = {
                "before": np.broadcast_to(before[:, None, :, metric_index], delta[:, :, :, metric_index].shape).mean(axis=0).tolist(),
                "after": after[:, :, :, metric_index].mean(axis=0).tolist(),
                "delta": delta[:, :, :, metric_index].mean(axis=0).tolist(),
                "delta_ci95_low": low.tolist(),
                "delta_ci95_high": high.tolist(),
            }
        summary["modes"][mode] = mode_summary
    raw_path = run_root / "transfer_raw.npz"
    np.savez(raw_path, dataset_order=np.asarray(DATASETS), metric_order=np.asarray(metric_names), **raw)
    summary["raw_sha256"] = sha256_file(raw_path)
    write_json(run_root / "transfer_summary.json", summary)
    return summary


def evaluate_conflict_gate(protocol: dict, gradient: dict, transfer: dict, safety: dict) -> dict:
    thresholds = protocol["conflict_gate"]
    cosine_floor = float(thresholds["core_pair_cosine_ci_low_min"])
    regression = float(thresholds["significant_transfer_regression_mm"])
    overall = gradient["cross_dataset"]["overall"]
    core_cosine_failures = []
    for left in range(4):
        for right in range(left + 1, 4):
            low = overall["ci95_low"][left][right]
            if low < cosine_floor:
                core_cosine_failures.append({"source": DATASETS[left], "target": DATASETS[right], "ci95_low": low})
    transfer_failures = []
    for mode, mode_summary in transfer["modes"].items():
        for metric in ("pa", "root"):
            result = mode_summary["metrics"][metric]
            for source in range(4):
                for target in range(4):
                    if source == target:
                        continue
                    mean = result["delta"][source][target]
                    low = result["delta_ci95_low"][source][target]
                    if mean > regression and low > 0.0:
                        transfer_failures.append({
                            "mode": mode, "metric": metric, "source": DATASETS[source],
                            "target": DATASETS[target], "delta_mm": mean, "ci95_low": low,
                        })
    equal_weight_supported = not core_cosine_failures and not transfer_failures
    pa_index, root_index = COMPONENTS.index("pa"), COMPONENTS.index("root")
    dex_index = DATASETS.index("dexycb")
    within = gradient["within_dataset_component_cosine"]
    dex_ci = [
        within["ci95_low"][dex_index][pa_index][root_index],
        within["ci95_high"][dex_index][pa_index][root_index],
    ]
    dex_conflict = dex_ci[1] < 0.0
    inter_index = DATASETS.index("interhand26m")
    inter_conflicts = []
    for core in range(4):
        high = overall["ci95_high"][core][inter_index]
        if high < 0.0:
            inter_conflicts.append({"dataset": DATASETS[core], "ci95_high": high})
    finger_scores = {}
    for finger in FINGER_ORDER:
        matrix = gradient["cross_dataset"][finger]["mean"]
        finger_scores[finger] = float(np.mean([matrix[left][right] for left in range(4) for right in range(left + 1, 4)]))
    ranked = sorted(finger_scores, key=finger_scores.get)
    if safety["status"] != "PASS":
        decision = "STOP"
    elif equal_weight_supported:
        decision = "CONTINUE"
    elif core_cosine_failures or transfer_failures:
        decision = "STOP"
    else:
        decision = "NOT PROVEN"
    return {
        "decision": decision,
        "four_core_equal_or_near_equal_mix_supported": equal_weight_supported,
        "core_cosine_failures": core_cosine_failures,
        "core_transfer_failures": transfer_failures,
        "dexycb_pa_vs_root": {
            "mean_cosine": within["mean"][dex_index][pa_index][root_index],
            "ci95": dex_ci,
            "significant_direction_conflict": dex_conflict,
        },
        "interhand_vs_core": {
            "significant_overall_gradient_conflicts": inter_conflicts,
            "significantly_conflicting": bool(inter_conflicts),
        },
        "finger_conflict_ranking_most_negative_first": ranked,
        "finger_mean_core_pair_cosine": finger_scores,
        "thumb_or_ring_most_conflicted": ranked[0] in {"thumb", "ring"},
        "safety_gate": safety["status"],
        "training_cells_authorized": decision == "CONTINUE" and safety["status"] == "PASS",
    }


def _markdown_ci_matrix(title: str, result: dict) -> list[str]:
    lines = [f"### {title}", "", "Cells are mean `[95% CI]`.", "", "| source \\ target | " + " | ".join(DATASETS) + " |", "|---|" + "---:|" * 5]
    for source, row, low, high in zip(
        DATASETS, result["mean"], result["ci95_low"], result["ci95_high"], strict=True
    ):
        cells = [f"{value:+.3f} `[{lo:+.3f},{hi:+.3f}]`" for value, lo, hi in zip(row, low, high, strict=True)]
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _markdown_transfer(title: str, result: dict) -> list[str]:
    lines = [
        f"### {title}", "",
        "Cells are absolute after-step mm; `(signed delta [95% CI])`. Negative delta improves.", "",
        "| source \\ target | " + " | ".join(DATASETS) + " |", "|---|" + "---:|" * 5,
    ]
    for source, after, delta, low, high in zip(
        DATASETS,
        result["after"],
        result["delta"],
        result["delta_ci95_low"],
        result["delta_ci95_high"],
        strict=True,
    ):
        cells = [
            f"{value:.3f} `({change:+.3f} [{lo:+.3f},{hi:+.3f}])`"
            for value, change, lo, hi in zip(after, delta, low, high, strict=True)
        ]
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_report(run_root: Path, protocol: dict, selection: dict, gradient: dict, transfer: dict, safety: dict, gate: dict) -> None:
    lines = [
        "# DirectPose multi-dataset gradient-conflict audit", "",
        f"Decision: **{gate['decision']}**. Four-core equal/near-equal mixing supported: "
        f"**{gate['four_core_equal_or_near_equal_mix_supported']}**.", "",
        "All five inputs are fixed training-split rows. InterHand is stress/audit only. "
        "Rope is GT-derived ideal geometry, not physical-sensor evidence.", "",
        "## Safety gate", "", f"Exact fallback: **{safety['status']}**.", "",
        "## Fixed sample boundary", "",
        "| dataset | source | update | probe | update episodes | probe episodes | episode overlap |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in DATASETS:
        row = selection[name]
        lines.append(f"| {name} | {row['source_rows']} | {row['update_rows']} | {row['probe_rows']} | {row['update_episode_count']} | {row['probe_episode_count']} | {len(row['episode_overlap'])} |")
    lines += [""]
    lines += _markdown_ci_matrix("Overall gradient cosine", gradient["cross_dataset"]["overall"])
    for finger in FINGER_ORDER:
        lines += _markdown_ci_matrix(f"{finger} gradient cosine", gradient["cross_dataset"][finger])
    lines += ["## PA-vs-root within-domain cosine", "", "| dataset | mean | 95% CI |", "|---|---:|---:|"]
    pa_index, root_index = COMPONENTS.index("pa"), COMPONENTS.index("root")
    component = gradient["within_dataset_component_cosine"]
    for index, name in enumerate(DATASETS):
        lines.append(
            f"| {name} | {component['mean'][index][pa_index][root_index]:+.4f} | "
            f"[{component['ci95_low'][index][pa_index][root_index]:+.4f},"
            f"{component['ci95_high'][index][pa_index][root_index]:+.4f}] |"
        )
    lines += ["", "## One-step transfer", ""]
    for mode in ("adamw", "normalized_gradient"):
        lines += [
            f"### {mode} step size", "",
            "Parameter L2 by source: " + ", ".join(
                f"{name}={value:.4f}" for name, value in zip(DATASETS, transfer["modes"][mode]["step_l2_mean"], strict=True)
            ) + ".", "",
        ]
        metrics = ("pa", "root", *FINGER_ORDER) if mode == "adamw" else ("pa", "root")
        for metric in metrics:
            lines += _markdown_transfer(f"{mode}: {metric}", transfer["modes"][mode]["metrics"][metric])
    lines += [
        "## Required conclusions", "",
        f"- ARCTIC/HOT3D/HO3D/DexYCB equal or near-equal mix: **{gate['four_core_equal_or_near_equal_mix_supported']}**.",
        f"- DexYCB PA-vs-root significant directional conflict: **{gate['dexycb_pa_vs_root']['significant_direction_conflict']}**; cosine {gate['dexycb_pa_vs_root']['mean_cosine']:+.4f}, CI {gate['dexycb_pa_vs_root']['ci95']}.",
        f"- InterHand significantly conflicts with core: **{gate['interhand_vs_core']['significantly_conflicting']}**.",
        f"- Most conflicted finger: **{gate['finger_conflict_ranking_most_negative_first'][0]}**; thumb/ring concentration flag: **{gate['thumb_or_ring_most_conflicted']}**.",
        f"- Training cells authorized: **{gate['training_cells_authorized']}**.", "",
        "## Boundary", "",
        "The formal release remains RopeAlphaStudent. DirectPose changes only the experimental 45D local MANO residual; global orientation, camera translation, betas, and scale remain frozen. No result here validates physical rope calibration, slack, hysteresis, drift, latency, wear, or dropout.", "",
    ]
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_attribution(protocol: dict, attribution: dict, safety: dict) -> dict:
    names = attribution["dataset_order"]
    scenarios = attribution["scenarios"]

    def cell(scenario, left, right, section="cross_dataset", key="overall"):
        result = scenarios[scenario][section][key]
        i, j = names.index(left), names.index(right)
        return {
            "mean": result["mean"][i][j],
            "ci95": [result["ci95_low"][i][j], result["ci95_high"][i][j]],
        }

    def effect(scenario, left, right):
        result = attribution["scenario_effect_vs_reference"][scenario]["overall"]
        i, j = names.index(left), names.index(right)
        return {
            "mean": result["mean"][i][j],
            "ci95": [result["ci95_low"][i][j], result["ci95_high"][i][j]],
        }

    pairs = {}
    for left, right in protocol["attribution"]["diagnostic_pairs"]:
        key = f"{left}__{right}"
        full = cell("full_correct", left, right)
        shuffled = cell("full_shuffled", left, right)
        task = cell("task_correct", left, right)
        full_match = attribution["rope_state_matched_pairs"][key]["scenarios"]["full_correct"]
        task_match = attribution["rope_state_matched_pairs"][key]["scenarios"]["task_correct"]
        pairs[key] = {
            "full_correct": full,
            "full_shuffled": shuffled,
            "task_correct": task,
            "shuffle_minus_full": effect("full_shuffled", left, right),
            "task_minus_full": effect("task_correct", left, right),
            "rope_state_matched_full": {
                "mean": full_match["mean"]["overall"],
                "ci95": [full_match["ci95_low"]["overall"], full_match["ci95_high"]["overall"]],
            },
            "rope_state_matched_task": {
                "mean": task_match["mean"]["overall"],
                "ci95": [task_match["ci95_low"]["overall"], task_match["ci95_high"]["overall"]],
            },
            "significant_full_conflict": full["ci95"][1] < 0.0,
            "paired_rope_input_contributes": effect("full_shuffled", left, right)["ci95"][0] > 0.0,
            "auxiliary_losses_contribute": effect("task_correct", left, right)["ci95"][0] > 0.0,
            "persistent_after_task_and_rope_state_control": task["ci95"][1] < 0.0 and task_match["ci95_high"]["overall"] < 0.0,
        }
    hot_dex = pairs["hot3d__dexycb"]
    if hot_dex["persistent_after_task_and_rope_state_control"]:
        hot_dex_diagnosis = "persistent domain-conditioned pose/RGB mapping conflict"
    elif hot_dex["paired_rope_input_contributes"] or hot_dex["auxiliary_losses_contribute"]:
        hot_dex_diagnosis = "rope conditioning or auxiliary objective contributes materially"
    else:
        hot_dex_diagnosis = "sample-state distribution contributes; exact source not proven"
    return {
        "decision": "CONTINUE_HEAD_ONLY_PRODUCT_PILOT" if safety["status"] == "PASS" else "STOP",
        "hot3d_dexycb_diagnosis": hot_dex_diagnosis,
        "diagnostic_pairs": pairs,
        "training_roles": {
            "positive_target": ["hot3d"],
            "priority_retention": ["arctic"],
            "secondary_retention": ["ho3d_v3", "dexycb"],
            "stress_only": ["interhand26m"],
        },
        "head_only_product_pilot_authorized": safety["status"] == "PASS",
        "equal_weight_joint_training_authorized": False,
        "decoder_adaptation_authorized": False,
        "physical_sensor_validated": False,
    }


def write_attribution_report(run_root: Path, protocol: dict, attribution: dict, safety: dict, gate: dict) -> None:
    lines = [
        "# DirectPose conflict-source attribution", "",
        f"Decision: **{gate['decision']}**. Exact fallback: **{safety['status']}**.", "",
        "This is a fixed training-split diagnostic. It does not validate a physical rope sensor or authorize equal-weight four-domain training or decoder adaptation.", "",
        "## Dataset operating point", "",
        "| dataset | WiLoR PA/root | current head PA/root | valid rope | token RMS |", "|---|---:|---:|---:|---:|",
    ]
    for name in DATASETS:
        row = attribution["dataset_stats"][name]
        lines.append(
            f"| {name} | {row['wilor_base_metrics'][0]:.3f}/{row['wilor_base_metrics'][1]:.3f} | "
            f"{row['current_head_metrics'][0]:.3f}/{row['current_head_metrics'][1]:.3f} | "
            f"{row['valid_fraction']:.3f} | {row['token_rms']:.3f} |"
        )
    lines += ["", "## Overall scenario matrices", ""]
    for scenario_name in attribution["scenario_order"]:
        lines += _markdown_ci_matrix(scenario_name, attribution["scenarios"][scenario_name]["cross_dataset"]["overall"])
    lines += ["## Parameter-layer matrices under full correct rope", ""]
    for group in PARAMETER_GROUPS:
        lines += _markdown_ci_matrix(group, attribution["scenarios"]["full_correct"]["parameter_groups"][group])
    lines += ["## Full-correct per-finger matrices", ""]
    for finger in FINGER_ORDER:
        lines += _markdown_ci_matrix(finger, attribution["scenarios"]["full_correct"]["cross_dataset"][finger])
    lines += ["## Rope-state matched diagnostic pairs", "", "| pair | scenario | cosine [95% CI] | rows | match distance mean/P95 |", "|---|---|---:|---:|---:|"]
    for pair, result in attribution["rope_state_matched_pairs"].items():
        for scenario, values in result["scenarios"].items():
            lines.append(
                f"| {pair} | {scenario} | {values['mean']['overall']:+.3f} "
                f"[{values['ci95_low']['overall']:+.3f},{values['ci95_high']['overall']:+.3f}] | "
                f"{result['rows']} | {result['match_distance_mean']:.3f}/{result['match_distance_p95']:.3f} |"
            )
    lines += ["", "## Predeclared interpretation", "", f"- HOT3D-DexYCB: **{gate['hot3d_dexycb_diagnosis']}**."]
    for pair, result in gate["diagnostic_pairs"].items():
        lines += [
            f"- {pair}: full conflict={result['significant_full_conflict']}; paired-rope contribution={result['paired_rope_input_contributes']}; "
            f"auxiliary-loss contribution={result['auxiliary_losses_contribute']}; persistent after task/state control={result['persistent_after_task_and_rope_state_control']}."
        ]
    lines += [
        "", "## Next bounded experiment", "",
        "Use HOT3D as the positive product-proxy target, ARCTIC as priority retention, HO3D/DexYCB as secondary retention, and InterHand only as stress evaluation. Keep WiLoR/MANO/decoder frozen and compare matched RGB-only, clean rope, and robust rope heads. Do not reinterpret these GT-derived ideal rope values as physical calibration evidence.", "",
    ]
    (run_root / "attribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def verify_attribution_run(run_root: Path) -> dict:
    required = (
        "protocol.json", "sample_manifests/summary.json", "safety_gate.json",
        "attribution_raw.npz", "attribution_summary.json", "summary.json", "attribution_report.md",
    )
    missing = [name for name in required if not (run_root / name).is_file()]
    checks = {"required_artifacts": not missing}
    errors = [] if not missing else [f"missing artifacts: {missing}"]
    if not missing:
        protocol = json.loads((run_root / "protocol.json").read_text())
        selection = json.loads((run_root / "sample_manifests/summary.json").read_text())
        safety = json.loads((run_root / "safety_gate.json").read_text())
        attribution = json.loads((run_root / "attribution_summary.json").read_text())
        summary = json.loads((run_root / "summary.json").read_text())
        checks["protocol_sha256"] = summary["protocol_sha256"] == sha256_file(run_root / "protocol.json")
        checks["immutable_inputs"] = summary["input_sha256"] == verify_input_hashes(protocol)
        checks["training_only"] = all(protocol["datasets"][name]["split_role"] == "training" for name in DATASETS)
        checks["episode_disjoint"] = all(not selection[name]["episode_overlap"] for name in DATASETS)
        checks["safety"] = safety["status"] == "PASS"
        checks["raw_hash"] = attribution["raw_sha256"] == sha256_file(run_root / "attribution_raw.npz")
        checks["scenario_order"] = attribution["scenario_order"] == list(protocol["attribution"]["scenarios"])
        with np.load(run_root / "attribution_raw.npz") as raw:
            expected = (
                len(protocol["attribution"]["scenarios"]), 1 + len(FINGER_ORDER),
                int(protocol["sampling"]["num_batches"]), len(DATASETS), len(DATASETS),
            )
            checks["matrix_shape"] = raw["cross_dataset_cosine"].shape == expected
        for name, passed in checks.items():
            if not passed:
                errors.append(f"failed check: {name}")
    result = {"status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors, "verified_at_unix": time.time()}
    write_json(run_root / "attribution_verification.json", result)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def verify_run(run_root: Path) -> dict:
    required = (
        "protocol.json", "sample_manifests/summary.json", "safety_gate.json",
        "gradient_raw.npz", "gradient_summary.json", "transfer_raw.npz",
        "transfer_summary.json", "summary.json", "report.md",
    )
    missing = [name for name in required if not (run_root / name).is_file()]
    checks = {"required_artifacts": not missing}
    errors = [] if not missing else [f"missing artifacts: {missing}"]
    if not missing:
        protocol = json.loads((run_root / "protocol.json").read_text())
        selection = json.loads((run_root / "sample_manifests/summary.json").read_text())
        safety = json.loads((run_root / "safety_gate.json").read_text())
        gradient = json.loads((run_root / "gradient_summary.json").read_text())
        transfer = json.loads((run_root / "transfer_summary.json").read_text())
        summary = json.loads((run_root / "summary.json").read_text())
        checks["protocol_sha256"] = summary["protocol_sha256"] == sha256_file(run_root / "protocol.json")
        checks["immutable_inputs"] = summary["input_sha256"] == verify_input_hashes(protocol)
        checks["training_only"] = all(protocol["datasets"][name]["split_role"] == "training" for name in DATASETS)
        checks["episode_disjoint"] = all(not selection[name]["episode_overlap"] for name in DATASETS)
        checks["sample_hash_disjoint"] = all(selection[name]["update_sample_id_sha256"] != selection[name]["probe_sample_id_sha256"] for name in DATASETS)
        checks["safety"] = safety["status"] == "PASS"
        checks["gradient_raw_hash"] = gradient["raw_sha256"] == sha256_file(run_root / "gradient_raw.npz")
        checks["transfer_raw_hash"] = transfer["raw_sha256"] == sha256_file(run_root / "transfer_raw.npz")
        checks["matrix_shapes"] = all(np.asarray(gradient["cross_dataset"][key]["mean"]).shape == (5, 5) for key in ("overall", *FINGER_ORDER))
        with np.load(run_root / "transfer_raw.npz") as raw:
            delta = raw["adamw_delta"]
            checks["transfer_reconstruction"] = bool(np.allclose(
                delta.mean(axis=0)[:, :, 0],
                np.asarray(transfer["modes"]["adamw"]["metrics"]["pa"]["delta"]),
                atol=1e-10,
            ))
        for name, passed in checks.items():
            if not passed:
                errors.append(f"failed check: {name}")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
        "verified_at_unix": time.time(),
    }
    write_json(run_root / "artifact_verification.json", result)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def render_report(run_root: Path) -> Path:
    protocol = json.loads((run_root / "protocol.json").read_text())
    selection = json.loads((run_root / "sample_manifests/summary.json").read_text())
    gradient = json.loads((run_root / "gradient_summary.json").read_text())
    transfer = json.loads((run_root / "transfer_summary.json").read_text())
    safety = json.loads((run_root / "safety_gate.json").read_text())
    summary = json.loads((run_root / "summary.json").read_text())
    write_report(run_root, protocol, selection, gradient, transfer, safety, summary["gate"])
    return run_root / "report.md"


def run_audit(protocol_path: Path, run_root: Path, device: str) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if tuple(protocol.get("dataset_order", ())) != DATASETS:
        raise ValueError(f"dataset_order must be {DATASETS}")
    input_hashes = verify_input_hashes(protocol)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "protocol.json").write_bytes(protocol_bytes)
    selected, selection = prepare_selected_data(protocol, run_root)
    checkpoint = Path(protocol["checkpoint"]["path"])
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != protocol["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash changed after protocol freeze")
    raw_model, model, checkpoint_config = load_checkpoint_models(checkpoint, device)
    if int(checkpoint_config["hidden_dim"]) != 128:
        raise ValueError("audit protocol requires current h128 checkpoint")
    safety = exact_fallback_gate(checkpoint, selected[DATASETS[0]]["arrays"], device)
    write_json(run_root / "safety_gate.json", safety)
    if safety["status"] != "PASS":
        raise ValueError("exact fallback safety gate failed")
    del raw_model
    gradient = run_gradient_audit(protocol, selected, model, device, run_root)
    transfer = run_transfer_audit(protocol, selected, model, device, run_root)
    gate = evaluate_conflict_gate(protocol, gradient, transfer, safety)
    summary = {
        "status": "completed",
        "decision": gate["decision"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint_sha256": checkpoint_hash,
        "input_sha256": input_hashes,
        "dataset_order": list(DATASETS),
        "gate": gate,
        "training_cells_started": False,
        "physical_sensor_validated": False,
    }
    write_json(run_root / "summary.json", summary)
    write_report(run_root, protocol, selection, gradient, transfer, safety, gate)
    return summary


def run_attribution(protocol_path: Path, run_root: Path, device: str) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if tuple(protocol.get("dataset_order", ())) != DATASETS:
        raise ValueError(f"dataset_order must be {DATASETS}")
    if not protocol.get("attribution"):
        raise ValueError("protocol is missing attribution configuration")
    input_hashes = verify_input_hashes(protocol)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "protocol.json").write_bytes(protocol_bytes)
    selected, _ = prepare_selected_data(protocol, run_root)
    checkpoint = Path(protocol["checkpoint"]["path"])
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != protocol["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash changed after protocol freeze")
    raw_model, model, checkpoint_config = load_checkpoint_models(checkpoint, device)
    if int(checkpoint_config["hidden_dim"]) != 128:
        raise ValueError("attribution protocol requires current h128 checkpoint")
    safety = exact_fallback_gate(checkpoint, selected[DATASETS[0]]["arrays"], device)
    write_json(run_root / "safety_gate.json", safety)
    if safety["status"] != "PASS":
        raise ValueError("exact fallback safety gate failed")
    del raw_model
    attribution = run_attribution_audit(protocol, selected, model, device, run_root)
    gate = evaluate_attribution(protocol, attribution, safety)
    summary = {
        "status": "completed", "decision": gate["decision"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "checkpoint_sha256": checkpoint_hash, "input_sha256": input_hashes,
        "dataset_order": list(DATASETS), "gate": gate,
        "training_started": False, "physical_sensor_validated": False,
    }
    write_json(run_root / "summary.json", summary)
    write_attribution_report(run_root, protocol, attribution, safety, gate)
    verify_attribution_run(run_root)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--run-root", type=Path, required=True)
    audit.add_argument("--device", default="cuda")
    verify = sub.add_parser("verify")
    verify.add_argument("--run-root", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--run-root", type=Path, required=True)
    attribution = sub.add_parser("attribute")
    attribution.add_argument("--protocol", type=Path, required=True)
    attribution.add_argument("--run-root", type=Path, required=True)
    attribution.add_argument("--device", default="cuda")
    verify_attribution = sub.add_parser("verify-attribution")
    verify_attribution.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "verify":
        return verify_run(args.run_root)
    if args.command == "report":
        return render_report(args.run_root)
    if args.command == "verify-attribution":
        return verify_attribution_run(args.run_root)
    if args.command == "attribute":
        return run_attribution(args.protocol, args.run_root, args.device)
    return run_audit(args.protocol, args.run_root, args.device)


if __name__ == "__main__":
    main()
