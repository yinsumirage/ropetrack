"""Training-only rope observability and no-training candidate reranking audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ropetrack.refine.apply import mano_layer, torch_aa_to_rotmat, torch_rope_norm
from ropetrack.refine.direct_pose import decoded_batch, load_arrays, pa_align, tensor_batch
from ropetrack.refine.direct_pose_audit import (
    DATASETS,
    _identity_rows,
    _load_dataset_arrays,
    _load_manifest,
    select_update_probe_batches,
    verify_input_hashes,
)
from ropetrack.rope import FINGER_CHAINS, FINGER_ORDER


REPRESENTATIONS = {
    "rope": ("rope",),
    "base": ("base",),
    "token": ("token",),
    "base_rope": ("base", "rope"),
    "token_base": ("token", "base"),
    "token_base_rope": ("token", "base", "rope"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_external_hashes(protocol: dict) -> dict[str, dict[str, str]]:
    verified = {}
    for name, spec in protocol["external"].items():
        paths = {"episode_manifest": spec["episode_manifest"], **spec["source"]}
        expected = spec["sha256"]
        if set(expected) != set(paths):
            raise ValueError(f"{name} external hash keys do not match declared inputs")
        verified[name] = {}
        for key, value in paths.items():
            actual = sha256_file(Path(value))
            if actual != expected[key]:
                raise ValueError(f"{name} external {key} hash changed after protocol freeze")
            verified[name][key] = actual
    return verified


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def project_tokens(tokens: np.ndarray, output_channels: int, seed: int, device: str) -> np.ndarray:
    """Fixed random channel projection; preserves the 4x3 spatial token layout."""
    values = np.asarray(tokens)
    if values.ndim != 3:
        raise ValueError(f"tokens must be [N,T,C], got {values.shape}")
    rng = np.random.default_rng(seed)
    projection = rng.choice((-1.0, 1.0), size=(values.shape[2], output_channels)).astype(np.float32)
    projection /= math.sqrt(output_channels)
    matrix = torch.from_numpy(projection).to(device)
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            batch = torch.from_numpy(values[start:start + 1024].astype(np.float32)).to(device)
            rows.append((batch @ matrix).flatten(1).cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def balanced_rows(episodes: np.ndarray, pool: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic episode round-robin sampling."""
    rng = np.random.default_rng(seed)
    grouped = {}
    for row in pool:
        grouped.setdefault(str(episodes[row]), []).append(int(row))
    names = np.asarray(sorted(grouped))
    names = names[rng.permutation(len(names))]
    for name in names:
        values = np.asarray(grouped[str(name)])
        grouped[str(name)] = values[rng.permutation(len(values))].tolist()
    selected = []
    while len(selected) < min(count, len(pool)):
        advanced = False
        for name in names:
            values = grouped[str(name)]
            if values:
                selected.append(values.pop())
                advanced = True
                if len(selected) == min(count, len(pool)):
                    break
        if not advanced:
            break
    return np.asarray(selected, dtype=np.int64)


def _copy_rows(arrays: dict[str, np.ndarray], rows: np.ndarray, token_channels: int, seed: int, device: str):
    return {
        "sample_id": np.asarray(arrays["sample_id"])[rows].astype(str),
        "rope": np.asarray(arrays["input_rope_norm"], dtype=np.float32)[rows].copy(),
        "base": np.asarray(arrays["base_hand_pose"], dtype=np.float32)[rows].copy(),
        "token": project_tokens(arrays["tokens"][rows], token_channels, seed, device),
        "gt_xyz": np.asarray(arrays["gt_xyz"], dtype=np.float32)[rows].copy(),
    }


def jacobian_summary(arrays: dict[str, np.ndarray], rows: np.ndarray, mano, device: str) -> dict:
    singular_rows = []
    for start in range(0, len(rows), 32):
        selected = rows[start:start + 32]
        pose = torch.from_numpy(
            np.asarray(arrays["base_hand_pose"][selected], dtype=np.float32)
        ).to(device).requires_grad_(True)
        orient = torch.from_numpy(
            np.asarray(arrays["global_orient"][selected], dtype=np.float32)
        ).to(device)
        betas = torch.from_numpy(np.asarray(arrays["betas"][selected], dtype=np.float32)).to(device)
        chain = torch.from_numpy(
            np.asarray(arrays["rope_chain_m"][selected], dtype=np.float32)
        ).to(device)
        fist = torch.from_numpy(
            np.asarray(arrays["fist_ratio"][selected], dtype=np.float32)
        ).to(device)
        output = mano(
            global_orient=torch_aa_to_rotmat(orient)[:, None],
            hand_pose=torch_aa_to_rotmat(pose.reshape(-1, 15, 3)),
            betas=betas,
            pose2rot=False,
        )
        rope = torch_rope_norm(output.joints, FINGER_CHAINS["freihand"], chain, fist)
        gradients = [
            torch.autograd.grad(rope[:, finger].sum(), pose, retain_graph=True)[0]
            for finger in range(5)
        ]
        jacobian = torch.stack(gradients, dim=1)
        singular_rows.append(torch.linalg.svdvals(jacobian).detach().cpu().numpy())
    singular = np.concatenate(singular_rows)
    rank = (singular > singular[:, :1] * 1e-4).sum(axis=1)
    return {
        "samples": int(len(rows)),
        "median_singular_values": np.median(singular, axis=0).tolist(),
        "rank_counts": {
            str(value): int(np.sum(rank == value)) for value in sorted(set(rank.tolist()))
        },
        "median_rank": float(np.median(rank)),
        "median_local_null_dimension_lower_bound": float(45 - np.median(rank)),
    }


def load_training_banks(protocol: dict, device: str):
    parent = json.loads(Path(protocol["parent_protocol"]).read_text(encoding="utf-8"))
    verified = verify_input_hashes(parent)
    mano = mano_layer(device)
    mano.requires_grad_(False)
    banks, queries, selections, jacobians = {}, {}, {}, {}
    sampling = protocol["sampling"]
    for dataset_index, name in enumerate(DATASETS):
        spec = parent["datasets"][name]
        arrays = _load_dataset_arrays(spec)
        ids = np.asarray(arrays["sample_id"]).astype(str)
        manifest = _load_manifest(Path(spec["episode_manifest"]))
        episodes, subjects, splits = _identity_rows(name, ids, manifest)
        if {"test", "val", "validation", "evaluation"} & {value.lower() for value in splits}:
            raise ValueError(f"{name} is not training-only")
        valid = np.asarray(arrays["rope_valid"], dtype=bool).all(axis=1)
        valid &= np.isfinite(arrays["input_rope_norm"]).all(axis=1)
        valid_rows = np.flatnonzero(valid)
        ref_batch, query_batch = select_update_probe_batches(
            ids[valid_rows],
            episodes[valid_rows],
            batch_size=int(sampling["query_rows"]),
            num_batches=int(sampling["reference_rows"]) // int(sampling["query_rows"]),
            seed=int(protocol["seed"]) + dataset_index,
        )
        reference_rows = valid_rows[ref_batch.reshape(-1)]
        query_rows = valid_rows[query_batch[0]]
        bank = _copy_rows(
            arrays, reference_rows, int(protocol["token_projection"]["channels"]),
            int(protocol["token_projection"]["seed"]), device,
        )
        query = _copy_rows(
            arrays, query_rows, int(protocol["token_projection"]["channels"]),
            int(protocol["token_projection"]["seed"]), device,
        )
        bank["episode"] = episodes[reference_rows].astype(str)
        bank["subject"] = subjects[reference_rows].astype(str)
        query["episode"] = episodes[query_rows].astype(str)
        query["subject"] = subjects[query_rows].astype(str)
        banks[name], queries[name] = bank, query
        reference_episodes = set(bank["episode"].tolist())
        query_episodes = set(query["episode"].tolist())
        selections[name] = {
            "source_rows": int(len(ids)),
            "all_valid_rows": int(len(valid_rows)),
            "reference_rows": int(len(reference_rows)),
            "query_rows": int(len(query_rows)),
            "episode_overlap": len(reference_episodes & query_episodes),
            "reference_sample_sha256": hashlib.sha256(
                "\n".join(bank["sample_id"]).encode()
            ).hexdigest(),
            "query_sample_sha256": hashlib.sha256(
                "\n".join(query["sample_id"]).encode()
            ).hexdigest(),
        }
        jacobians[name] = jacobian_summary(
            arrays, query_rows[: int(protocol["jacobian_samples"])], mano, device
        )
        del arrays
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return banks, queries, selections, jacobians, verified, mano


def concatenate_banks(values: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([value[key] for value in values])
        for key in values[0]
    }


def representation(query: dict, reference: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    query_parts, reference_parts = [], []
    blocks = REPRESENTATIONS[name]
    for block in blocks:
        ref = np.asarray(reference[block], dtype=np.float32)
        qry = np.asarray(query[block], dtype=np.float32)
        mean = ref.mean(axis=0)
        scale = ref.std(axis=0)
        scale[scale < 1e-6] = 1.0
        reference_parts.append((ref - mean) / scale / math.sqrt(ref.shape[1]))
        query_parts.append((qry - mean) / scale / math.sqrt(ref.shape[1]))
    divisor = math.sqrt(len(blocks))
    return (
        np.concatenate(query_parts, axis=1) / divisor,
        np.concatenate(reference_parts, axis=1) / divisor,
    )


def topk_neighbors(query: np.ndarray, reference: np.ndarray, k: int, device: str):
    ref = torch.from_numpy(np.asarray(reference, dtype=np.float32)).to(device)
    ref_norm = (ref * ref).sum(dim=1)[None]
    distances, indices = [], []
    with torch.no_grad():
        for start in range(0, len(query), 128):
            qry = torch.from_numpy(np.asarray(query[start:start + 128], dtype=np.float32)).to(device)
            distance = (qry * qry).sum(dim=1)[:, None] + ref_norm - 2.0 * qry @ ref.T
            values, rows = torch.topk(distance.clamp_min_(0.0), k, largest=False)
            distances.append(values.cpu().numpy())
            indices.append(rows.cpu().numpy())
    return np.concatenate(distances), np.concatenate(indices)


def pair_errors(query_xyz: np.ndarray, reference_xyz: np.ndarray, indices: np.ndarray, device: str):
    candidates = reference_xyz[indices]
    targets = np.repeat(query_xyz[:, None], indices.shape[1], axis=1)
    flat_candidates = candidates.reshape(-1, 21, 3)
    flat_targets = targets.reshape(-1, 21, 3)
    pa_rows, root_rows, finger_rows = [], [], []
    finger_ids = [np.asarray(chain[1:], dtype=np.int64) for chain in FINGER_CHAINS["freihand"]]
    with torch.no_grad():
        for start in range(0, len(flat_candidates), 4096):
            candidate = torch.from_numpy(flat_candidates[start:start + 4096]).to(device)
            target = torch.from_numpy(flat_targets[start:start + 4096]).to(device)
            aligned = pa_align(candidate, target)
            point = torch.linalg.norm(aligned - target, dim=-1) * 1000.0
            pa_rows.append(point.mean(dim=1).cpu().numpy())
            candidate_root = candidate - candidate[:, :1]
            target_root = target - target[:, :1]
            root_rows.append(
                (torch.linalg.norm(candidate_root - target_root, dim=-1) * 1000.0)
                .mean(dim=1).cpu().numpy()
            )
            finger_rows.append(torch.stack(
                [point[:, torch.as_tensor(ids, device=device)].mean(dim=1) for ids in finger_ids],
                dim=1,
            ).cpu().numpy())
    shape = indices.shape
    return {
        "pa": np.concatenate(pa_rows).reshape(shape),
        "root": np.concatenate(root_rows).reshape(shape),
        "finger": np.concatenate(finger_rows).reshape(*shape, 5),
    }


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def bootstrap_delta(candidate, reference, groups, replicates: int, seed: int):
    delta = np.asarray(candidate) - np.asarray(reference)
    groups = np.asarray(groups).astype(str)
    unique = np.asarray(sorted(set(groups.tolist())))
    by_group = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.integers(0, len(unique), len(unique))
        rows = np.concatenate([by_group[index] for index in selected])
        values.append(float(np.mean(delta[rows])))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate_query_set(
    query: dict,
    reference: dict,
    protocol: dict,
    device: str,
    raw: dict[str, np.ndarray],
    prefix: str,
) -> dict:
    k = int(protocol["candidate_k"])
    result = {"representations": {}}
    cached = {}
    for name in REPRESENTATIONS:
        query_rep, reference_rep = representation(query, reference, name)
        distance, indices = topk_neighbors(query_rep, reference_rep, k, device)
        errors = pair_errors(query["gt_xyz"], reference["gt_xyz"], indices, device)
        cached[name] = (distance, indices, errors)
        result["representations"][name] = {
            "rank1_pa_mm": summarize(errors["pa"][:, 0]),
            "topk_oracle_pa_mm": summarize(errors["pa"].min(axis=1)),
            "rank1_root_mm": summarize(errors["root"][:, 0]),
            "topk_oracle_root_mm": summarize(errors["root"].min(axis=1)),
            "rank1_finger_pa_mm": {
                finger: summarize(errors["finger"][:, 0, finger_index])
                for finger_index, finger in enumerate(FINGER_ORDER)
            },
            "topk_oracle_finger_pa_mm": {
                finger: summarize(errors["finger"][:, :, finger_index].min(axis=1))
                for finger_index, finger in enumerate(FINGER_ORDER)
            },
        }
        raw[f"{prefix}__{name}__rank1_pa"] = errors["pa"][:, 0].astype(np.float32)
        raw[f"{prefix}__{name}__oracle_pa"] = errors["pa"].min(axis=1).astype(np.float32)

    _, visual_indices, visual_errors = cached["token_base"]
    rope_distance = np.mean(
        (query["rope"][:, None] - reference["rope"][visual_indices]) ** 2, axis=2
    )
    selected = rope_distance.argmin(axis=1)
    rows = np.arange(len(query["sample_id"]))
    visual = visual_errors["pa"][:, 0]
    reranked = visual_errors["pa"][rows, selected]
    oracle = visual_errors["pa"].min(axis=1)
    delta = reranked - visual
    visual_root = visual_errors["root"][:, 0]
    reranked_root = visual_errors["root"][rows, selected]
    visual_finger = visual_errors["finger"][:, 0]
    reranked_finger = visual_errors["finger"][rows, selected]
    result["visual_candidate_reranking"] = {
        "visual_rank1_pa_mm": summarize(visual),
        "rope_reranked_pa_mm": summarize(reranked),
        "visual_topk_oracle_pa_mm": summarize(oracle),
        "rerank_minus_visual_mean_mm": float(np.mean(delta)),
        "rerank_minus_visual_episode_bootstrap_ci": bootstrap_delta(
            reranked, visual, query["episode"],
            int(protocol["bootstrap"]["replicates"]),
            int(protocol["bootstrap"]["seed"]),
        ),
        "oracle_gap_recovered_fraction": float(
            np.mean(visual - reranked) / max(float(np.mean(visual - oracle)), 1e-9)
        ),
        "root": {
            "visual_rank1_mm": summarize(visual_root),
            "rope_reranked_mm": summarize(reranked_root),
            "rerank_minus_visual_mean_mm": float(np.mean(reranked_root - visual_root)),
        },
        "finger_pa": {
            finger: {
                "visual_rank1_mm": summarize(visual_finger[:, finger_index]),
                "rope_reranked_mm": summarize(reranked_finger[:, finger_index]),
                "rerank_minus_visual_mean_mm": float(
                    np.mean(reranked_finger[:, finger_index] - visual_finger[:, finger_index])
                ),
            }
            for finger_index, finger in enumerate(FINGER_ORDER)
        },
    }
    raw[f"{prefix}__visual_pa"] = visual.astype(np.float32)
    raw[f"{prefix}__reranked_pa"] = reranked.astype(np.float32)
    raw[f"{prefix}__visual_oracle_pa"] = oracle.astype(np.float32)
    raw[f"{prefix}__visual_root"] = visual_root.astype(np.float32)
    raw[f"{prefix}__reranked_root"] = reranked_root.astype(np.float32)
    raw[f"{prefix}__visual_finger_pa"] = visual_finger.astype(np.float32)
    raw[f"{prefix}__reranked_finger_pa"] = reranked_finger.astype(np.float32)

    base_rank1 = cached["base"][2]["pa"][:, 0]
    token_base_rank1 = cached["token_base"][2]["pa"][:, 0]
    result["token_base_minus_base"] = {
        "mean_mm": float(np.mean(token_base_rank1 - base_rank1)),
        "episode_bootstrap_ci": bootstrap_delta(
            token_base_rank1,
            base_rank1,
            query["episode"],
            int(protocol["bootstrap"]["replicates"]),
            int(protocol["bootstrap"]["seed"]),
        ),
    }

    rope_dist, _, rope_errors = cached["rope"]
    nearest_distance = np.sqrt(rope_dist[:, 0])
    result["rope_collision"] = {
        "nearest_standardized_distance": summarize(nearest_distance),
        "nearest_pose_pa_mm": summarize(rope_errors["pa"][:, 0]),
        "thresholds": {},
    }
    raw[f"{prefix}__rope_nearest_distance"] = nearest_distance.astype(np.float32)
    raw[f"{prefix}__rope_nearest_pa"] = rope_errors["pa"][:, 0].astype(np.float32)
    raw_rope_distance = np.sqrt(np.mean(
        (query["rope"][:, None] - reference["rope"][cached["rope"][1]]) ** 2, axis=2
    ))
    for threshold in protocol["rope_collision_thresholds"]:
        mask = raw_rope_distance[:, 0] <= float(threshold)
        result["rope_collision"]["thresholds"][str(threshold)] = {
            "query_fraction_with_neighbor": float(mask.mean()),
            "nearest_pose_pa_mm": summarize(rope_errors["pa"][mask, 0]) if mask.any() else None,
        }
    return result


def pairwise_training_summary(
    banks: dict,
    queries: dict,
    training: dict,
    protocol: dict,
    device: str,
) -> dict:
    names = ("rope", "base", "token_base")
    result = {
        representation_name: {
            metric: {target: {} for target in DATASETS}
            for metric in ("rank1_pa_mm", "topk_oracle_pa_mm")
        }
        for representation_name in names
    }
    k = int(protocol["candidate_k"])
    for target in DATASETS:
        for source in DATASETS:
            if target == source:
                rows = training[target]["same_domain_reference"]["representations"]
                for representation_name in names:
                    for metric in ("rank1_pa_mm", "topk_oracle_pa_mm"):
                        result[representation_name][metric][target][source] = rows[
                            representation_name
                        ][metric]["mean"]
                continue
            for representation_name in names:
                query_rep, reference_rep = representation(
                    queries[target], banks[source], representation_name
                )
                _, indices = topk_neighbors(query_rep, reference_rep, k, device)
                errors = pair_errors(
                    queries[target]["gt_xyz"], banks[source]["gt_xyz"], indices, device
                )
                result[representation_name]["rank1_pa_mm"][target][source] = float(
                    np.mean(errors["pa"][:, 0])
                )
                result[representation_name]["topk_oracle_pa_mm"][target][source] = float(
                    np.mean(errors["pa"].min(axis=1))
                )
    return result


class _IdentityHead:
    def __call__(self, base_pose, base_rope, input_rope, rope_valid, tokens=None):
        return base_pose


def decode_base_xyz(arrays: dict[str, np.ndarray], rows: np.ndarray, dataset: str, mano, device: str):
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 256):
            batch = tensor_batch(arrays, rows[start:start + 256], device)
            _, joints, _ = decoded_batch(_IdentityHead(), mano, batch, dataset=dataset)
            output.append(joints.cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def load_external_query(spec: dict, protocol: dict, mano, device: str, seed: int):
    arrays = load_arrays(
        Path(spec["source"]["cache"]),
        Path(spec["source"]["mano_cache"]),
        Path(spec["source"]["gt_xyz"]),
        Path(spec["source"]["run_meta"]),
        Path(spec["source"]["feature_cache"]),
    )
    ids = np.asarray(arrays["sample_id"]).astype(str)
    manifest = _load_manifest(Path(spec["episode_manifest"]))
    episodes, subjects, _ = _identity_rows(spec["decoder_dataset"], ids, manifest)
    valid = np.asarray(arrays["rope_valid"], dtype=bool).all(axis=1)
    valid &= np.isfinite(arrays["input_rope_norm"]).all(axis=1)
    if spec.get("subjects"):
        valid &= np.isin(subjects, np.asarray(spec["subjects"]))
    pool = np.flatnonzero(valid)
    rows = balanced_rows(episodes, pool, int(protocol["external_query_rows"]), seed)
    query = _copy_rows(
        arrays, rows, int(protocol["token_projection"]["channels"]),
        int(protocol["token_projection"]["seed"]), device,
    )
    query["episode"] = episodes[rows].astype(str)
    query["subject"] = subjects[rows].astype(str)
    query["base_xyz"] = decode_base_xyz(
        arrays, rows, spec["decoder_dataset"], mano, device
    )
    query["phase"] = np.asarray([
        str(manifest.get(sample_id, {}).get("phase", "all")) for sample_id in query["sample_id"]
    ])
    del arrays
    gc.collect()
    return query


def base_pa_error(query: dict, device: str) -> np.ndarray:
    rows = np.arange(len(query["sample_id"]))[:, None]
    return pair_errors(query["gt_xyz"], query["base_xyz"], rows, device)["pa"][:, 0]


def markdown_report(summary: dict) -> str:
    lines = [
        "# Five-rope observability and candidate-reranking audit",
        "",
        f"Decision: **{summary['decision']}**.",
        "",
        "This is a retrieval/observability diagnostic, not a trained or deployable gate.",
        "",
        "## External anchors",
        "",
        "| dataset/slice | WiLoR base | visual retrieval | rope reranked | delta | 95% CI | visual top-k oracle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summary["external"].items():
        for slice_name, row in result["slices"].items():
            rerank = row["visual_candidate_reranking"]
            lines.append(
                f"| {name}/{slice_name} | {row['base_pa_mm']['mean']:.3f} | "
                f"{rerank['visual_rank1_pa_mm']['mean']:.3f} | "
                f"{rerank['rope_reranked_pa_mm']['mean']:.3f} | "
                f"{rerank['rerank_minus_visual_mean_mm']:+.3f} | "
                f"[{rerank['rerank_minus_visual_episode_bootstrap_ci'][0]:+.3f},"
                f"{rerank['rerank_minus_visual_episode_bootstrap_ci'][1]:+.3f}] | "
                f"{rerank['visual_topk_oracle_pa_mm']['mean']:.3f} |"
            )
    lines += [
        "",
        "## Jacobian",
        "",
        "| dataset | median rank | local null dimension lower bound |",
        "|---|---:|---:|",
    ]
    for name, value in summary["jacobian"].items():
        lines.append(
            f"| {name} | {value['median_rank']:.1f} | "
            f"{value['median_local_null_dimension_lower_bound']:.1f} |"
        )
    lines += [
        "",
        summary["research_boundary"],
        "",
    ]
    return "\n".join(lines)


def run(protocol_path: Path, out: Path, device: str) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_results"):
        raise ValueError("protocol must be frozen before results")
    out.mkdir(parents=True, exist_ok=True)
    banks, queries, selections, jacobians, verified, mano = load_training_banks(protocol, device)
    verified_external = verify_external_hashes(protocol)
    raw = {}
    sample_manifest = {
        "training": {
            name: {
                role: [
                    {
                        "sample_id": str(sample_id),
                        "episode": str(episode),
                        "subject": str(subject),
                    }
                    for sample_id, episode, subject in zip(
                        values["sample_id"], values["episode"], values["subject"]
                    )
                ]
                for role, values in (("reference", banks[name]), ("query", queries[name]))
            }
            for name in DATASETS
        },
        "external": {},
    }
    training = {}
    for name in DATASETS:
        same = evaluate_query_set(
            queries[name], banks[name], protocol, device, raw, f"train_{name}_same"
        )
        others = concatenate_banks([banks[value] for value in DATASETS if value != name])
        cross = evaluate_query_set(
            queries[name], others, protocol, device, raw, f"train_{name}_other"
        )
        training[name] = {"same_domain_reference": same, "other_domain_reference": cross}
    pairwise_training = pairwise_training_summary(
        banks, queries, training, protocol, device
    )

    external = {}
    for index, (name, spec) in enumerate(protocol["external"].items()):
        query = load_external_query(spec, protocol, mano, device, int(protocol["seed"]) + 100 + index)
        overlap = set(query["sample_id"].tolist()) & set(
            banks[spec["reference_bank"]]["sample_id"].tolist()
        )
        if overlap:
            raise ValueError(f"{name} external query overlaps its training reference bank")
        sample_manifest["external"][name] = [
            {
                "sample_id": str(sample_id),
                "episode": str(episode),
                "subject": str(subject),
                "phase": str(phase),
            }
            for sample_id, episode, subject, phase in zip(
                query["sample_id"], query["episode"], query["subject"], query["phase"]
            )
        ]
        result = evaluate_query_set(query, banks[spec["reference_bank"]], protocol, device, raw, f"external_{name}")
        base = base_pa_error(query, device)
        result["base_pa_mm"] = summarize(base)
        raw[f"external_{name}__base_pa"] = base.astype(np.float32)
        slices = {"all": result}
        for phase in sorted(set(query["phase"].tolist()) - {"all"}):
            mask = query["phase"] == phase
            if mask.sum() < 10:
                continue
            sliced_query = {
                key: value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value
                for key, value in query.items()
            }
            sliced = evaluate_query_set(
                sliced_query, banks[spec["reference_bank"]], protocol, device, raw,
                f"external_{name}_{phase}",
            )
            sliced_base = base[mask]
            sliced["base_pa_mm"] = summarize(sliced_base)
            slices[phase] = sliced
        external[name] = {
            "rows": int(len(query["sample_id"])),
            "sample_id_sha256": hashlib.sha256("\n".join(query["sample_id"]).encode()).hexdigest(),
            "training_reference_overlap": 0,
            "slices": slices,
        }

    hot = external["hot3d"]["slices"]
    low = hot.get("low_visibility", hot["all"])
    context = hot.get("context")
    low_token_delta = low["token_base_minus_base"]
    context_token_delta = None if context is None else context["token_base_minus_base"]
    rerank = low["visual_candidate_reranking"]
    arctic_rerank = external["arctic"]["slices"]["all"]["visual_candidate_reranking"]
    gates = {
        "hot_low_visibility_token_corruption": bool(
            low_token_delta["mean_mm"]
            >= protocol["gates"]["token_minus_base_low_visibility_min_mm"]
            and low_token_delta["episode_bootstrap_ci"][0] > 0.0
            and (
                context_token_delta is None
                or context_token_delta["mean_mm"]
                <= protocol["gates"]["token_minus_base_context_max_mm"]
            )
        ),
        "hot_low_visibility_rope_rerank": bool(
            rerank["rerank_minus_visual_mean_mm"]
            <= -protocol["gates"]["rerank_improvement_min_mm"]
            and rerank["rerank_minus_visual_episode_bootstrap_ci"][1] < 0.0
        ),
        "arctic_rerank_nonregression": bool(
            arctic_rerank["rerank_minus_visual_mean_mm"]
            <= protocol["gates"]["arctic_rerank_regression_max_mm"]
        ),
    }
    if gates["hot_low_visibility_token_corruption"]:
        decision = "TOKEN_ADAPTATION_GATE_SUPPORTED"
    elif gates["hot_low_visibility_rope_rerank"] and gates["arctic_rerank_nonregression"]:
        decision = "CONTINUE_GEOMETRIC_POSTERIOR"
    else:
        decision = "STOP_ARCHITECTURE_CHANGE_NOT_PROVEN"

    raw_path = out / "raw.npz"
    np.savez_compressed(raw_path, **raw)
    sample_manifest_path = out / "sample_manifest.json"
    write_json(sample_manifest_path, sample_manifest)
    summary = {
        "decision": decision,
        "gates": gates,
        "diagnostics": {
            "hot_low_visibility_token_minus_base": low_token_delta,
            "hot_context_token_minus_base": context_token_delta,
        },
        "selection": selections,
        "jacobian": jacobians,
        "training": training,
        "pairwise_training": pairwise_training,
        "external": external,
        "verified_inputs": {"training": verified, "external": verified_external},
        "protocol_sha256": sha256_file(protocol_path),
        "raw_sha256": sha256_file(raw_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "research_boundary": protocol["research_boundary"],
    }
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    verification = {
        "status": "PASS" if (
            all(value["episode_overlap"] == 0 for value in selections.values())
            and all(value["training_reference_overlap"] == 0 for value in external.values())
            and sha256_file(raw_path) == summary["raw_sha256"]
            and sha256_file(sample_manifest_path) == summary["sample_manifest_sha256"]
        ) else "FAIL",
        "protocol_sha256": summary["protocol_sha256"],
        "raw_sha256": summary["raw_sha256"],
        "sample_manifest_sha256": summary["sample_manifest_sha256"],
    }
    write_json(out / "verification.json", verification)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run(args.protocol, args.out, args.device)
    print(json.dumps({"decision": summary["decision"], "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
