#!/usr/bin/env python3
"""Direct-GT rope-conditioned MANO hand-pose residual screen.

B uses base pose plus five rope measurements. C adds one cross-attention over
frozen spatial WiLoR tokens. Both predict a bounded 45D hand-pose residual and
train directly against GT 3D joints instead of optimizer alphas.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.actions import FINGER_POSE_GROUPS
from ropetrack.refine.analysis import perturb_rope_reading
from ropetrack.refine.cache import align_rows_by_sample_id
from ropetrack.rope import FINGER_CHAINS, FINGER_ORDER
from ropetrack.eval.pipeline import load_mano_j_regressor
from ropetrack.eval.protocols import DEXYCB_TIP_VERTEX_IDS, FREIHAND_JOINT_ORDER, canonical_dataset
from scripts.rope_refiner.apply_rope_refinement import (
    mano_layer,
    mano_predictions,
    torch_aa_to_rotmat,
    torch_rope_norm,
)


def finger_pose_dims() -> np.ndarray:
    return np.asarray(
        [[3 * joint + axis for joint in joints for axis in range(3)] for joints in FINGER_POSE_GROUPS],
        dtype=np.int64,
    )


class DirectPoseHead(nn.Module):
    def __init__(self, token_dim: int = 0, hidden_dim: int = 128, max_delta: float = 0.5) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        if self.token_dim < 0 or self.hidden_dim <= 0 or self.max_delta <= 0:
            raise ValueError("token_dim must be nonnegative; hidden_dim and max_delta must be positive")
        self.register_buffer("pose_dims", torch.from_numpy(finger_pose_dims()), persistent=False)
        self.query = nn.Sequential(nn.Linear(18, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        if self.token_dim:
            self.token_proj = nn.Sequential(nn.Linear(self.token_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        else:
            self.token_proj = None
            self.attention = None
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 9))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        base_pose: torch.Tensor,
        base_rope: torch.Tensor,
        input_rope: torch.Tensor,
        rope_valid: torch.Tensor,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_pose.ndim != 2 or base_pose.shape[1] != 45:
            raise ValueError(f"base_pose must be [B,45], got {tuple(base_pose.shape)}")
        batch = base_pose.shape[0]
        if any(tuple(x.shape) != (batch, 5) for x in (base_rope, input_rope, rope_valid)):
            raise ValueError("rope tensors must all be [B,5]")
        pose = base_pose[:, self.pose_dims]
        sensor = torch.stack((base_rope, input_rope, input_rope - base_rope, rope_valid), dim=-1)
        finger_id = torch.eye(5, device=base_pose.device, dtype=base_pose.dtype)[None].expand(batch, -1, -1)
        query = self.query(torch.cat((pose, sensor, finger_id), dim=-1))
        if self.token_dim:
            if tokens is None or tokens.ndim != 3 or tokens.shape[0] != batch or tokens.shape[2] != self.token_dim:
                raise ValueError(f"tokens must be [B,T,{self.token_dim}]")
            image = self.token_proj(tokens)
            attended, _ = self.attention(query, image, image, need_weights=False)
            query = query + attended
        elif tokens is not None:
            raise ValueError("tokens supplied to a rope+pose-only checkpoint")
        finger_delta = self.max_delta * torch.tanh(self.output(query))
        finger_delta = torch.where(
            rope_valid[:, :, None] > 0.5, finger_delta, torch.zeros_like(finger_delta)
        )
        index = self.pose_dims.reshape(1, -1).expand(batch, -1)
        delta = torch.zeros_like(base_pose).scatter(1, index, finger_delta.reshape(batch, -1))
        return base_pose + delta


def pa_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Differentiable-through-pred Procrustes alignment matching the scorer."""
    gt_mean = gt.mean(dim=1, keepdim=True)
    pred_mean = pred.mean(dim=1, keepdim=True)
    gt_center = gt - gt_mean
    pred_center = pred - pred_mean
    gt_scale = torch.linalg.norm(gt_center, dim=(1, 2), keepdim=True).clamp_min(1e-8)
    pred_scale = torch.linalg.norm(pred_center, dim=(1, 2), keepdim=True).clamp_min(1e-8)
    gt_norm = gt_center / gt_scale
    pred_norm = pred_center / pred_scale
    with torch.no_grad():
        u, singular, vh = torch.linalg.svd(gt_norm.transpose(1, 2) @ pred_norm)
        rotation = u @ vh
        scale = singular.sum(dim=1).reshape(-1, 1, 1)
    return pred_norm @ rotation.transpose(1, 2) * scale * gt_scale + gt_mean


def direct_losses(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_rope: torch.Tensor,
    input_rope: torch.Tensor,
    rope_valid: torch.Tensor,
    refined_pose: torch.Tensor,
    base_pose: torch.Tensor,
    *,
    root_weight: float,
    rope_weight: float,
    delta_weight: float,
) -> dict[str, torch.Tensor]:
    aligned = pa_align(pred_joints, gt_joints)
    pa = torch.nn.functional.l1_loss(aligned, gt_joints)
    pred_root = pred_joints - pred_joints[:, :1]
    gt_root = gt_joints - gt_joints[:, :1]
    root = torch.nn.functional.l1_loss(pred_root, gt_root)
    rope_denom = rope_valid.sum().clamp_min(1.0)
    rope = (((pred_rope - input_rope) * rope_valid) ** 2).sum() / rope_denom
    delta = ((refined_pose - base_pose) ** 2).mean()
    total = pa + root_weight * root + rope_weight * rope + delta_weight * delta
    pa_mpjpe_mm = torch.linalg.norm(aligned - gt_joints, dim=-1).mean() * 1000.0
    root_mpjpe_mm = torch.linalg.norm(pred_root - gt_root, dim=-1).mean() * 1000.0
    return {"loss": total, "pa": pa, "root": root, "rope": rope, "delta": delta,
            "pa_mpjpe_mm": pa_mpjpe_mm, "root_mpjpe_mm": root_mpjpe_mm}


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def load_gt(path: Path, run_meta: Path, sample_ids: np.ndarray) -> np.ndarray:
    rows = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)
    meta = json.loads(run_meta.read_text(encoding="utf-8"))
    order = meta["sample_order"]
    if rows.shape != (len(order), 21, 3):
        raise ValueError(f"GT shape/order mismatch: rows={rows.shape} order={len(order)}")
    return rows[align_rows_by_sample_id(sample_ids, order)]


def load_arrays(
    cache_path: Path,
    mano_cache_path: Path,
    gt_xyz_path: Path | None = None,
    run_meta_path: Path | None = None,
    feature_cache_path: Path | None = None,
) -> dict[str, np.ndarray]:
    cache = read_npz(cache_path)
    sample_ids = np.asarray(cache["sample_id"]).astype(str)
    mano = read_npz(mano_cache_path)
    perm = align_rows_by_sample_id(sample_ids, mano["sample_id"])
    arrays = {key: np.asarray(cache[key]) for key in (
        "sample_id", "base_hand_pose", "base_rope_norm", "input_rope_norm",
        "rope_valid", "rope_chain_m", "fist_ratio",
    )}
    is_right = mano["is_right"] if "is_right" in mano else np.ones(len(mano["sample_id"]), dtype=bool)
    arrays.update({
        "global_orient": np.asarray(mano["base_global_orient"], dtype=np.float32)[perm],
        "betas": np.asarray(mano["base_betas"], dtype=np.float32)[perm],
        "is_right": np.asarray(is_right, dtype=bool)[perm],
    })
    if gt_xyz_path is not None:
        if run_meta_path is None:
            run_meta_path = mano_cache_path.parent / "run_meta.json"
        arrays["gt_xyz"] = load_gt(gt_xyz_path, run_meta_path, sample_ids)
    if feature_cache_path is not None:
        features = read_npz(feature_cache_path)
        if "tokens" not in features:
            raise ValueError("feature cache has no tokens; extract with --save-tokens")
        arrays["tokens"] = np.asarray(features["tokens"])[
            align_rows_by_sample_id(sample_ids, features["sample_id"])
        ]
    return arrays


def append_bundles(arrays: dict[str, np.ndarray], paths: list[Path]) -> dict[str, np.ndarray]:
    """Append pre-aligned training arrays without adding a second data loader."""
    for path in paths:
        extra = read_npz(path)
        if set(extra) != set(arrays):
            missing = sorted(set(arrays) - set(extra))
            surplus = sorted(set(extra) - set(arrays))
            raise ValueError(f"bundle keys differ: missing={missing} surplus={surplus}")
        overlap = set(np.asarray(arrays["sample_id"]).astype(str)) & set(np.asarray(extra["sample_id"]).astype(str))
        if overlap:
            raise ValueError(f"bundle sample ids overlap: {sorted(overlap)[:5]}")
        for key in arrays:
            if arrays[key].shape[1:] != extra[key].shape[1:]:
                raise ValueError(f"bundle shape differs for {key}: {arrays[key].shape} vs {extra[key].shape}")
            arrays[key] = np.concatenate((arrays[key], extra[key]), axis=0)
    return arrays


def episode_split(
    sample_ids: np.ndarray,
    val_fraction: float,
    seed: int,
    episode_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    episodes = (
        np.asarray(episode_ids).astype(str)
        if episode_ids is not None
        else np.asarray([str(sid).replace("\\", "/").rsplit("/", 1)[0] for sid in sample_ids])
    )
    if episodes.shape != np.asarray(sample_ids).shape:
        raise ValueError("episode_ids must match sample_ids")
    unique = np.asarray(sorted(set(episodes.tolist())))
    if len(unique) < 2:
        raise ValueError("need at least two episodes for train/validation split")
    rng = np.random.default_rng(seed)
    unique = unique[rng.permutation(len(unique))]
    num_val = max(1, int(math.ceil(len(unique) * val_fraction)))
    val_names = unique[:num_val]
    val = np.flatnonzero(np.isin(episodes, val_names))
    train = np.flatnonzero(~np.isin(episodes, val_names))
    if not len(train) or not len(val):
        raise ValueError("episode split produced an empty side")
    return train, val


def load_episode_ids(path: Path, sample_ids: np.ndarray) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    by_id = {str(row["sample_id"]): str(row["episode_id"]) for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("episode manifest has duplicate sample_id values")
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in by_id]
    if missing:
        raise ValueError(f"episode manifest missing sample ids: {missing[:5]}")
    return np.asarray([by_id[str(sample_id)] for sample_id in sample_ids])


def sample_id_sha256(sample_ids) -> str:
    values = np.asarray(sample_ids).astype(str)
    return hashlib.sha256("\n".join(values.tolist()).encode()).hexdigest()


def training_provenance(args, arrays, train_idx, val_idx) -> dict:
    protocol = None
    protocol_sha256 = None
    if args.protocol_json is not None:
        raw = args.protocol_json.read_bytes()
        protocol = json.loads(raw)
        protocol_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "git_commit": commit,
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "sample_id_sha256": sample_id_sha256(arrays["sample_id"]),
        "train_sample_id_sha256": sample_id_sha256(arrays["sample_id"][train_idx]),
        "val_sample_id_sha256": sample_id_sha256(arrays["sample_id"][val_idx]),
        "inputs": {
            "cache": str(args.cache),
            "mano_cache": str(args.mano_cache),
            "gt_xyz": str(args.gt_xyz),
            "run_meta": str(args.run_meta) if args.run_meta is not None else None,
            "feature_cache": str(args.feature_cache) if args.feature_cache is not None else None,
            "extra_bundles": [str(path) for path in args.extra_bundle],
            "episode_manifest": str(args.episode_manifest) if getattr(args, "episode_manifest", None) is not None else None,
        },
    }


def apply_rope_mode(arrays: dict[str, np.ndarray], mode: str, seed: int, groups=()) -> None:
    if mode == "correct":
        return
    if mode == "zero":
        arrays["input_rope_norm"] = np.asarray(arrays["base_rope_norm"], dtype=np.float32).copy()
        arrays["rope_valid"] = np.zeros_like(arrays["rope_valid"], dtype=bool)
        return
    if mode != "shuffle":
        raise ValueError(f"unsupported rope mode: {mode}")
    rng = np.random.default_rng(seed)
    source = np.asarray(arrays["input_rope_norm"]).copy()
    valid = np.asarray(arrays["rope_valid"]).copy()
    if not groups:
        groups = (np.arange(len(source)),)
    for rows in groups:
        perm = rng.permutation(rows)
        arrays["input_rope_norm"][rows] = source[perm]
        arrays["rope_valid"][rows] = valid[perm]


def apply_sensor_perturbation(arrays: dict[str, np.ndarray], args) -> None:
    gain = float(args.rope_gain_fixed)
    if gain <= 0.0:
        raise ValueError("rope_gain_fixed must be positive")
    rope, valid = perturb_rope_reading(
        np.asarray(arrays["input_rope_norm"], dtype=np.float32) * gain,
        arrays["rope_valid"],
        args.rope_noise_std,
        args.rope_dropout,
        np.random.default_rng(args.seed),
        bias_std=args.rope_bias_std,
        bias_fixed=args.rope_bias_fixed,
        scale_range=args.rope_scale_range,
    )
    for finger in getattr(args, "rope_missing_finger", ()):
        finger_index = FINGER_ORDER.index(finger)
        valid[:, finger_index] = False
    arrays["input_rope_norm"], arrays["rope_valid"] = rope, valid


def tensor_batch(arrays: dict[str, np.ndarray], rows: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    keys = ("base_hand_pose", "base_rope_norm", "input_rope_norm", "rope_valid",
            "rope_chain_m", "fist_ratio", "global_orient", "betas", "is_right", "gt_xyz")
    batch = {}
    for key in keys:
        if key not in arrays:
            continue
        value = torch.from_numpy(np.asarray(arrays[key][rows]))
        batch[key] = value.to(device=device, dtype=torch.bool if key == "is_right" else torch.float32)
    if "tokens" in arrays:
        batch["tokens"] = torch.from_numpy(np.asarray(arrays["tokens"][rows], dtype=np.float32)).to(device)
    return batch


def decoded_batch(model, mano, batch, dataset: str = "freihand", j_regressor=None):
    refined = model(
        batch["base_hand_pose"], batch["base_rope_norm"], batch["input_rope_norm"],
        batch["rope_valid"], batch.get("tokens"),
    )
    output = mano(
        global_orient=torch_aa_to_rotmat(batch["global_orient"])[:, None],
        hand_pose=torch_aa_to_rotmat(refined.reshape(-1, 15, 3)),
        betas=batch["betas"], pose2rot=False,
    )
    joints = output.joints
    if canonical_dataset(dataset) == "dexycb":
        if j_regressor is None:
            raise ValueError("DexYCB training decode requires the MANO joint regressor")
        joints16 = torch.einsum("jv,bvc->bjc", j_regressor, output.vertices)
        tips = output.vertices[:, torch.as_tensor(DEXYCB_TIP_VERTEX_IDS, device=output.vertices.device)]
        native = torch.cat((joints16, tips), dim=1)
        joints = native[:, torch.as_tensor(FREIHAND_JOINT_ORDER, device=native.device)]
    mirror = torch.ones((len(joints), 1, 3), device=joints.device, dtype=joints.dtype)
    mirror[:, 0, 0] = torch.where(batch["is_right"], 1.0, -1.0)
    joints_eval = joints * mirror
    pred_rope = torch_rope_norm(joints, FINGER_CHAINS["freihand"], batch["rope_chain_m"], batch["fist_ratio"])
    return refined, joints_eval, pred_rope


def evaluate(model, mano, arrays, rows, batch_size, device, weights, dataset="freihand", j_regressor=None):
    model.eval()
    totals = {key: 0.0 for key in ("loss", "pa_mpjpe_mm", "root_mpjpe_mm", "rope")}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            batch = tensor_batch(arrays, selected, device)
            refined, joints, pred_rope = decoded_batch(model, mano, batch, dataset, j_regressor)
            losses = direct_losses(joints, batch["gt_xyz"], pred_rope, batch["input_rope_norm"],
                                   batch["rope_valid"], refined, batch["base_hand_pose"], **weights)
            for key in totals:
                totals[key] += float(losses[key]) * len(selected)
    return {key: value / len(rows) for key, value in totals.items()}


def train(args) -> Path:
    arrays = load_arrays(args.cache, args.mano_cache, args.gt_xyz, args.run_meta, args.feature_cache)
    arrays = append_bundles(arrays, args.extra_bundle)
    episode_ids = load_episode_ids(args.episode_manifest, arrays["sample_id"]) if args.episode_manifest else None
    train_idx, val_idx = episode_split(arrays["sample_id"], args.val_fraction, args.seed, episode_ids)
    apply_rope_mode(arrays, args.rope_mode, args.seed + 1, (train_idx, val_idx))
    token_dim = int(arrays["tokens"].shape[2]) if "tokens" in arrays else 0
    model = DirectPoseHead(token_dim, args.hidden_dim, args.max_delta).to(args.device)
    mano = mano_layer(args.device)
    mano.requires_grad_(False)
    j_regressor = None
    if canonical_dataset(args.dataset) == "dexycb":
        j_regressor = torch.from_numpy(load_mano_j_regressor(
            Path(__file__).resolve().parents[2] / "mano_data" / "MANO_RIGHT.pkl"
        )).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    weights = {"root_weight": args.root_weight, "rope_weight": args.rope_weight, "delta_weight": args.delta_weight}
    best, best_state, best_epoch, log = float("inf"), None, -1, []
    for epoch in range(args.max_epochs):
        model.train()
        order = rng.permutation(train_idx)
        train_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            batch = tensor_batch(arrays, selected, args.device)
            refined, joints, pred_rope = decoded_batch(model, mano, batch, args.dataset, j_regressor)
            losses = direct_losses(joints, batch["gt_xyz"], pred_rope, batch["input_rope_norm"],
                                   batch["rope_valid"], refined, batch["base_hand_pose"], **weights)
            optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(losses["loss"].detach()) * len(selected)
        metrics = evaluate(model, mano, arrays, val_idx, args.batch_size, args.device, weights, args.dataset, j_regressor)
        row = {"epoch": epoch, "train_loss": train_loss / len(train_idx), **{f"val_{k}": v for k, v in metrics.items()}}
        log.append(row)
        print(json.dumps(row), flush=True)
        if metrics["pa_mpjpe_mm"] < best - args.min_delta:
            best, best_epoch = metrics["pa_mpjpe_mm"], epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    config = {
        "token_dim": token_dim, "hidden_dim": args.hidden_dim, "max_delta": args.max_delta,
        "rope_mode": args.rope_mode, "weights": weights, "best_epoch": best_epoch,
        "best_val_pa_mpjpe_mm": best, "num_train": int(len(train_idx)), "num_val": int(len(val_idx)),
        "training_recipe": {
            "dataset": args.dataset, "seed": args.seed, "batch_size": args.batch_size,
            "optimizer": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs, "patience": args.patience, "min_delta": args.min_delta,
            "val_fraction": args.val_fraction, "checkpoint_selection": "minimum internal-validation PA-MPJPE",
        },
        "provenance": training_provenance(args, arrays, train_idx, val_idx),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, args.out_dir / "model.pt")
    (args.out_dir / "train_log.json").write_text(json.dumps({"config": config, "log": log}, indent=2), encoding="utf-8")
    return args.out_dir


def load_model(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    model = DirectPoseHead(config["token_dim"], config["hidden_dim"], config["max_delta"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def apply(args) -> Path:
    arrays = load_arrays(args.cache, args.mano_cache, feature_cache_path=args.feature_cache)
    apply_rope_mode(arrays, args.rope_mode, args.seed)
    apply_sensor_perturbation(arrays, args)
    model, config = load_model(args.checkpoint, args.device)
    if ("tokens" in arrays) != bool(config["token_dim"]):
        raise ValueError("feature-cache presence does not match checkpoint")
    refined_rows = []
    with torch.no_grad():
        for start in range(0, len(arrays["sample_id"]), args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, len(arrays["sample_id"])))
            batch = tensor_batch(arrays, rows, args.device)
            refined_rows.append(model(batch["base_hand_pose"], batch["base_rope_norm"], batch["input_rope_norm"],
                                      batch["rope_valid"], batch.get("tokens")).cpu().numpy())
    refined = np.concatenate(refined_rows).astype(np.float32)
    xyz, verts = mano_predictions(args.dataset, refined, arrays["sample_id"], args.mano_cache,
                                  args.device, args.batch_size, keep_vertices=not args.no_vertices)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pred.json").write_text(json.dumps([xyz, verts], separators=(",", ":")), encoding="utf-8")
    np.save(args.out_dir / "sample_id.npy", arrays["sample_id"])
    np.save(args.out_dir / "refined_hand_pose.npy", refined)
    summary = {"checkpoint": str(args.checkpoint), "rope_mode": args.rope_mode,
               "rope_sensor": {"noise_std": args.rope_noise_std, "dropout": args.rope_dropout,
                               "bias_std": args.rope_bias_std, "bias_fixed": args.rope_bias_fixed,
                               "scale_range": args.rope_scale_range, "gain_fixed": args.rope_gain_fixed,
                               "seed": args.seed},
               "num_samples": len(refined), "mean_abs_delta": float(np.mean(np.abs(refined - arrays["base_hand_pose"]))) }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return args.out_dir


def common(parser):
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--rope-mode", choices=("correct", "zero", "shuffle"), default="correct")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train")
    common(train_p)
    train_p.add_argument("--gt-xyz", type=Path, required=True)
    train_p.add_argument("--run-meta", type=Path, default=None)
    train_p.add_argument("--extra-bundle", type=Path, action="append", default=[])
    train_p.add_argument("--dataset", default="freihand")
    train_p.add_argument("--episode-manifest", type=Path, default=None)
    train_p.add_argument("--protocol-json", type=Path, default=None)
    train_p.add_argument("--out-dir", type=Path, required=True)
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--max-delta", type=float, default=0.5)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--max-epochs", type=int, default=80)
    train_p.add_argument("--patience", type=int, default=10)
    train_p.add_argument("--min-delta", type=float, default=0.01)
    train_p.add_argument("--val-fraction", type=float, default=0.1)
    train_p.add_argument("--root-weight", type=float, default=0.1)
    train_p.add_argument("--rope-weight", type=float, default=0.1)
    train_p.add_argument("--delta-weight", type=float, default=1e-3)
    apply_p = sub.add_parser("apply")
    common(apply_p)
    apply_p.add_argument("--checkpoint", type=Path, required=True)
    apply_p.add_argument("--dataset", required=True)
    apply_p.add_argument("--out-dir", type=Path, required=True)
    apply_p.add_argument("--rope-noise-std", type=float, default=0.0)
    apply_p.add_argument("--rope-dropout", type=float, default=0.0)
    apply_p.add_argument("--rope-bias-std", type=float, default=0.0)
    apply_p.add_argument("--rope-bias-fixed", type=float, default=0.0)
    apply_p.add_argument("--rope-scale-range", type=float, default=0.0)
    apply_p.add_argument("--rope-gain-fixed", type=float, default=1.0)
    apply_p.add_argument(
        "--rope-missing-finger", action="append", choices=FINGER_ORDER, default=[],
        help="Mark one named rope channel invalid; repeat to invalidate multiple fingers.",
    )
    apply_p.add_argument("--no-vertices", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return train(args) if args.command == "train" else apply(args)


if __name__ == "__main__":
    main()
