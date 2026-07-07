#!/usr/bin/env python3
"""Distill the test-time rope optimizer into a one-pass alpha student (P2).

Teacher data comes from apply_rope_refinement.py --mode optimize run on the
TRAINING split with the frozen winner recipe: its out-dir provides
refiner_eval_cache.npz (student inputs) and alpha.npy (targets), plus
flex_directions.npy when the action space is flex*.

Losses:
- imitation: L1 to the teacher alphas (the teacher already encodes the gate:
  ungated fingers have alpha exactly 0);
- alpha L2 regularization;
- optional rope-consistency (--rope-loss-weight > 0): re-apply the predicted
  alphas through MANO and match the clean rope labels. Needs --mano-cache
  (and the teacher's flex_directions.npy for flex spaces).

Safeguards (the 0026 lessons, all on by default):
- validation split + early stopping on the composite val loss;
- sensor-noise augmentation: gaussian noise / per-finger dropout on the rope
  reading while the target stays the clean-teacher alpha (denoising);
- --shuffle-rope control: permute rope readings across samples; a student
  trained this way must lose its gains, proving the rope is what is learned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.actions import ACTION_SPACES, FLEX_ACTION_SPACES, alpha_dim, apply_action_torch
from ropetrack.refine.alpha_student import (
    STUDENT_FEATURE_DIM,
    RopeAlphaStudent,
    build_student_features,
    feature_stats,
    join_image_features,
    load_image_feature_cache,
    normalize_features,
    save_student_checkpoint,
)
from ropetrack.refine.analysis import json_sanitize
from ropetrack.rope import FINGER_CHAINS


def load_teacher_dir(teacher_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cache_path = teacher_dir / "refiner_eval_cache.npz"
    alpha_path = teacher_dir / "alpha.npy"
    with np.load(cache_path) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    alpha = np.load(alpha_path).astype(np.float32)
    if len(alpha) != len(cache["sample_id"]):
        raise ValueError(f"teacher alpha rows ({len(alpha)}) != cache samples ({len(cache['sample_id'])})")
    return cache, alpha


MERGE_KEYS = ("base_hand_pose", "base_rope_norm", "input_rope_norm", "rope_valid")
OPTIONAL_MERGE_KEYS = ("gt_rope_norm", "rope_chain_m", "fist_ratio")


def load_teacher_dirs(teacher_dirs: list[Path]) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict]]:
    """Concatenate several teacher runs into one training set.

    Multi-dataset distillation: e.g. a FreiHAND mask70 train teacher plus an
    HO3D train teacher. Sample ids are prefixed with the teacher dir name so
    provenance survives the merge; alpha dims must agree across teachers.
    """
    caches, alphas, sources = [], [], []
    for teacher_dir in teacher_dirs:
        cache, alpha = load_teacher_dir(teacher_dir)
        caches.append(cache)
        alphas.append(alpha)
        sources.append({"dir": str(teacher_dir), "num_samples": int(len(alpha))})
    dims = {alpha.shape[1] for alpha in alphas}
    if len(dims) != 1:
        raise ValueError(f"teacher alpha dims disagree across dirs: {sorted(dims)}")
    if len(caches) == 1:
        return caches[0], alphas[0], sources

    merged: dict[str, np.ndarray] = {}
    for key in MERGE_KEYS:
        merged[key] = np.concatenate([np.asarray(cache[key]) for cache in caches], axis=0)
    for key in OPTIONAL_MERGE_KEYS:
        if all(key in cache for cache in caches):
            merged[key] = np.concatenate([np.asarray(cache[key]) for cache in caches], axis=0)
    merged["sample_id"] = np.concatenate([
        np.asarray([f"{teacher_dir.name}/{sid}" for sid in cache["sample_id"]])
        for teacher_dir, cache in zip(teacher_dirs, caches, strict=True)
    ])
    return merged, np.concatenate(alphas, axis=0).astype(np.float32), sources


def shuffle_rope_rows(cache: dict[str, np.ndarray], seed: int) -> None:
    """Control experiment: destroy per-sample rope information in place."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cache["sample_id"]))
    for key in ("base_rope_norm", "input_rope_norm", "rope_valid"):
        cache[key] = np.asarray(cache[key])[perm]


def perturb_rope_reading(
    input_rope: np.ndarray,
    valid: np.ndarray,
    noise_std: float,
    dropout: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rope = np.asarray(input_rope, dtype=np.float32).copy()
    keep = np.asarray(valid, dtype=bool).copy()
    if noise_std > 0.0:
        rope = np.clip(rope + rng.normal(scale=noise_std, size=rope.shape).astype(np.float32), 0.0, 1.0)
    if dropout > 0.0:
        keep = keep & ~(rng.uniform(size=keep.shape) < dropout)
    rope[~keep] = 0.0
    return rope, keep


class RopeConsistency:
    """Optional differentiable rope loss for the student's applied alphas."""

    def __init__(self, cache: dict, action_space: str, directions: np.ndarray | None, mano_cache: Path, device: str, mano_module=None):
        from scripts.rope_refiner.apply_rope_refinement import load_mano_globals, mano_layer, torch_aa_to_rotmat

        self.action_space = action_space
        self.torch_aa_to_rotmat = torch_aa_to_rotmat
        orient_np, betas_np, _ = load_mano_globals(mano_cache, cache["sample_id"])
        self.orient_rot = torch_aa_to_rotmat(torch.from_numpy(orient_np).to(device))[:, None]
        self.betas = torch.from_numpy(betas_np).to(device)
        self.base_pose = torch.from_numpy(np.asarray(cache["base_hand_pose"], dtype=np.float32)).to(device)
        self.gt_rope = torch.from_numpy(np.asarray(cache["gt_rope_norm"], dtype=np.float32)).to(device)
        self.valid = torch.from_numpy(np.asarray(cache["rope_valid"], dtype=np.float32)).to(device)
        self.chain = torch.from_numpy(np.asarray(cache["rope_chain_m"], dtype=np.float32)).to(device).clamp_min(1e-8)
        fist = np.asarray(cache.get("fist_ratio", np.full(len(self.base_pose), 0.5)), dtype=np.float32)
        self.fist_ratio = torch.from_numpy(fist).to(device)
        self.directions = torch.from_numpy(np.asarray(directions, dtype=np.float32)).to(device) if directions is not None else None
        if action_space in FLEX_ACTION_SPACES and self.directions is None:
            raise ValueError(f"{action_space} rope loss requires the teacher's flex_directions.npy")
        self.mano = mano_module if mano_module is not None else mano_layer(device)
        self.chains = FINGER_CHAINS["freihand"]  # wrapper joints are OpenPose-ordered

    def __call__(self, alpha: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        from scripts.rope_refiner.apply_rope_refinement import torch_rope_norm

        dirs = self.directions[index] if self.directions is not None else None
        hand_pose = apply_action_torch(self.base_pose[index], alpha, self.action_space, dirs)
        out = self.mano(
            global_orient=self.orient_rot[index],
            hand_pose=self.torch_aa_to_rotmat(hand_pose.reshape(-1, 15, 3)),
            betas=self.betas[index],
            pose2rot=False,
        )
        pred_rope = torch_rope_norm(out.joints, self.chains, self.chain[index], self.fist_ratio[index])
        diff = (pred_rope - self.gt_rope[index]) * self.valid[index]
        return (diff * diff).sum() / self.valid[index].sum().clamp_min(1.0)


def train_student(
    cache: dict[str, np.ndarray],
    teacher_alpha: np.ndarray,
    action_space: str,
    out_dir: Path,
    *,
    gate_threshold: float | None,
    max_alpha: float = 0.5,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    batch_size: int = 512,
    max_epochs: int = 200,
    patience: int = 20,
    val_frac: float = 0.1,
    seed: int = 0,
    aug_noise_std: float = 0.05,
    aug_dropout: float = 0.1,
    alpha_l2: float = 1e-4,
    rope_loss_weight: float = 0.0,
    shuffle_rope: bool = False,
    mano_cache: Path | None = None,
    directions: np.ndarray | None = None,
    device: str = "cpu",
    mano_module=None,
    sources: list[dict] | None = None,
    image_features: np.ndarray | None = None,
) -> dict:
    if action_space not in ACTION_SPACES:
        raise ValueError(f"unsupported action space: {action_space}")
    out_dim = alpha_dim(action_space)
    if teacher_alpha.shape[1] != out_dim:
        raise ValueError(f"teacher alpha dim {teacher_alpha.shape[1]} != {out_dim} for {action_space}")

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if shuffle_rope:
        shuffle_rope_rows(cache, seed + 1)

    num = len(cache["sample_id"])
    perm = rng.permutation(num)
    val_count = max(1, int(round(num * val_frac)))
    val_idx, train_idx = perm[:val_count], perm[val_count:]
    if len(train_idx) == 0:
        raise ValueError("no training samples left after validation split")

    pose = np.asarray(cache["base_hand_pose"], dtype=np.float32)
    base_rope = np.asarray(cache["base_rope_norm"], dtype=np.float32)
    input_rope = np.asarray(cache["input_rope_norm"], dtype=np.float32)
    valid = np.asarray(cache["rope_valid"], dtype=bool)

    image_feature_dim = 0
    if image_features is not None:
        image_features = np.asarray(image_features, dtype=np.float32)
        if len(image_features) != num:
            raise ValueError(f"image features rows ({len(image_features)}) != samples ({num})")
        image_feature_dim = int(image_features.shape[1])

    def full_features(rows: np.ndarray, rope_rows: np.ndarray, valid_rows: np.ndarray) -> np.ndarray:
        base = build_student_features(pose[rows], base_rope[rows], rope_rows, valid_rows)
        if image_features is None:
            return base
        return np.concatenate([base, image_features[rows]], axis=1)

    all_rows = np.arange(num)
    clean_features = full_features(all_rows, input_rope, valid)
    mean, std = feature_stats(clean_features[train_idx])
    val_features = torch.from_numpy(normalize_features(clean_features[val_idx], mean, std)).to(device)
    target = torch.from_numpy(teacher_alpha).to(device)

    rope_loss = None
    if rope_loss_weight > 0.0:
        if mano_cache is None:
            raise ValueError("--rope-loss-weight > 0 requires --mano-cache")
        rope_loss = RopeConsistency(cache, action_space, directions, mano_cache, device, mano_module=mano_module)

    in_dim = STUDENT_FEATURE_DIM + image_feature_dim
    model = RopeAlphaStudent(out_dim=out_dim, hidden_dim=hidden_dim, max_alpha=max_alpha, in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def composite_loss(alpha_pred: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.l1_loss(alpha_pred, target[index])
        loss = loss + alpha_l2 * (alpha_pred * alpha_pred).mean()
        if rope_loss is not None:
            loss = loss + rope_loss_weight * rope_loss(alpha_pred, index)
        return loss

    # predict-zero baseline: what the untrained (zero-init) student scores
    zero_baseline = float(torch.nn.functional.l1_loss(torch.zeros_like(target[val_idx]), target[val_idx]))

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    log = []
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_idx)
        train_loss_sum, train_batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            rope_batch, valid_batch = perturb_rope_reading(input_rope[batch], valid[batch], aug_noise_std, aug_dropout, rng)
            features = full_features(batch, rope_batch, valid_batch)
            features_t = torch.from_numpy(normalize_features(features, mean, std)).to(device)
            index_t = torch.from_numpy(batch).to(device)
            alpha_pred = model(features_t)
            loss = composite_loss(alpha_pred, index_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach())
            train_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = float(composite_loss(model(val_features), torch.from_numpy(val_idx).to(device)))
        log.append({"epoch": epoch, "train_loss": train_loss_sum / max(train_batches, 1), "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    config = {
        "out_dim": out_dim,
        "hidden_dim": hidden_dim,
        "max_alpha": max_alpha,
        "in_dim": in_dim,
        "image_feature_dim": image_feature_dim,
        "action_space": action_space,
        "gate_threshold": gate_threshold,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "aug_noise_std": aug_noise_std,
        "aug_dropout": aug_dropout,
        "alpha_l2": alpha_l2,
        "rope_loss_weight": rope_loss_weight,
        "shuffle_rope": shuffle_rope,
        "seed": seed,
        "num_train": int(len(train_idx)),
        "num_val": int(len(val_idx)),
        "sources": sources or [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_student_checkpoint(out_dir / "student.pt", model, config)
    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(log),
        "zero_baseline_val_l1": zero_baseline,
        "beats_zero_baseline": bool(best_val < zero_baseline),
        "config": config,
    }
    (out_dir / "train_log.json").write_text(json.dumps(json_sanitize({"summary": summary, "log": log}), indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill the rope test-time optimizer into a one-pass alpha student.")
    parser.add_argument("--teacher-dir", type=Path, required=True, nargs="+",
                        help="One or more apply_rope_refinement.py optimize out-dirs on TRAINING splits "
                             "(each with refiner_eval_cache.npz + alpha.npy); multiple dirs are concatenated "
                             "for multi-dataset distillation.")
    parser.add_argument("--action-space", choices=list(ACTION_SPACES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate-threshold", type=float, default=0.1,
                        help="Stored in the checkpoint; applied as a hard rule at inference (not learned).")
    parser.add_argument("--max-alpha", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aug-noise-std", type=float, default=0.05)
    parser.add_argument("--aug-dropout", type=float, default=0.1)
    parser.add_argument("--alpha-l2", type=float, default=1e-4)
    parser.add_argument("--rope-loss-weight", type=float, default=0.0)
    parser.add_argument("--mano-cache", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None,
                        help="P3: frozen backbone feature_cache.npz (scripts/rope_head/extract_feature_cache.py) "
                             "for the SAME split as the teacher dir; joined by sample_id and concatenated to the "
                             "65-d rope/pose features. Single --teacher-dir only for now.")
    parser.add_argument("--shuffle-rope", action="store_true",
                        help="Control: permute rope readings across samples; the trained student's gains must vanish.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.rope_loss_weight > 0.0 and len(args.teacher_dir) > 1:
        raise ValueError("--rope-loss-weight > 0 currently supports a single --teacher-dir "
                         "(per-dataset MANO caches are not merged)")
    if args.feature_cache is not None and len(args.teacher_dir) > 1:
        raise ValueError("--feature-cache currently supports a single --teacher-dir "
                         "(one feature cache per split; multi-dataset P3 comes later)")
    cache, teacher_alpha, sources = load_teacher_dirs(args.teacher_dir)
    image_features = None
    if args.feature_cache is not None:
        feature_ids, features = load_image_feature_cache(args.feature_cache)
        image_features = join_image_features(cache["sample_id"], feature_ids, features)
    directions = None
    if len(args.teacher_dir) == 1:
        directions_path = args.teacher_dir[0] / "flex_directions.npy"
        if args.action_space in FLEX_ACTION_SPACES and directions_path.exists():
            directions = np.load(directions_path)
    summary = train_student(
        cache,
        teacher_alpha,
        args.action_space,
        args.out_dir,
        gate_threshold=args.gate_threshold,
        max_alpha=args.max_alpha,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        seed=args.seed,
        aug_noise_std=args.aug_noise_std,
        aug_dropout=args.aug_dropout,
        alpha_l2=args.alpha_l2,
        rope_loss_weight=args.rope_loss_weight,
        shuffle_rope=args.shuffle_rope,
        mano_cache=args.mano_cache,
        directions=directions,
        device=args.device,
        sources=sources,
        image_features=image_features,
    )
    print(json.dumps(json_sanitize(summary), indent=2))
    return summary


if __name__ == "__main__":
    main()
