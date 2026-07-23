#!/usr/bin/env python3
"""Train and apply a small RGB/rope residual global-orientation head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.apply import load_mano_globals, mano_layer, torch_aa_to_rotmat
from ropetrack.refine.direct_pose import (
    episode_split,
    load_arrays,
    load_model,
    read_npz,
    tensor_batch,
)


class OrientationHead(nn.Module):
    def __init__(self, mode: str, token_dim: int, hidden_dim: int = 128, max_delta: float = math.pi) -> None:
        super().__init__()
        if mode not in {"rgb", "rope", "rgb_rope"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.max_delta = max_delta
        self.state = nn.Sequential(nn.Linear(48, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.sensor = (
            nn.Sequential(nn.Linear(20, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
            if "rope" in mode else None
        )
        if "rgb" in mode:
            self.token_proj = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        else:
            self.token_proj = self.attention = None
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.state(torch.cat((batch["global_orient"], batch["base_hand_pose"]), dim=1))
        if self.sensor is not None:
            sensor = torch.cat(
                (
                    batch["base_rope_norm"],
                    batch["input_rope_norm"],
                    batch["input_rope_norm"] - batch["base_rope_norm"],
                    batch["rope_valid"],
                ),
                dim=1,
            )
            hidden = hidden + self.sensor(sensor)
        if self.attention is not None:
            tokens = self.token_proj(batch["tokens"])
            attended, _ = self.attention(hidden[:, None], tokens, tokens, need_weights=False)
            hidden = hidden + attended[:, 0]
        return self.max_delta * torch.tanh(self.output(hidden))


def participant_id(sample_id: object) -> str:
    return str(sample_id).replace("\\", "/").split("/", 1)[0].split("_", 1)[0]


def shuffle_rope_within_participant(arrays: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    """Destroy frame correspondence while preserving each participant's rope distribution."""
    shuffled_arrays = dict(arrays)
    input_rope = arrays["input_rope_norm"].copy()
    rope_valid = arrays["rope_valid"].copy()
    participants = np.asarray([participant_id(value) for value in arrays["sample_id"]])
    rng = np.random.default_rng(seed)
    for participant in np.unique(participants):
        rows = np.flatnonzero(participants == participant)
        source = rng.permutation(rows)
        input_rope[rows] = arrays["input_rope_norm"][source]
        rope_valid[rows] = arrays["rope_valid"][source]
    shuffled_arrays["input_rope_norm"] = input_rope
    shuffled_arrays["rope_valid"] = rope_valid
    return shuffled_arrays


def decode(pose_model, orient_model, mano, batch, orient_batch=None):
    if orient_batch is None:
        orient_batch = batch
    with torch.no_grad():
        refined_pose = pose_model(
            batch["base_hand_pose"],
            batch["base_rope_norm"],
            batch["input_rope_norm"],
            batch["rope_valid"],
            batch["tokens"],
        )
    delta_rotation = torch_aa_to_rotmat(orient_model(orient_batch))
    global_rotation = delta_rotation @ torch_aa_to_rotmat(batch["global_orient"])
    output = mano(
        global_orient=global_rotation[:, None],
        hand_pose=torch_aa_to_rotmat(refined_pose.reshape(-1, 15, 3)),
        betas=batch["betas"],
        pose2rot=False,
    )
    joints = output.joints
    mirror = torch.ones((len(joints), 1, 3), device=joints.device)
    mirror[:, 0, 0] = torch.where(batch["is_right"], 1.0, -1.0)
    return joints * mirror, global_rotation


def root_metrics(joints: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_root, gt_root = joints - joints[:, :1], gt - gt[:, :1]
    loss = torch.nn.functional.l1_loss(pred_root, gt_root)
    mm = torch.linalg.norm(pred_root - gt_root, dim=-1).mean() * 1000.0
    return loss, mm


def evaluate(pose_model, orient_model, mano, arrays, rows, batch_size, device) -> float:
    orient_model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            batch = tensor_batch(arrays, selected, device)
            joints, _ = decode(pose_model, orient_model, mano, batch)
            _, value = root_metrics(joints, batch["gt_xyz"])
            total += float(value) * len(selected)
    return total / len(rows)


def train(args) -> None:
    arrays = read_npz(args.bundle)
    train_rows, val_rows = episode_split(arrays["sample_id"], args.val_fraction, args.seed)
    pose_model, _ = load_model(args.pose_checkpoint, args.device)
    pose_model.requires_grad_(False)
    token_dim = int(arrays["tokens"].shape[2])
    model = OrientationHead(args.mode, token_dim, args.hidden_dim, args.max_delta).to(args.device)
    mano = mano_layer(args.device).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, best_epoch, best_state, log = float("inf"), -1, None, []
    for epoch in range(args.max_epochs):
        model.train()
        order = rng.permutation(train_rows)
        for start in range(0, len(train_rows), args.batch_size):
            selected = order[start : start + args.batch_size]
            batch = tensor_batch(arrays, selected, args.device)
            joints, _ = decode(pose_model, model, mano, batch)
            loss, _ = root_metrics(joints, batch["gt_xyz"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val = evaluate(pose_model, model, mano, arrays, val_rows, args.batch_size, args.device)
        log.append({"epoch": epoch, "val_root_mm": val})
        print(json.dumps(log[-1]), flush=True)
        if val < best - args.min_delta:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": args.mode,
        "token_dim": token_dim,
        "hidden_dim": args.hidden_dim,
        "max_delta": args.max_delta,
        "best_epoch": best_epoch,
        "best_val_root_mm": best,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
    }
    torch.save({"model_state": model.state_dict(), "config": config}, args.out_dir / "model.pt")
    (args.out_dir / "train_log.json").write_text(json.dumps({"config": config, "log": log}, indent=2))


def apply(args) -> None:
    arrays = load_arrays(args.cache, args.mano_cache, feature_cache_path=args.feature_cache)
    orient_arrays = (
        shuffle_rope_within_participant(arrays, args.orient_rope_seed)
        if args.orient_rope_mode == "shuffle" else arrays
    )
    pose_model, _ = load_model(args.pose_checkpoint, args.device)
    payload = torch.load(args.orient_checkpoint, map_location=args.device, weights_only=True)
    config = payload["config"]
    model = OrientationHead(
        config["mode"], config["token_dim"], config["hidden_dim"], config["max_delta"]
    ).to(args.device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    mano = mano_layer(args.device).requires_grad_(False)
    _, _, cam_t = load_mano_globals(args.mano_cache, arrays["sample_id"])
    xyz = []
    with torch.no_grad():
        for start in range(0, len(arrays["sample_id"]), args.batch_size):
            rows = np.arange(start, min(start + args.batch_size, len(arrays["sample_id"])))
            batch = tensor_batch(arrays, rows, args.device)
            orient_batch = tensor_batch(orient_arrays, rows, args.device)
            joints, _ = decode(pose_model, model, mano, batch, orient_batch)
            xyz.extend((joints.cpu().numpy() + cam_t[rows, None, :]).tolist())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pred.json").write_text(json.dumps([xyz, [None] * len(xyz)], separators=(",", ":")))
    np.save(args.out_dir / "sample_id.npy", arrays["sample_id"])
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": config,
                "num_samples": len(xyz),
                "orient_rope_mode": args.orient_rope_mode,
                "orient_rope_seed": args.orient_rope_seed,
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    train_p = sub.add_parser("train")
    train_p.add_argument("--bundle", type=Path, required=True)
    train_p.add_argument("--pose-checkpoint", type=Path, required=True)
    train_p.add_argument("--mode", choices=("rgb", "rope", "rgb_rope"), required=True)
    train_p.add_argument("--out-dir", type=Path, required=True)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--batch-size", type=int, default=512)
    train_p.add_argument("--hidden-dim", type=int, default=128)
    train_p.add_argument("--max-delta", type=float, default=math.pi)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--max-epochs", type=int, default=40)
    train_p.add_argument("--patience", type=int, default=6)
    train_p.add_argument("--min-delta", type=float, default=0.1)
    train_p.add_argument("--val-fraction", type=float, default=0.1)
    train_p.add_argument("--seed", type=int, default=0)
    apply_p = sub.add_parser("apply")
    apply_p.add_argument("--cache", type=Path, required=True)
    apply_p.add_argument("--mano-cache", type=Path, required=True)
    apply_p.add_argument("--feature-cache", type=Path, required=True)
    apply_p.add_argument("--pose-checkpoint", type=Path, required=True)
    apply_p.add_argument("--orient-checkpoint", type=Path, required=True)
    apply_p.add_argument("--out-dir", type=Path, required=True)
    apply_p.add_argument("--device", default="cuda")
    apply_p.add_argument("--batch-size", type=int, default=512)
    apply_p.add_argument("--orient-rope-mode", choices=("correct", "shuffle"), default="correct")
    apply_p.add_argument("--orient-rope-seed", type=int, default=4242)
    return parser.parse_args()


def self_check() -> None:
    batch = {
        "global_orient": torch.randn(2, 3),
        "base_hand_pose": torch.randn(2, 45),
        "base_rope_norm": torch.rand(2, 5),
        "input_rope_norm": torch.rand(2, 5),
        "rope_valid": torch.ones(2, 5),
        "tokens": torch.randn(2, 12, 8),
    }
    for mode in ("rgb", "rope", "rgb_rope"):
        torch.testing.assert_close(OrientationHead(mode, 8)(batch), torch.zeros(2, 3))


if __name__ == "__main__":
    args = parse_args()
    if args.command == "self-check":
        self_check()
        print("self-check-ok")
    elif args.command == "train":
        train(args)
    else:
        apply(args)
