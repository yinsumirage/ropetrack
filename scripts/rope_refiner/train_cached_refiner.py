from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ropetrack.refine.cache import load_npz_cache, make_refiner_features, validate_cache
from ropetrack.refine.rope_refiner import RopePoseRefiner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny cached rope pose refiner.")
    parser.add_argument("cache", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--delta-l2", type=float, default=1e-4)
    parser.add_argument("--rope-key", default="input_rope_norm")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cache = load_npz_cache(args.cache)
    validate_cache(cache, require_target_hand_pose=True)
    arrays = make_refiner_features(cache, args.rope_key)
    if "target_hand_pose" not in arrays:
        raise ValueError("training requires target_hand_pose in the cache")

    device = torch.device(args.device)
    model = RopePoseRefiner(hidden_dim=args.hidden_dim).to(device)
    tensors = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}

    for _ in range(args.steps):
        refined, delta = model(
            tensors["base_hand_pose"],
            tensors["base_rope_norm"],
            tensors["input_rope_norm"],
            tensors["rope_valid"],
        )
        loss = torch.nn.functional.l1_loss(refined, tensors["target_hand_pose"]) + args.delta_l2 * delta.square().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= args.lr * param.grad

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "rope_key": args.rope_key,
            },
        },
        args.checkpoint,
    )
    return args.checkpoint


if __name__ == "__main__":
    main()
