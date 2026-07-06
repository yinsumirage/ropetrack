from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from ropetrack.refine.cache import load_npz_cache, make_refiner_features, validate_cache
from ropetrack.refine.rope_refiner import RopePoseRefiner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cached rope pose refiner.")
    parser.add_argument("cache", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rope-key", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    rope_key = args.rope_key or ckpt.get("config", {}).get("rope_key", "input_rope_norm")
    cache = load_npz_cache(args.cache)
    validate_cache(cache, require_target_hand_pose=False)
    arrays = make_refiner_features(cache, rope_key)

    device = torch.device(args.device)
    model = RopePoseRefiner(hidden_dim=ckpt.get("config", {}).get("hidden_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tensors = {key: torch.from_numpy(value).to(device) for key, value in arrays.items() if key != "target_hand_pose"}

    with torch.no_grad():
        refined, delta = model(
            tensors["base_hand_pose"],
            tensors["base_rope_norm"],
            tensors["input_rope_norm"],
            tensors["rope_valid"],
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "refined_hand_pose.npy", refined.cpu().numpy())
    np.save(args.out_dir / "sample_id.npy", cache["sample_id"])
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "mean_abs_delta": float(delta.abs().mean().cpu()),
                "num_samples": int(refined.shape[0]),
            },
            indent=2,
        )
    )
    return args.out_dir


if __name__ == "__main__":
    main()
