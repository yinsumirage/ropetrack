#!/usr/bin/env python3
"""Cache frozen backbone image features for the P3 rope-conditioned head.

The rope-only student is information-limited: rope observes finger curl but
not where along the finger to bend, abduction, or twist (strong-oracle gaps
of 0.3-0.8 mm, see docs/2026-07-07-report-results-pack.md Act 5). The P3 head
adds frozen backbone image features as evidence. This script harvests those
features once, offline, so head training consumes cached arrays exactly like
the MANO/alpha caches.

Alignment guarantees:

- samples and GT bboxes come from the same adapters as the benchmark export
  (`iter_hand_pose_samples` / `load_gt_bbox_candidates`), and the crop
  preprocessing is the export's own `CrossImageBBoxDataset`, so features
  correspond to the same crops the cached MANO predictions came from;
- features are captured with a forward hook on `model.backbone` during a
  normal full-model forward - whatever the head consumed, we cache;
- rows are keyed by `sample_id` in export order.

Output `feature_cache.npz`: `sample_id`, `features` [N, C] (pooled, fp32),
optional `tokens` [N, T, C] (fp16; ~16 GB for a 32.5k train split - only with
--save-tokens), plus backend/pooling metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.datasets.hand_pose import (  # noqa: E402
    iter_hand_pose_samples,
    load_gt_bbox_candidates,
)
from ropetrack.eval.config import DEFAULT_CONFIG, build_run_args  # noqa: E402


def first_candidate_per_sample(num_samples: int, candidates: list) -> list:
    """One bbox candidate per sample (gt_bbox mode); strict on gaps."""
    first: dict[int, object] = {}
    for candidate in candidates:
        first.setdefault(candidate.sample_index, candidate)
    missing = [idx for idx in range(num_samples) if idx not in first]
    if missing:
        raise ValueError(f"samples without a gt bbox candidate: {missing[:5]} (total {len(missing)})")
    return [first[idx] for idx in range(num_samples)]


def dataset_root(args) -> Path:
    if args.adapter == "freihand":
        return args.freihand_root
    if args.adapter == "ho3d":
        return args.ho3d_root
    return args.root


def select_feature_tensor(output):
    """Select the image-feature tensor from common backbone output containers."""
    import torch

    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("img_feat", "vit_out", "features", "feature", "tokens"):
            if key in output:
                try:
                    return select_feature_tensor(output[key])
                except ValueError:
                    pass
        values = list(output.values())
    elif isinstance(output, (tuple, list)):
        # WiLoR's ViT backbone returns (mano_params, cam, mano_feats, img_feat).
        # Search from the end so the image feature map wins over parameter dicts.
        values = list(reversed(output))
    else:
        raise ValueError(f"unsupported backbone output type: {type(output).__name__}")

    for value in values:
        try:
            feat = select_feature_tensor(value)
        except ValueError:
            continue
        if feat.dim() in (3, 4):
            return feat
    raise ValueError(f"no 3D/4D tensor feature found in backbone output type {type(output).__name__}")


def pool_feature_map(feat, pooling: str, token_grid: tuple[int, int] | None = None):
    """Pool a backbone output to [B, C]; accepts [B, C, H, W] or [B, T, C]."""
    import torch

    feat = select_feature_tensor(feat)
    if feat.dim() == 4:  # [B, C, H, W]
        if token_grid is not None:
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, token_grid)
        tokens = feat.flatten(2).transpose(1, 2)  # [B, T, C]
    elif feat.dim() == 3:  # [B, T, C]
        if token_grid is not None:
            raise ValueError("--token-grid requires a 4D spatial feature map")
        tokens = feat
    else:
        raise ValueError(f"unsupported backbone feature shape: {tuple(feat.shape)}")
    if pooling == "mean":
        pooled = tokens.mean(dim=1)
    elif pooling == "meanmax":
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=1)
    else:
        raise ValueError(f"unsupported pooling: {pooling}")
    return pooled, tokens


def run_extraction(
    model,
    loader,
    num_samples: int,
    device: str,
    pooling: str = "mean",
    save_tokens: bool = False,
    token_grid: tuple[int, int] | None = None,
):
    """Feature harvest loop; model must expose a hookable `.backbone`."""
    import torch

    captured: dict[str, object] = {}

    def hook(_module, _inputs, output):
        captured["feat"] = output

    handle = model.backbone.register_forward_hook(hook)
    pooled_rows: list = [None] * num_samples
    token_rows: list = [None] * num_samples
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                captured.pop("feat", None)
                model(batch)
                if "feat" not in captured:
                    raise RuntimeError("backbone hook captured nothing; is model.backbone the right module?")
                pooled, tokens = pool_feature_map(captured["feat"], pooling, token_grid)
                indices = batch["candidate_index"].detach().cpu().numpy().astype(int).tolist()
                pooled_np = pooled.detach().float().cpu().numpy()
                tokens_np = tokens.detach().half().cpu().numpy() if save_tokens else None
                for row, candidate_index in enumerate(indices):
                    pooled_rows[candidate_index] = pooled_np[row]
                    if save_tokens:
                        token_rows[candidate_index] = tokens_np[row]
    finally:
        handle.remove()

    if any(row is None for row in pooled_rows):
        missing = sum(row is None for row in pooled_rows)
        raise RuntimeError(f"{missing} samples received no features")
    features = np.stack(pooled_rows, axis=0)
    tokens_arr = np.stack(token_rows, axis=0) if save_tokens else None
    return features, tokens_arr


def write_feature_cache(
    path: Path,
    sample_ids: list[str],
    features: np.ndarray,
    tokens: np.ndarray | None,
    meta: dict,
) -> Path:
    if len(sample_ids) != len(features):
        raise ValueError(f"sample_id rows ({len(sample_ids)}) != feature rows ({len(features)})")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sample_id": np.asarray(sample_ids),
        "features": np.asarray(features, dtype=np.float32),
        "meta_json": np.asarray(json.dumps(meta)),
    }
    if tokens is not None:
        arrays["tokens"] = np.asarray(tokens, dtype=np.float16)
    np.savez(path, **arrays)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen backbone features for the rope-conditioned head.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, help="Dataset config name (same registry as scripts/eval.py).")
    parser.add_argument("--method", default=None)
    parser.add_argument("--split", choices=["evaluation", "training"], default="evaluation")
    parser.add_argument("--limit", type=int, default=None, help="<=0 means all.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--freihand-root", type=Path, default=None,
                        help="Override the dataset config root (e.g. the mask70 TRAIN hard root).")
    parser.add_argument("--ho3d-root", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None,
                        help="Override the root for generic adapters such as HOT3D.")
    parser.add_argument("--pooling", choices=["mean", "meanmax"], default="mean")
    parser.add_argument("--save-tokens", action="store_true",
                        help="Also store the full fp16 token grid (large: ~16 GB for a 32.5k split).")
    parser.add_argument("--token-grid", type=int, nargs=2, metavar=("H", "W"), default=None,
                        help="Adaptive-pool a spatial map before --save-tokens (e.g. 4 3).")
    parser.add_argument("--output", type=Path, required=True, help="feature_cache.npz path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    import torch

    cli = parse_args(argv)
    args = build_run_args(**{k: v for k, v in vars(cli).items()
                             if v is not None and k not in {"pooling", "save_tokens", "token_grid", "output", "split"}})
    if args.mode != "gt_bbox":
        raise ValueError("feature extraction currently supports gt_bbox datasets only")

    from ropetrack.backends.hand_predictor import HandPredictor
    from ropetrack.eval.pipeline import CrossImageBBoxDataset, backend_model_and_cfg, predictor_kwargs

    root = dataset_root(args)
    samples = list(iter_hand_pose_samples(args.adapter, root, args.limit, cli.split))
    candidates = first_candidate_per_sample(len(samples), load_gt_bbox_candidates(args.adapter, samples))

    predictor = HandPredictor(**predictor_kwargs(args))
    model, cfg = backend_model_and_cfg(predictor, args.backend)
    dataset = CrossImageBBoxDataset(cfg, candidates, args.backend, predictor.rescale_factor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    features, tokens = run_extraction(
        model, loader, len(samples), args.device, pooling=cli.pooling,
        save_tokens=cli.save_tokens, token_grid=tuple(cli.token_grid) if cli.token_grid else None,
    )
    meta = {
        "dataset": args.dataset,
        "split": cli.split,
        "backend": args.backend,
        "method": args.method,
        "pooling": cli.pooling,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(len(samples)),
        "save_tokens": bool(cli.save_tokens),
        "token_grid": cli.token_grid,
        "mode": "gt_bbox",
    }
    write_feature_cache(cli.output, [sample.sample_id for sample in samples], features, tokens, meta)
    print(f"Wrote feature cache: {cli.output} features={features.shape}"
          + (f" tokens={tokens.shape}" if tokens is not None else ""))
    return cli.output


if __name__ == "__main__":
    main()
