from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ropetrack.eval.config import DEFAULT_CONFIG, build_run_args  # noqa: E402
from ropetrack.eval.pipeline import run_export  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RopeTrack evaluation export.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to run; <=0 means all.")
    parser.add_argument("--split", choices=["evaluation", "training"], default="evaluation")
    parser.add_argument("--mode", choices=["detector", "gt_bbox"], default=None)
    parser.add_argument("--backend", choices=["wilor", "hamer"], default=None)
    parser.add_argument("--units", choices=["m", "mm"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--detector-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--joint-source", choices=["mano_vertices", "model_keypoints", "anyhand_keypoints"], default=None)
    parser.add_argument("--ho3d-root", type=Path, default=None)
    parser.add_argument("--freihand-root", type=Path, default=None)
    parser.add_argument("--protocol-check-samples", type=int, default=None)
    parser.add_argument("--protocol-tolerance-m", type=float, default=None)
    parser.add_argument("--wilor-ckpt", type=Path, default=None)
    parser.add_argument("--wilor-cfg", type=Path, default=None)
    parser.add_argument("--hamer-ckpt", type=Path, default=None)
    parser.add_argument("--save-mano-cache", action="store_true", default=False)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-num-workers", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    cli = parse_args(argv)
    args = build_run_args(**{k: v for k, v in vars(cli).items() if v is not None})
    args.split = cli.split
    args.save_mano_cache = cli.save_mano_cache
    return run_export(args)


if __name__ == "__main__":
    main()
