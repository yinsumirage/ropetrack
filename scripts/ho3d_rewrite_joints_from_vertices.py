from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bench_ho3d_v2_wilor import ho3d_joints_from_vertices, load_mano_j_regressor


def rewrite(args: argparse.Namespace) -> Path:
    pred = json.loads((args.src / "eval_input" / "pred.json").read_text())
    _, verts = pred
    j_regressor = load_mano_j_regressor(args.mano_path)
    xyz = [ho3d_joints_from_vertices(v, j_regressor).tolist() for v in verts]

    eval_input = args.out_dir / "eval_input"
    eval_input.mkdir(parents=True, exist_ok=True)
    (eval_input / "pred.json").write_text(json.dumps([xyz, verts]))
    for name in ["evaluation_xyz.json", "evaluation_verts.json"]:
        shutil.copy2(args.src / "eval_input" / name, eval_input / name)
    return args.out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--mano-path",
        type=Path,
        default=Path("third_party/anyhand/mano_data/MANO_RIGHT.pkl"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    rewrite(parse_args())
