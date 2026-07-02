from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

HO3D_TIP_VERTEX_IDS = np.asarray([744, 333, 444, 555, 672], dtype=np.int64)


@dataclass(frozen=True)
class Ho3dSample:
    sample_id: str
    image_path: Path
    meta_path: Path


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def iter_ho3d_samples(root: Path, limit: int | None) -> Iterable[Ho3dSample]:
    eval_dir = root / "evaluation"
    list_path = root / "evaluation.txt"
    if list_path.exists():
        ids = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    elif (root / "evaluation_xyz.json").exists():
        ids = infer_sample_order_from_gt_roots(root)
    else:
        ids = [
            f"{seq.name}/{img.stem}"
            for seq in sorted(eval_dir.iterdir())
            if seq.is_dir()
            for img in sorted((seq / "rgb").glob("*.png"))
        ]

    if limit is not None and limit <= 0:
        limit = None

    for sample_id in ids[:limit]:
        seq, frame = sample_id.split("/")
        yield Ho3dSample(
            sample_id=sample_id,
            image_path=eval_dir / seq / "rgb" / f"{frame}.png",
            meta_path=eval_dir / seq / "meta" / f"{frame}.pkl",
        )


def infer_sample_order_from_gt_roots(root: Path) -> list[str]:
    eval_dir = root / "evaluation"
    meta_items = []
    for meta_path in eval_dir.glob("*/meta/*.pkl"):
        with meta_path.open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        seq = meta_path.parents[1].name
        sample_id = f"{seq}/{meta_path.stem}"
        meta_items.append((sample_id, np.asarray(meta["handJoints3D"], dtype=np.float64).reshape(-1)[:3]))

    meta_ids = [item[0] for item in meta_items]
    meta_roots = np.stack([item[1] for item in meta_items], axis=0)
    gt_roots = np.asarray(json.loads((root / "evaluation_xyz.json").read_text()), dtype=np.float64)[:, 0, :]
    ids = []
    used = set()
    max_dist = 0.0
    for idx, root_xyz in enumerate(gt_roots):
        dists = np.linalg.norm(meta_roots - root_xyz[None, :], axis=1)
        match = next(int(i) for i in np.argsort(dists) if int(i) not in used)
        used.add(match)
        max_dist = max(max_dist, float(dists[match]))
        ids.append(meta_ids[match])
    if max_dist > 1e-4:
        raise ValueError(f"HO3D order matching too loose: max root distance {max_dist}")
    return ids


def select_hand(hands):
    if not hands:
        return None
    return max(hands, key=lambda hand: float(hand.score))


def hand_bbox_from_meta(meta: dict) -> np.ndarray:
    return np.asarray(meta["handBoundingBox"], dtype=np.float32).reshape(1, 4)


def to_opengl_camera(points, cam_t, units: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) + np.asarray(cam_t, dtype=np.float32)[None, :]
    if units == "mm":
        pts = pts / 1000.0
    elif units != "m":
        raise ValueError(f"unsupported units: {units}")
    pts[:, 1] *= -1.0
    pts[:, 2] *= -1.0
    return pts


def dense_regressor(regressor) -> np.ndarray:
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    return np.asarray(regressor, dtype=np.float32)


def load_mano_j_regressor(mano_path: Path) -> np.ndarray:
    with mano_path.open("rb") as f:
        mano = pickle.load(f, encoding="latin1")
    return dense_regressor(mano["J_regressor"])


def ho3d_joints_from_vertices(vertices: np.ndarray, j_regressor: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    joints16 = np.asarray(j_regressor, dtype=np.float32) @ verts
    tips = verts[HO3D_TIP_VERTEX_IDS]
    return np.concatenate([joints16, tips], axis=0)


def optional_path_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def run_export(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    anyhand_root = repo / "third_party" / "anyhand"
    sys.path.insert(0, str(anyhand_root))

    from scripts.rgb_predictor import AnyHandPredictor
    import cv2

    samples = list(iter_ho3d_samples(args.ho3d_root, args.limit))
    out_dir = args.out_dir
    eval_input = out_dir / "eval_input"
    eval_results = out_dir / "eval_results"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_results.mkdir(parents=True, exist_ok=True)

    j_regressor = load_mano_j_regressor(anyhand_root / "mano_data" / "MANO_RIGHT.pkl")
    xyz_pred, verts_pred, failures = [], [], []

    with pushd(anyhand_root):
        predictor = AnyHandPredictor(
            backend="wilor",
            device=args.device,
            batch_size=args.batch_size,
            wilor_ckpt=optional_path_str(args.wilor_ckpt),
            wilor_cfg=optional_path_str(args.wilor_cfg),
        )

        for idx, sample in enumerate(samples):
            try:
                if args.mode == "gt_bbox":
                    img = cv2.imread(str(sample.image_path))
                    if img is None:
                        raise FileNotFoundError(sample.image_path)
                    with sample.meta_path.open("rb") as f:
                        meta = pickle.load(f, encoding="latin1")
                    hands = predictor._run_wilor(
                        img,
                        hand_bbox_from_meta(meta),
                        np.asarray([1.0], dtype=np.float32),
                        np.asarray([1.0], dtype=np.float32),
                    )
                else:
                    hands = predictor.predict(str(sample.image_path))
                hand = select_hand(hands)
                if hand is None:
                    raise RuntimeError("no hand detected")
                verts = to_opengl_camera(hand.vertices, hand.cam_t, args.units)
                if args.joint_source == "mano_vertices":
                    xyz = ho3d_joints_from_vertices(verts, j_regressor)
                else:
                    xyz = to_opengl_camera(hand.keypoints_3d, hand.cam_t, args.units)
            except Exception as exc:
                failures.append({"idx": idx, "sample_id": sample.sample_id, "error": repr(exc)})
                xyz = np.zeros((21, 3), dtype=np.float32)
                verts = np.zeros((778, 3), dtype=np.float32)
            xyz_pred.append(xyz.tolist())
            verts_pred.append(verts.tolist())

    (eval_input / "pred.json").write_text(json.dumps([xyz_pred, verts_pred]))
    shutil.copy2(args.ho3d_root / "evaluation_xyz.json", eval_input / "evaluation_xyz.json")
    shutil.copy2(args.ho3d_root / "evaluation_verts.json", eval_input / "evaluation_verts.json")
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "backend": "anyhand_wilor",
        "mode": args.mode,
        "limit": args.limit,
        "num_samples": len(samples),
        "num_failures": len(failures),
        "units": args.units,
        "joint_source": args.joint_source,
        "wilor_ckpt": optional_path_str(args.wilor_ckpt),
        "wilor_cfg": optional_path_str(args.wilor_cfg),
        "coordinate_transform": "points + cam_t; output metres; flip y and z to OpenGL",
        "sample_order": [sample.sample_id for sample in samples],
    }, indent=2))

    if args.run_eval:
        cmd = [
            sys.executable,
            str(repo / "third_party" / "ho3d_eval" / "eval.py"),
            str(eval_input),
            str(eval_results),
            "--version",
            "v2",
        ]
        subprocess.run(cmd, check=True)

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ho3d-root", type=Path, default=Path("/data/wentao/ropetrack/HO3D_v2_eval"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to run; <=0 means all.")
    parser.add_argument("--mode", choices=["detector", "gt_bbox"], default="detector")
    parser.add_argument("--units", choices=["m", "mm"], default="m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--joint-source", choices=["mano_vertices", "anyhand_keypoints"], default="mano_vertices")
    parser.add_argument("--wilor-ckpt", type=Path, default=None)
    parser.add_argument("--wilor-cfg", type=Path, default=None)
    parser.add_argument("--run-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_export(parse_args())
