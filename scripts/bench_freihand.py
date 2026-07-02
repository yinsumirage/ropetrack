from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_ho3d import (  # noqa: E402
    BBoxItem,
    load_mano_j_regressor,
    optional_path_str,
    predictor_kwargs,
    pushd,
    resolve_image_path,
    run_bbox_batch_predictions,
    select_sample_predictions,
)

FREIHAND_TIP_VERTEX_IDS = np.asarray([744, 320, 443, 555, 672], dtype=np.int64)
FREIHAND_JOINT_ORDER = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)


@dataclass(frozen=True)
class FreiHandSample:
    sample_id: str
    image_path: Path
    bbox_xyxy: np.ndarray


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def project_points(points, K) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    intrinsics = np.asarray(K, dtype=np.float32)
    uvw = (intrinsics @ pts.T).T
    return uvw[:, :2] / uvw[:, 2:3]


def bbox_from_projected_points(points, K, image_size: int = 224) -> np.ndarray:
    uv = project_points(points, K)
    valid = np.isfinite(uv).all(axis=1)
    if not valid.any():
        raise ValueError("no finite projected points for bbox")
    xy_min = uv[valid].min(axis=0)
    xy_max = uv[valid].max(axis=0)
    xyxy = np.concatenate([xy_min, xy_max]).astype(np.float32)
    return np.clip(xyxy, 0.0, float(image_size))


def iter_freihand_eval_samples(root: Path, limit: int | None) -> Iterable[FreiHandSample]:
    Ks = read_json(root / "evaluation_K.json")
    verts = read_json(root / "evaluation_verts.json")
    if len(Ks) != len(verts):
        raise ValueError(f"FreiHAND eval length mismatch: K={len(Ks)} verts={len(verts)}")
    if limit is not None and limit <= 0:
        limit = None

    for idx, (K, sample_verts) in enumerate(zip(Ks, verts)):
        if limit is not None and idx >= limit:
            break
        frame = f"{idx:08d}"
        yield FreiHandSample(
            sample_id=frame,
            image_path=resolve_image_path(root / "evaluation" / "rgb", frame),
            bbox_xyxy=bbox_from_projected_points(sample_verts, K),
        )


def load_gt_bbox_candidates(samples: list[FreiHandSample]) -> list[BBoxItem]:
    return [
        BBoxItem(
            sample_index=idx,
            bbox_index=0,
            sample=sample,
            bbox_xyxy=sample.bbox_xyxy,
            is_right=True,
            score=1.0,
            source="gt_bbox",
        )
        for idx, sample in enumerate(samples)
    ]


def to_camera(points, cam_t, units: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) + np.asarray(cam_t, dtype=np.float32)[None, :]
    if units == "mm":
        return pts / 1000.0
    if units != "m":
        raise ValueError(f"unsupported units: {units}")
    return pts


def freihand_joints_from_vertices(vertices: np.ndarray, j_regressor: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    joints16 = np.asarray(j_regressor, dtype=np.float32) @ verts
    tips = verts[FREIHAND_TIP_VERTEX_IDS]
    return np.concatenate([joints16, tips], axis=0)[FREIHAND_JOINT_ORDER]


def check_joint_protocol(root: Path, j_regressor: np.ndarray, max_samples: int, tolerance_m: float) -> float:
    xyz = read_json(root / "evaluation_xyz.json")
    verts = read_json(root / "evaluation_verts.json")
    max_err = 0.0
    for gt_xyz, gt_verts in zip(xyz[:max_samples], verts[:max_samples]):
        pred_xyz = freihand_joints_from_vertices(np.asarray(gt_verts, dtype=np.float32), j_regressor)
        err = float(np.linalg.norm(pred_xyz - np.asarray(gt_xyz, dtype=np.float32), axis=1).max())
        max_err = max(max_err, err)
    if max_err > tolerance_m:
        raise ValueError(f"FreiHAND MANO joint protocol mismatch: max_err={max_err:.6f}m")
    return max_err


def run_export(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    anyhand_root = repo / "third_party" / "anyhand"
    sys.path.insert(0, str(anyhand_root))

    from scripts.rgb_predictor import AnyHandPredictor

    samples = list(iter_freihand_eval_samples(args.freihand_root, args.limit))
    candidates = load_gt_bbox_candidates(samples)
    out_dir = args.out_dir
    eval_input = out_dir / "eval_input"
    eval_results = out_dir / "eval_results"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_results.mkdir(parents=True, exist_ok=True)

    j_regressor = load_mano_j_regressor(anyhand_root / "mano_data" / "MANO_RIGHT.pkl")
    protocol_max_err = check_joint_protocol(
        args.freihand_root,
        j_regressor,
        args.protocol_check_samples,
        args.protocol_tolerance_m,
    )
    xyz_pred, verts_pred, failures = [], [], []

    with pushd(anyhand_root):
        predictor = AnyHandPredictor(**predictor_kwargs(args))
        candidate_predictions = run_bbox_batch_predictions(
            predictor,
            args.backend,
            candidates,
            args.batch_size,
            args.num_workers,
        )
        hands_by_sample, missing_failures = select_sample_predictions(samples, candidate_predictions)
        failures.extend(missing_failures)

        for idx, (sample, hand) in enumerate(zip(samples, hands_by_sample)):
            try:
                if hand is None:
                    raise RuntimeError("no hand predicted")
                verts = to_camera(hand.vertices, hand.cam_t, args.units)
                if args.joint_source == "mano_vertices":
                    xyz = freihand_joints_from_vertices(verts, j_regressor)
                else:
                    xyz = to_camera(hand.keypoints_3d, hand.cam_t, args.units)
            except Exception as exc:
                if hand is not None:
                    failures.append({"idx": idx, "sample_id": sample.sample_id, "error": repr(exc)})
                xyz = np.zeros((21, 3), dtype=np.float32)
                verts = np.zeros((778, 3), dtype=np.float32)
            xyz_pred.append(xyz.tolist())
            verts_pred.append(verts.tolist())

    (eval_input / "pred.json").write_text(json.dumps([xyz_pred, verts_pred]))
    for gt_name in ("evaluation_xyz.json", "evaluation_verts.json"):
        shutil.copy2(args.freihand_root / gt_name, eval_input / gt_name)
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "backend": f"anyhand_{args.backend}",
        "dataset": "freihand",
        "mode": args.mode,
        "limit": args.limit,
        "num_samples": len(samples),
        "num_failures": len(failures),
        "num_bbox_candidates": len(candidates),
        "prediction_path": "gt_bbox_cross_image_batch",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "units": args.units,
        "joint_source": args.joint_source,
        "protocol_max_err_m": protocol_max_err,
        "wilor_ckpt": optional_path_str(args.wilor_ckpt),
        "wilor_cfg": optional_path_str(args.wilor_cfg),
        "hamer_ckpt": optional_path_str(args.hamer_ckpt),
        "coordinate_transform": "points + cam_t; output metres; no OpenGL y/z flip for FreiHAND",
        "sample_order": [sample.sample_id for sample in samples],
    }, indent=2))

    if args.run_eval:
        subprocess.run([
            sys.executable,
            str(repo / "scripts" / "eval_parallel.py"),
            str(eval_input),
            str(eval_results),
            "--num-workers",
            str(args.eval_num_workers),
        ], check=True)

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freihand-root", type=Path, default=Path("/data/wentao/ropetrack/FreiHAND"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to run; <=0 means all.")
    parser.add_argument("--mode", choices=["gt_bbox"], default="gt_bbox")
    parser.add_argument("--backend", choices=["wilor", "hamer"], default="wilor")
    parser.add_argument("--units", choices=["m", "mm"], default="m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--joint-source", choices=["mano_vertices", "anyhand_keypoints"], default="mano_vertices")
    parser.add_argument("--protocol-check-samples", type=int, default=32)
    parser.add_argument("--protocol-tolerance-m", type=float, default=1e-3)
    parser.add_argument("--wilor-ckpt", type=Path, default=None)
    parser.add_argument("--wilor-cfg", type=Path, default=None)
    parser.add_argument("--hamer-ckpt", type=Path, default=None)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-num-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run_export(parse_args())
