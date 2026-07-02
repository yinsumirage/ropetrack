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


@dataclass(frozen=True)
class GtBBoxItem:
    sample: Ho3dSample
    bbox_xyxy: np.ndarray


@dataclass(frozen=True)
class BatchHandPrediction:
    vertices: np.ndarray
    keypoints_3d: np.ndarray
    cam_t: np.ndarray
    score: float


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
            image_path=resolve_image_path(eval_dir / seq / "rgb", frame),
            meta_path=eval_dir / seq / "meta" / f"{frame}.pkl",
        )


def resolve_image_path(rgb_dir: Path, frame: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = rgb_dir / f"{frame}{suffix}"
        if path.exists():
            return path
    return rgb_dir / f"{frame}.png"


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


def load_gt_bbox_items(samples: Iterable[Ho3dSample]) -> list[GtBBoxItem]:
    items = []
    for sample in samples:
        with sample.meta_path.open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        items.append(GtBBoxItem(sample=sample, bbox_xyxy=hand_bbox_from_meta(meta)[0]))
    return items


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


def predictor_kwargs(args: argparse.Namespace) -> dict:
    return {
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "wilor_ckpt": optional_path_str(args.wilor_ckpt),
        "wilor_cfg": optional_path_str(args.wilor_cfg),
        "hamer_ckpt": optional_path_str(args.hamer_ckpt),
    }


def run_backend_with_bbox(predictor, backend: str, img, boxes, is_right, scores):
    if backend == "wilor":
        return predictor._run_wilor(img, boxes, is_right, scores)
    if backend == "hamer":
        return predictor._run_hamer(img, boxes, is_right, scores)
    raise ValueError(f"unsupported backend: {backend}")


def backend_model_and_cfg(predictor, backend: str):
    if backend == "wilor":
        return predictor._wilor_model, predictor._wilor_model_cfg
    if backend == "hamer":
        return predictor._hamer_model, predictor._hamer_model_cfg
    raise ValueError(f"unsupported backend: {backend}")


def backend_dataset_utils(backend: str):
    if backend == "wilor":
        from wilor.datasets import utils
    elif backend == "hamer":
        from hamer.datasets import utils
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return utils


class CrossImageGtBBoxDataset:
    def __init__(self, cfg, items: list[GtBBoxItem], backend: str, rescale_factor: float):
        self.cfg = cfg
        self.items = items
        self.rescale_factor = rescale_factor
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.asarray(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        self.std = 255.0 * np.asarray(cfg.MODEL.IMAGE_STD, dtype=np.float32)
        self.utils = backend_dataset_utils(backend)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import cv2
        from skimage.filters import gaussian

        item = self.items[idx]
        img = cv2.imread(str(item.sample.image_path))
        if img is None:
            raise FileNotFoundError(item.sample.image_path)

        box = item.bbox_xyxy.astype(np.float32)
        center = (box[2:4] + box[0:2]) / 2.0
        scale = self.rescale_factor * (box[2:4] - box[0:2]) / 200.0
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = self.utils.expand_to_aspect_ratio(scale * 200, target_aspect_ratio=bbox_shape).max()
        right = np.float32(1.0)
        flip = False

        cvimg = img.copy()
        downsampling_factor = (float(bbox_size) / float(self.img_size)) / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        img_patch_cv, _ = self.utils.generate_image_patch_cv2(
            cvimg,
            float(center[0]),
            float(center[1]),
            float(bbox_size),
            float(bbox_size),
            int(self.img_size),
            int(self.img_size),
            flip,
            1.0,
            0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch = self.utils.convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])
        for channel in range(min(img.shape[2], 3)):
            img_patch[channel, :, :] = (img_patch[channel, :, :] - self.mean[channel]) / self.std[channel]

        return {
            "img": img_patch,
            "box_center": center.astype(np.float32),
            "box_size": np.float32(bbox_size),
            "img_size": np.asarray([cvimg.shape[1], cvimg.shape[0]], dtype=np.float32),
            "right": right,
            "sample_index": np.int64(idx),
        }


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length):
    import torch

    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def run_gt_bbox_batch_predictions(predictor, backend: str, items: list[GtBBoxItem], batch_size: int, num_workers: int):
    import torch

    model, cfg = backend_model_and_cfg(predictor, backend)
    dataset = CrossImageGtBBoxDataset(cfg, items, backend, predictor.rescale_factor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions: list[BatchHandPrediction | None] = [None] * len(items)

    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(predictor.device) for k, v in batch.items()}
            out = model(batch)
            multiplier = (2 * batch["right"] - 1).float()
            pred_cam = out["pred_cam"].clone()
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            img_size = batch["img_size"].float()
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max(dim=1).values
            cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(), img_size, focal)

            for batch_i, item_i in enumerate(batch["sample_index"].detach().cpu().numpy().astype(int).tolist()):
                verts = out["pred_vertices"][batch_i].detach().cpu().numpy().astype(np.float32)
                joints = out["pred_keypoints_3d"][batch_i].detach().cpu().numpy().astype(np.float32)
                predictions[item_i] = BatchHandPrediction(
                    vertices=verts,
                    keypoints_3d=joints,
                    cam_t=cam_t[batch_i].detach().cpu().numpy().astype(np.float32),
                    score=1.0,
                )

    if any(pred is None for pred in predictions):
        raise RuntimeError("missing batch predictions")
    return predictions


def run_export(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[1]
    anyhand_root = repo / "third_party" / "anyhand"
    sys.path.insert(0, str(anyhand_root))

    from scripts.rgb_predictor import AnyHandPredictor

    samples = list(iter_ho3d_samples(args.ho3d_root, args.limit))
    out_dir = args.out_dir
    eval_input = out_dir / "eval_input"
    eval_results = out_dir / "eval_results"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_results.mkdir(parents=True, exist_ok=True)

    j_regressor = load_mano_j_regressor(anyhand_root / "mano_data" / "MANO_RIGHT.pkl")
    xyz_pred, verts_pred, failures = [], [], []

    with pushd(anyhand_root):
        predictor = AnyHandPredictor(**predictor_kwargs(args))

        if args.mode == "gt_bbox":
            items = load_gt_bbox_items(samples)
            hands_by_sample = run_gt_bbox_batch_predictions(
                predictor,
                args.backend,
                items,
                args.batch_size,
                args.num_workers,
            )
            for idx, (sample, hand) in enumerate(zip(samples, hands_by_sample)):
                try:
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
        else:
            for idx, sample in enumerate(samples):
                try:
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
    for gt_name in ("evaluation_xyz.json", "evaluation_verts.json"):
        gt_path = args.ho3d_root / gt_name
        if gt_path.exists():
            shutil.copy2(gt_path, eval_input / gt_name)
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "backend": f"anyhand_{args.backend}",
        "mode": args.mode,
        "limit": args.limit,
        "num_samples": len(samples),
        "num_failures": len(failures),
        "prediction_path": "gt_bbox_cross_image_batch" if args.mode == "gt_bbox" else "detector_wrapper_serial",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "units": args.units,
        "joint_source": args.joint_source,
        "wilor_ckpt": optional_path_str(args.wilor_ckpt),
        "wilor_cfg": optional_path_str(args.wilor_cfg),
        "hamer_ckpt": optional_path_str(args.hamer_ckpt),
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
    parser.add_argument("--backend", choices=["wilor", "hamer"], default="wilor")
    parser.add_argument("--units", choices=["m", "mm"], default="m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--joint-source", choices=["mano_vertices", "anyhand_keypoints"], default="mano_vertices")
    parser.add_argument("--wilor-ckpt", type=Path, default=None)
    parser.add_argument("--wilor-cfg", type=Path, default=None)
    parser.add_argument("--hamer-ckpt", type=Path, default=None)
    parser.add_argument("--run-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_export(parse_args())
