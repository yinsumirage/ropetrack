from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import (
    BBoxItem,
    bbox_candidates_from_sample,
    iter_eval_samples,
    load_gt_bbox_candidates,
    validate_eval_protocol,
    write_eval_gt_subset,
)
from .protocols import eval_points_from_model, joints_from_vertices


@dataclass(frozen=True)
class BatchHandPrediction:
    candidate: BBoxItem
    vertices: np.ndarray
    keypoints_3d: np.ndarray
    cam_t: np.ndarray

    @property
    def score(self) -> float:
        return self.candidate.score


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def dense_regressor(regressor) -> np.ndarray:
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    return np.asarray(regressor, dtype=np.float32)


def load_mano_j_regressor(mano_path: Path) -> np.ndarray:
    with mano_path.open("rb") as f:
        mano = pickle.load(f, encoding="latin1")
    return dense_regressor(mano["J_regressor"])


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


class CrossImageBBoxDataset:
    def __init__(self, cfg, candidates: list[BBoxItem], backend: str, rescale_factor: float):
        self.cfg = cfg
        self.candidates = candidates
        self.rescale_factor = rescale_factor
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255.0 * np.asarray(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
        self.std = 255.0 * np.asarray(cfg.MODEL.IMAGE_STD, dtype=np.float32)
        self.utils = backend_dataset_utils(backend)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int):
        import cv2
        from skimage.filters import gaussian

        candidate = self.candidates[idx]
        img = cv2.imread(str(candidate.sample.image_path))
        if img is None:
            raise FileNotFoundError(candidate.sample.image_path)

        box = candidate.bbox_xyxy.astype(np.float32)
        center = (box[2:4] + box[0:2]) / 2.0
        scale = self.rescale_factor * (box[2:4] - box[0:2]) / 200.0
        bbox_shape = self.cfg.MODEL.get("BBOX_SHAPE", None)
        bbox_size = self.utils.expand_to_aspect_ratio(scale * 200, target_aspect_ratio=bbox_shape).max()
        right = np.float32(1.0 if candidate.is_right else 0.0)
        flip = not candidate.is_right

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
            "candidate_index": np.int64(idx),
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


def run_bbox_batch_predictions(predictor, backend: str, candidates: list[BBoxItem], batch_size: int, num_workers: int):
    import torch

    if not candidates:
        return []
    model, cfg = backend_model_and_cfg(predictor, backend)
    dataset = CrossImageBBoxDataset(cfg, candidates, backend, predictor.rescale_factor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions: list[BatchHandPrediction | None] = [None] * len(candidates)

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

            for batch_i, candidate_i in enumerate(batch["candidate_index"].detach().cpu().numpy().astype(int).tolist()):
                predictions[candidate_i] = BatchHandPrediction(
                    candidate=candidates[candidate_i],
                    vertices=out["pred_vertices"][batch_i].detach().cpu().numpy().astype(np.float32),
                    keypoints_3d=out["pred_keypoints_3d"][batch_i].detach().cpu().numpy().astype(np.float32),
                    cam_t=cam_t[batch_i].detach().cpu().numpy().astype(np.float32),
                )

    if any(pred is None for pred in predictions):
        raise RuntimeError("missing batch predictions")
    return predictions


def detect_bbox_candidates_batched(predictor, samples: list, detector_batch_size: int) -> list[BBoxItem]:
    import cv2

    candidates: list[BBoxItem] = []
    for start in range(0, len(samples), detector_batch_size):
        batch_samples = samples[start:start + detector_batch_size]
        images = []
        for sample in batch_samples:
            img = cv2.imread(str(sample.image_path))
            if img is None:
                raise FileNotFoundError(sample.image_path)
            images.append(img)

        results = predictor._detector(images, verbose=False, conf=predictor.det_conf, iou=predictor.det_iou)
        for offset, result in enumerate(results):
            sample_index = start + offset
            sample = samples[sample_index]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            candidates.extend(bbox_candidates_from_sample(
                sample_index,
                sample,
                result.boxes.xyxy.cpu().numpy().astype(np.float32),
                result.boxes.cls.cpu().numpy().astype(int) == 1,
                result.boxes.conf.cpu().numpy().astype(np.float32),
                "detector",
            ))
    return candidates


def select_sample_predictions(samples: list, predictions: list[BatchHandPrediction]):
    grouped: list[list[BatchHandPrediction]] = [[] for _ in samples]
    for pred in predictions:
        grouped[pred.candidate.sample_index].append(pred)

    selected: list[BatchHandPrediction | None] = []
    failures = []
    for idx, sample_preds in enumerate(grouped):
        if not sample_preds:
            selected.append(None)
            failures.append({"idx": idx, "sample_id": samples[idx].sample_id, "error": "RuntimeError('no hand detected')"})
        else:
            selected.append(max(sample_preds, key=lambda pred: pred.score))
    return selected, failures


def format_prediction(dataset: str, hand: BatchHandPrediction, j_regressor: np.ndarray, joint_source: str, units: str):
    verts = eval_points_from_model(dataset, hand.vertices, hand.cam_t, units)
    if joint_source in {"model_keypoints", "anyhand_keypoints"}:
        xyz = eval_points_from_model(dataset, hand.keypoints_3d, hand.cam_t, units)
    elif joint_source == "mano_vertices":
        xyz = joints_from_vertices(dataset, verts, j_regressor)
    else:
        raise ValueError(f"unsupported joint_source: {joint_source}")
    return xyz, verts


def run_export(args: argparse.Namespace) -> Path:
    repo = Path(__file__).resolve().parents[2]
    from ropetrack.backends.hand_predictor import HandPredictor

    root = args.freihand_root if args.adapter == "freihand" else args.ho3d_root
    samples = list(iter_eval_samples(args.adapter, root, args.limit))
    validate_eval_protocol(args.adapter, root, samples, args.protocol_check_samples, args.protocol_tolerance_m)
    out_dir = args.out_dir
    eval_input = out_dir / "eval_input"
    eval_results = out_dir / "eval_results"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_results.mkdir(parents=True, exist_ok=True)

    j_regressor = load_mano_j_regressor(HandPredictor.DEFAULT_MANO_RIGHT)
    predictor = HandPredictor(**predictor_kwargs(args))
    if args.mode == "gt_bbox":
        candidates = load_gt_bbox_candidates(args.adapter, samples)
    else:
        candidates = detect_bbox_candidates_batched(predictor, samples, args.detector_batch_size)
    candidate_predictions = run_bbox_batch_predictions(predictor, args.backend, candidates, args.batch_size, args.num_workers)
    hands_by_sample, failures = select_sample_predictions(samples, candidate_predictions)
    xyz_pred, verts_pred = [], []

    for idx, (sample, hand) in enumerate(zip(samples, hands_by_sample)):
        try:
            if hand is None:
                raise RuntimeError("no hand predicted")
            xyz, verts = format_prediction(args.adapter, hand, j_regressor, args.joint_source, args.units)
        except Exception as exc:
            if hand is not None:
                failures.append({"idx": idx, "sample_id": sample.sample_id, "error": repr(exc)})
            xyz = np.zeros((21, 3), dtype=np.float32)
            verts = np.zeros((778, 3), dtype=np.float32)
        xyz_pred.append(xyz.tolist())
        verts_pred.append(verts.tolist())

    (eval_input / "pred.json").write_text(json.dumps([xyz_pred, verts_pred]))
    gt_dir = root
    if args.limit is not None and args.limit > 0:
        write_eval_gt_subset(args.adapter, root, eval_input, len(samples))
        gt_dir = eval_input
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "run_meta.json").write_text(json.dumps({
        "dataset": args.dataset,
        "adapter": args.adapter,
        "method": args.method,
        "backend": args.backend,
        "mode": args.mode,
        "limit": args.limit,
        "num_samples": len(samples),
        "num_failures": len(failures),
        "num_bbox_candidates": len(candidates),
        "prediction_path": f"{args.mode}_cross_image_batch",
        "batch_size": args.batch_size,
        "detector_batch_size": args.detector_batch_size,
        "num_workers": args.num_workers,
        "units": args.units,
        "joint_source": args.joint_source,
        "wilor_ckpt": optional_path_str(args.wilor_ckpt),
        "wilor_cfg": optional_path_str(args.wilor_cfg),
        "hamer_ckpt": optional_path_str(args.hamer_ckpt),
        "sample_order": [sample.sample_id for sample in samples],
    }, indent=2))

    if args.run_eval:
        subprocess.run([
            sys.executable,
            str(repo / "scripts" / "score_predictions.py"),
            str(eval_input),
            str(eval_results),
            "--gt-dir",
            str(gt_dir),
            "--num-workers",
            str(args.eval_num_workers),
        ], check=True)

    return out_dir
