from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _torch_load_trusted():
    orig = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


@dataclass
class HandPrediction:
    mano_pose: np.ndarray
    mano_shape: np.ndarray
    vertices: np.ndarray
    keypoints_3d: np.ndarray
    keypoints_2d: np.ndarray
    cam_t: np.ndarray
    focal_length: float
    bbox: np.ndarray
    is_right: bool
    score: float
    backend: str


class HandPredictor:
    DEFAULT_MANO_RIGHT = REPO_ROOT / "mano_data" / "MANO_RIGHT.pkl"
    DEFAULT_WILOR_CKPT = REPO_ROOT / "pretrained_models" / "anyhand_wilor.ckpt"
    DEFAULT_WILOR_CFG = REPO_ROOT / "pretrained_models" / "model_config_wilor.yaml"
    DEFAULT_HAMER_CKPT = REPO_ROOT / "pretrained_models" / "hamer_ckpts" / "checkpoints" / "anyhand_hamer.ckpt"
    DEFAULT_DETECTOR_PT = REPO_ROOT / "pretrained_models" / "detector.pt"

    def __init__(
        self,
        backend: Literal["wilor", "hamer", "both"] = "wilor",
        device: str | None = None,
        wilor_ckpt: str | None = None,
        wilor_cfg: str | None = None,
        hamer_ckpt: str | None = None,
        detector_pt: str | None = None,
        det_conf: float = 0.3,
        det_iou: float = 0.3,
        rescale_factor: float = 2.0,
        batch_size: int = 16,
    ) -> None:
        if backend not in ("wilor", "hamer", "both"):
            raise ValueError(f"backend must be 'wilor', 'hamer', or 'both'. Got: {backend!r}")

        self.backend = backend
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.rescale_factor = rescale_factor
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._wilor_ckpt = wilor_ckpt or str(self.DEFAULT_WILOR_CKPT)
        self._wilor_cfg = wilor_cfg or str(self.DEFAULT_WILOR_CFG)
        self._hamer_ckpt = hamer_ckpt or str(self.DEFAULT_HAMER_CKPT)
        self._detector_pt = detector_pt or str(self.DEFAULT_DETECTOR_PT)

        self._wilor_model = None
        self._wilor_model_cfg = None
        self._hamer_model = None
        self._hamer_model_cfg = None

        self._load_detector()
        if backend in ("wilor", "both"):
            self._load_wilor()
        if backend in ("hamer", "both"):
            self._load_hamer()

    def _load_detector(self) -> None:
        from ultralytics import YOLO

        _check_file(self._detector_pt, "Link pretrained_models/detector.pt under the ropetrack repo root.")
        with _torch_load_trusted():
            self._detector = YOLO(self._detector_pt).to(self.device)

    def _load_wilor(self) -> None:
        _ensure_on_path(REPO_ROOT / "third_party" / "wilor", "Run: git submodule update --init third_party/wilor")
        _check_file(self._wilor_ckpt, "Link pretrained_models/ under the ropetrack repo root.")

        from wilor.models import load_wilor

        model, cfg = load_wilor(checkpoint_path=self._wilor_ckpt, cfg_path=self._wilor_cfg)
        self._wilor_model = model.to(self.device).eval()
        self._wilor_model_cfg = cfg

    def _load_hamer(self) -> None:
        _ensure_on_path(REPO_ROOT / "third_party" / "hamer", "Run: git submodule update --init --recursive third_party/hamer")
        _check_file(self._hamer_ckpt, "Link pretrained_models/ under the ropetrack repo root.")

        ckpt_dir = Path(self._hamer_ckpt).parent
        cfg_path = ckpt_dir / "model_config.yaml"
        _check_file(str(cfg_path), f"Expected HaMeR config next to checkpoint: {cfg_path}")

        from hamer.models.hamer import HAMER
        from omegaconf import OmegaConf

        model_cfg = OmegaConf.load(str(cfg_path))
        model = HAMER.load_from_checkpoint(self._hamer_ckpt, strict=False, cfg=model_cfg)
        self._hamer_model = model.to(self.device).eval()
        self._hamer_model_cfg = model_cfg

    def _detect_hands(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = self._detector(img_bgr, verbose=False, conf=self.det_conf, iou=self.det_iou)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.array([], dtype=bool),
                np.array([], dtype=np.float32),
            )

        det = results[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls = det.boxes.cls.cpu().numpy().astype(int)
        scores = det.boxes.conf.cpu().numpy().astype(np.float32)
        return boxes, cls == 1, scores

    def _run_wilor(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
    ) -> list[HandPrediction]:
        from wilor.datasets.vitdet_dataset import ViTDetDataset

        return self._run_model(
            ViTDetDataset,
            self._wilor_model,
            self._wilor_model_cfg,
            img_bgr,
            boxes,
            is_right,
            scores,
            "wilor",
        )

    def _run_hamer(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
    ) -> list[HandPrediction]:
        from hamer.datasets.vitdet_dataset import ViTDetDataset

        return self._run_model(
            ViTDetDataset,
            self._hamer_model,
            self._hamer_model_cfg,
            img_bgr,
            boxes,
            is_right,
            scores,
            "hamer",
        )

    def _run_model(
        self,
        dataset_cls,
        model,
        model_cfg,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
        backend: str,
    ) -> list[HandPrediction]:
        width, height = img_bgr.shape[1], img_bgr.shape[0]
        img_size = torch.tensor([width, height], dtype=torch.float32, device=self.device)
        scaled_focal = float(model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(width, height))
        dataset = dataset_cls(model_cfg, img_bgr, boxes, is_right, rescale_factor=self.rescale_factor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return self._collect_predictions(loader, model, boxes, is_right, scores, img_size, scaled_focal, backend)

    def _collect_predictions(
        self,
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        boxes: np.ndarray,
        is_right: np.ndarray,
        scores: np.ndarray,
        img_size: torch.Tensor,
        scaled_focal: float,
        backend: str,
    ) -> list[HandPrediction]:
        preds = []
        hand_idx = 0

        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                out = model(batch)
                multiplier = (2 * batch["right"] - 1).float()
                pred_cam = out["pred_cam"].clone()
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                cam_t_full = _cam_crop_to_full(
                    pred_cam,
                    batch["box_center"].float(),
                    batch["box_size"].float(),
                    img_size,
                    scaled_focal,
                )
                mano_params = out["pred_mano_params"]
                global_orient = mano_params["global_orient"]
                hand_pose = mano_params["hand_pose"]
                betas = mano_params["betas"]
                pred_verts = out["pred_vertices"]
                pred_kp3d = out["pred_keypoints_3d"]
                pred_kp2d = out["pred_keypoints_2d"]

                for batch_i in range(pred_cam.shape[0]):
                    flip = float(2 * float(batch["right"][batch_i].item()) - 1)
                    verts_np = pred_verts[batch_i].cpu().numpy().astype(np.float32)
                    joints_np = pred_kp3d[batch_i].cpu().numpy().astype(np.float32)
                    verts_np[:, 0] = flip * verts_np[:, 0]
                    joints_np[:, 0] = flip * joints_np[:, 0]
                    kp2d_px = _kp2d_crop_to_full(
                        pred_kp2d[batch_i].cpu().numpy(),
                        batch["box_center"][batch_i].cpu().numpy(),
                        float(batch["box_size"][batch_i].cpu()),
                        model_size=int(batch["img"].shape[-1]),
                    )

                    preds.append(HandPrediction(
                        mano_pose=np.concatenate([
                            _rotmat_to_aa(global_orient[batch_i].cpu().numpy()),
                            _rotmat_to_aa(hand_pose[batch_i].cpu().numpy()),
                        ]).astype(np.float32),
                        mano_shape=betas[batch_i].cpu().numpy().astype(np.float32),
                        vertices=verts_np,
                        keypoints_3d=joints_np,
                        keypoints_2d=kp2d_px.astype(np.float32),
                        cam_t=cam_t_full[batch_i].cpu().numpy().astype(np.float32),
                        focal_length=scaled_focal,
                        bbox=boxes[hand_idx].copy(),
                        is_right=bool(is_right[hand_idx]),
                        score=float(scores[hand_idx]),
                        backend=backend,
                    ))
                    hand_idx += 1

        return preds

    def predict(
        self,
        image: np.ndarray | str | Path | list[np.ndarray | str | Path],
        backend: Literal["wilor", "hamer", "both"] | None = None,
    ):
        backend = backend or self.backend
        if isinstance(image, list):
            return [self._predict_single(item, backend) for item in image]
        return self._predict_single(image, backend)

    def _predict_single(self, image: np.ndarray | str | Path, backend: str) -> list[HandPrediction]:
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        elif isinstance(image, np.ndarray):
            img_bgr = image
        else:
            raise TypeError(f"image must be str, Path, or np.ndarray. Got {type(image)}")

        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) BGR array. Got shape: {img_bgr.shape}")

        boxes, is_right, scores = self._detect_hands(img_bgr)
        if len(boxes) == 0:
            return []

        results = []
        if backend in ("wilor", "both"):
            results += self._run_wilor(img_bgr, boxes, is_right, scores)
        if backend in ("hamer", "both"):
            results += self._run_hamer(img_bgr, boxes, is_right, scores)
        return results

    @staticmethod
    def project_3d_to_2d(points: np.ndarray, cam_t: np.ndarray, focal_length: float, img_size: tuple[int, int]) -> np.ndarray:
        width, height = img_size
        pts = points + cam_t[None, :]
        pts_norm = pts / pts[:, 2:3]
        return np.stack([
            focal_length * pts_norm[:, 0] + width / 2.0,
            focal_length * pts_norm[:, 1] + height / 2.0,
        ], axis=1).astype(np.float32)


def _cam_crop_to_full(
    pred_cam: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    img_size: torch.Tensor,
    focal_length: float,
) -> torch.Tensor:
    denom = pred_cam[:, 0] * box_size + 1e-9
    tx = pred_cam[:, 1] + 2.0 * (box_center[:, 0] - img_size[0] * 0.5) / denom
    ty = pred_cam[:, 2] + 2.0 * (box_center[:, 1] - img_size[1] * 0.5) / denom
    tz = 2.0 * focal_length / denom
    return torch.stack([tx, ty, tz], dim=1)


def _rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    mats = rotmat.reshape(int(np.prod(rotmat.shape[:-2])), 3, 3)
    u, _, vt = np.linalg.svd(mats)
    mats_orth = u @ vt
    mats_orth[:, :, 2:] *= np.sign(np.linalg.det(mats_orth))[:, None, None]
    return Rotation.from_matrix(mats_orth).as_rotvec().astype(np.float32).reshape(-1)


def _kp2d_crop_to_full(kp2d_norm: np.ndarray, box_center: np.ndarray, box_size: float, model_size: int = 256) -> np.ndarray:
    _ = model_size
    kp = (kp2d_norm + 1.0) * 0.5 * box_size
    kp[:, 0] += box_center[0] - box_size * 0.5
    kp[:, 1] += box_center[1] - box_size * 0.5
    return kp.astype(np.float32)


def _check_file(path: str, hint: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}\n  -> {hint}")


def _ensure_on_path(pkg_dir: Path, hint: str) -> None:
    if not pkg_dir.exists():
        raise ImportError(f"Package directory not found: {pkg_dir}\n  -> {hint}")
    src = str(pkg_dir)
    if src not in sys.path:
        sys.path.insert(0, src)
