from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset: str
    split: str
    image_path: str
    hand_side: str
    bbox_xyxy: BBox
    intrinsics_K: Matrix3 | None = None
    joints3d_camera_mm_path: str | None = None
    joints2d_px_path: str | None = None
    mano_path: str | None = None
    visibility_path: str | None = None
    quality_flag: str = "valid"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    backend: str
    pred_joints3d_mm_path: str
    pred_vertices_mm_path: str | None = None
    pred_mano_path: str | None = None
    pred_joints2d_px_path: str | None = None
    bbox_xyxy: BBox | None = None
    hand_side: str | None = None
    confidence: float | None = None
    coordinate_notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
