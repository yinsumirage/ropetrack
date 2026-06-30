from __future__ import annotations

from math import sqrt
from typing import Sequence


Point3 = Sequence[float]
Joints3 = Sequence[Point3]

FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def point_l2_mm(a: Point3, b: Point3) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def mean_joint_error_mm(pred: Joints3, gt: Joints3) -> float:
    errors = [point_l2_mm(a, b) for a, b in zip(pred, gt, strict=True)]
    if not errors:
        raise ValueError("empty joint list")
    return sum(errors) / len(errors)


def fingertip_error_mm(pred: Joints3, gt: Joints3, tips: Sequence[int] = FINGERTIP_INDICES) -> float:
    return mean_joint_error_mm([pred[i] for i in tips], [gt[i] for i in tips])
