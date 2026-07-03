from __future__ import annotations

import numpy as np

HO3D_TIP_VERTEX_IDS = np.asarray([744, 333, 444, 555, 672], dtype=np.int64)
FREIHAND_TIP_VERTEX_IDS = np.asarray([744, 320, 443, 555, 672], dtype=np.int64)
FREIHAND_JOINT_ORDER = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)


def canonical_dataset(dataset: str) -> str:
    name = dataset.lower()
    if name in {"ho3d", "ho3d_v2", "ho3d_v3"}:
        return "ho3d"
    if name == "freihand":
        return "freihand"
    raise ValueError(f"unsupported dataset: {dataset}")


def eval_points_from_model(dataset: str, points, cam_t, units: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32) + np.asarray(cam_t, dtype=np.float32)[None, :]
    if units == "mm":
        pts = pts / 1000.0
    elif units != "m":
        raise ValueError(f"unsupported units: {units}")

    if canonical_dataset(dataset) == "ho3d":
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0
    return pts


def joints_from_vertices(dataset: str, vertices, j_regressor) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    joints16 = np.asarray(j_regressor, dtype=np.float32) @ verts
    if canonical_dataset(dataset) == "freihand":
        joints = np.concatenate([joints16, verts[FREIHAND_TIP_VERTEX_IDS]], axis=0)
        return joints[FREIHAND_JOINT_ORDER]
    return np.concatenate([joints16, verts[HO3D_TIP_VERTEX_IDS]], axis=0)
