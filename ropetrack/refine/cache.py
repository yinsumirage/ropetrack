from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_cache(cache: dict[str, np.ndarray], require_target_hand_pose: bool = True) -> None:
    missing = [key for key in ("sample_id", "base_hand_pose", "base_rope_norm", "rope_valid") if key not in cache]
    if "input_rope_norm" not in cache and "gt_rope_norm" not in cache:
        missing.append("input_rope_norm or gt_rope_norm")
    if require_target_hand_pose and "target_hand_pose" not in cache:
        missing.append("target_hand_pose")
    if missing:
        raise ValueError(f"cache missing required keys: {', '.join(missing)}")

    expected = len(cache["sample_id"])
    sample_keys = [
        "base_hand_pose",
        "base_rope_norm",
        "rope_valid",
        "input_rope_norm",
        "gt_rope_norm",
        "target_hand_pose",
    ]
    bad = {key: len(cache[key]) for key in sample_keys if key in cache and len(cache[key]) != expected}
    if bad:
        raise ValueError(f"cache first dimension mismatch: expected {expected}, got {bad}")

    shapes = {
        "base_hand_pose": (expected, 45),
        "base_rope_norm": (expected, 5),
        "rope_valid": (expected, 5),
    }
    if "input_rope_norm" in cache:
        shapes["input_rope_norm"] = (expected, 5)
    if "gt_rope_norm" in cache:
        shapes["gt_rope_norm"] = (expected, 5)
    if require_target_hand_pose or "target_hand_pose" in cache:
        shapes["target_hand_pose"] = (expected, 45)
    for key, shape in shapes.items():
        actual = tuple(np.shape(cache[key]))
        if actual != shape:
            raise ValueError(f"{key} shape must be {shape}, got {actual}")


def make_refiner_features(cache: dict[str, np.ndarray], rope_key: str = "input_rope_norm") -> dict[str, np.ndarray]:
    if rope_key not in cache:
        raise ValueError(f"cache missing rope key: {rope_key}")
    features = {
        "base_hand_pose": np.asarray(cache["base_hand_pose"], dtype=np.float32),
        "base_rope_norm": np.asarray(cache["base_rope_norm"], dtype=np.float32),
        "input_rope_norm": np.asarray(cache[rope_key], dtype=np.float32),
        "rope_valid": np.asarray(cache["rope_valid"], dtype=np.float32),
    }
    if "target_hand_pose" in cache:
        features["target_hand_pose"] = np.asarray(cache["target_hand_pose"], dtype=np.float32)
    valid = features["rope_valid"] > 0
    for key in ("base_rope_norm", "input_rope_norm"):
        rope = features[key].copy()
        if not np.isfinite(rope[valid]).all():
            raise ValueError(f"{key} has non-finite valid rope values")
        rope[~valid] = 0.0
        features[key] = rope
    return features
