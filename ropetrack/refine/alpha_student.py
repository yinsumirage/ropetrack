"""Amortized rope-optimization student (P2 distillation).

The test-time optimizer (the teacher) needs ~400 MANO forward/backward
passes per sample. The student is a small MLP trained to predict the
teacher's output alphas from the same inputs in one forward pass.

Why this avoids the 0026 refiner failure: the output stays in the
rope-observable alpha subspace (5/15 dims, same tanh * max_alpha bound as
the teacher), the target is the deterministic optimizer output rather than
GT pose, and training carries validation/early-stopping, sensor-noise
augmentation, and a shuffled-rope control.

Feature layout (STUDENT_FEATURE_KEYS order):
    base_hand_pose (45) | base_rope_norm (5) | input_rope_norm (5) |
    residual = input - base (5) | rope_valid (5)   -> 65 dims
Invalid fingers have their rope features zeroed, mirroring the cache
convention. The hard residual gate is NOT learned - it stays a rule applied
on top of the student's alphas at inference, same as for the teacher.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

STUDENT_FEATURE_DIM = 65


def build_student_features(
    base_hand_pose: np.ndarray,
    base_rope_norm: np.ndarray,
    input_rope_norm: np.ndarray,
    rope_valid: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(base_hand_pose, dtype=np.float32)
    base_rope = np.asarray(base_rope_norm, dtype=np.float32).copy()
    input_rope = np.asarray(input_rope_norm, dtype=np.float32).copy()
    valid = np.asarray(rope_valid, dtype=bool)
    if pose.ndim != 2 or pose.shape[1] != 45:
        raise ValueError(f"base_hand_pose must be [N, 45], got {pose.shape}")
    for name, rope in (("base_rope_norm", base_rope), ("input_rope_norm", input_rope)):
        if rope.shape != valid.shape or rope.shape != (pose.shape[0], 5):
            raise ValueError(f"{name} must be [N, 5] matching rope_valid, got {rope.shape}")
    base_rope[~valid] = 0.0
    input_rope[~valid] = 0.0
    residual = input_rope - base_rope
    residual[~valid] = 0.0
    return np.concatenate(
        [pose, base_rope, input_rope, residual, valid.astype(np.float32)], axis=1
    )


def features_from_cache(cache: dict[str, np.ndarray]) -> np.ndarray:
    return build_student_features(
        cache["base_hand_pose"],
        cache["base_rope_norm"],
        cache["input_rope_norm"],
        cache["rope_valid"],
    )


class RopeAlphaStudent(nn.Module):
    def __init__(self, out_dim: int, hidden_dim: int = 256, max_alpha: float = 0.5, in_dim: int = STUDENT_FEATURE_DIM) -> None:
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        final = self.net[-1]
        # zero init: the untrained student applies no correction
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_alpha * torch.tanh(self.net(features))


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(features, dtype=np.float32) - mean) / std


def feature_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-4).astype(np.float32)
    return mean, std


def save_student_checkpoint(path: Path, model: RopeAlphaStudent, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def student_from_payload(payload: dict, device: str) -> tuple[RopeAlphaStudent, dict]:
    config = payload["config"]
    model = RopeAlphaStudent(
        out_dim=int(config["out_dim"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        max_alpha=float(config.get("max_alpha", 0.5)),
        in_dim=int(config.get("in_dim", STUDENT_FEATURE_DIM)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, config


def load_student(path: Path, device: str) -> tuple[RopeAlphaStudent, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    return student_from_payload(payload, device)


def load_image_feature_cache(path: Path) -> tuple[list[str], np.ndarray]:
    """P3 feature cache from scripts/rope_head/extract_feature_cache.py."""
    with np.load(path) as data:
        ids = [str(sid) for sid in data["sample_id"]]
        features = np.asarray(data["features"], dtype=np.float32)
    if len(ids) != len(features):
        raise ValueError(f"feature cache rows mismatch: ids={len(ids)} features={len(features)}")
    return ids, features


def join_image_features(sample_ids, feature_ids: list[str], features: np.ndarray) -> np.ndarray:
    """Reorder cached image features to sample_ids; strict on missing rows."""
    from ropetrack.refine.cache import align_rows_by_sample_id

    return features[align_rows_by_sample_id(sample_ids, feature_ids)]


def student_alpha(
    cache: dict[str, np.ndarray],
    checkpoint: Path,
    device: str,
    image_features: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Predict alphas for a refiner eval cache; the gate is applied by the caller.

    ``image_features`` (rows aligned to ``cache['sample_id']``) is required
    iff the checkpoint was trained with them (P3 rope-conditioned head).
    """
    model, config = load_student(checkpoint, device)
    features = features_from_cache(cache)
    expected_image_dim = int(config.get("image_feature_dim", 0))
    if expected_image_dim:
        if image_features is None:
            raise ValueError("checkpoint was trained with image features; pass --feature-cache")
        image_features = np.asarray(image_features, dtype=np.float32)
        if image_features.shape != (features.shape[0], expected_image_dim):
            raise ValueError(
                f"image features must be {(features.shape[0], expected_image_dim)}, got {image_features.shape}"
            )
        features = np.concatenate([features, image_features], axis=1)
    elif image_features is not None:
        raise ValueError("checkpoint was trained without image features; do not pass --feature-cache")
    mean = np.asarray(config["feature_mean"], dtype=np.float32)
    std = np.asarray(config["feature_std"], dtype=np.float32)
    with torch.no_grad():
        alpha = model(torch.from_numpy(normalize_features(features, mean, std)).to(device))
    return alpha.cpu().numpy().astype(np.float32), config
