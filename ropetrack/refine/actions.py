"""Action spaces for per-sample rope pose correction.

An action space maps a low-dimensional alpha vector to a MANO hand_pose
(45-dim axis-angle) correction:

- ``mult5``: one multiplicative curl scalar per finger. The finger's pose
  triplets are scaled by ``(1 + alpha_f)``. This is the original probe space
  from experience/0027; it cannot create curl on a near-zero base pose.
- ``mult15``: one multiplicative scalar per MANO finger joint (15 joints).
- ``flex15``: one additive scalar per MANO finger joint, applied along a
  precomputed per-sample unit direction in that joint's axis-angle space
  (the frozen rope-distance gradient direction, see
  ``compute_flex_directions`` in ``ropetrack.refine.apply``).
  Additive corrections can close a finger that the backbone predicted open,
  which multiplicative scaling cannot.
- ``flex5``: one additive scalar per finger, applied along the finger's
  coupled 9-dim gradient direction (normalized over the whole finger, not
  per joint). Matched capacity: the rope gives one constraint per finger,
  so one dof per finger leaves no within-finger null space, while the
  additive form still fixes the multiplicative dead zone.
- ``pose45``: direct additive 45-dim axis-angle delta on ``hand_pose``.

Alpha column conventions:

- ``mult5`` / ``flex5``: columns follow rope ``FINGER_ORDER`` (thumb, index,
  middle, ring, pinky).
- ``mult15`` / ``flex15``: column ``j`` corresponds to MANO hand joint ``j``
  (pose dims ``3j..3j+2``). Use ``JOINT_TO_FINGER`` to aggregate per finger.
- ``pose45``: columns are raw hand_pose dims ``0..44``.

Only numpy is imported at module level so numpy-only analysis scripts can
use the constants; torch is imported lazily inside torch functions.
"""

from __future__ import annotations

import numpy as np

# MANO hand_pose joint triplets per finger, in rope FINGER_ORDER
# (thumb, index, middle, ring, pinky). Matches experience/0027.
FINGER_POSE_GROUPS = (
    (12, 13, 14),  # thumb
    (0, 1, 2),     # index
    (3, 4, 5),     # middle
    (9, 10, 11),   # ring
    (6, 7, 8),     # pinky
)

ACTION_SPACES = ("mult5", "mult15", "flex15", "flex5", "pose45")
FLEX_ACTION_SPACES = ("flex15", "flex5")

# MANO hand joint id (0..14) -> finger index (0..4) in FINGER_ORDER.
JOINT_TO_FINGER = np.zeros(15, dtype=np.int64)
for _finger_idx, _joints in enumerate(FINGER_POSE_GROUPS):
    for _joint in _joints:
        JOINT_TO_FINGER[_joint] = _finger_idx


def alpha_dim(action_space: str) -> int:
    if action_space in {"mult5", "flex5"}:
        return 5
    if action_space in {"mult15", "flex15"}:
        return 15
    if action_space == "pose45":
        return 45
    raise ValueError(f"unsupported action space: {action_space}")


def _check_alpha(alpha_shape: tuple[int, ...], action_space: str) -> None:
    expected = alpha_dim(action_space)
    if len(alpha_shape) != 2 or alpha_shape[1] != expected:
        raise ValueError(
            f"alpha shape must be [N, {expected}] for {action_space}, got {tuple(alpha_shape)}"
        )


def apply_action_np(
    base_hand_pose: np.ndarray,
    alpha: np.ndarray,
    action_space: str,
    directions: np.ndarray | None = None,
) -> np.ndarray:
    base = np.asarray(base_hand_pose, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    _check_alpha(alpha.shape, action_space)

    if action_space in {"mult5", "mult15"}:
        per_joint = alpha[:, JOINT_TO_FINGER] if action_space == "mult5" else alpha
        scale = np.repeat(per_joint, 3, axis=1)
        return base + scale * base
    if action_space == "pose45":
        return base + alpha

    if directions is None:
        raise ValueError(f"{action_space} requires per-sample directions [N, 15, 3]")
    dirs = np.asarray(directions, dtype=np.float32)
    if dirs.shape != (base.shape[0], 15, 3):
        raise ValueError(f"directions shape must be {(base.shape[0], 15, 3)}, got {dirs.shape}")
    per_joint = alpha[:, JOINT_TO_FINGER] if action_space == "flex5" else alpha
    delta = (per_joint[:, :, None] * dirs).reshape(base.shape[0], 45)
    return base + delta


def apply_action_torch(base_hand_pose, alpha, action_space: str, directions=None):
    import torch

    _check_alpha(tuple(alpha.shape), action_space)

    if action_space in {"mult5", "mult15"}:
        if action_space == "mult5":
            index = torch.as_tensor(JOINT_TO_FINGER, device=alpha.device)
            per_joint = alpha[:, index]
        else:
            per_joint = alpha
        scale = per_joint.repeat_interleave(3, dim=1)
        return base_hand_pose + scale * base_hand_pose
    if action_space == "pose45":
        return base_hand_pose + alpha

    if directions is None:
        raise ValueError(f"{action_space} requires per-sample directions [N, 15, 3]")
    if action_space == "flex5":
        index = torch.as_tensor(JOINT_TO_FINGER, device=alpha.device)
        per_joint = alpha[:, index]
    else:
        per_joint = alpha
    delta = (per_joint.unsqueeze(-1) * directions).reshape(base_hand_pose.shape[0], 45)
    return base_hand_pose + delta


def per_finger_alpha_abs(alpha: np.ndarray, action_space: str) -> np.ndarray:
    """Aggregate |alpha| to [N, 5] finger columns in FINGER_ORDER."""
    alpha = np.abs(np.asarray(alpha, dtype=np.float32))
    _check_alpha(alpha.shape, action_space)
    if action_space in {"mult5", "flex5"}:
        return alpha
    if action_space == "pose45":
        out = np.zeros((alpha.shape[0], 5), dtype=np.float32)
        for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
            dims = [3 * joint + axis for joint in joints for axis in range(3)]
            out[:, finger_idx] = alpha[:, dims].mean(axis=1)
        return out
    out = np.zeros((alpha.shape[0], 5), dtype=np.float32)
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        out[:, finger_idx] = alpha[:, list(joints)].mean(axis=1)
    return out


def per_finger_pose_magnitude(base_hand_pose: np.ndarray) -> np.ndarray:
    """L2 norm of each finger's 9 pose dims, [N, 5] in FINGER_ORDER."""
    base = np.asarray(base_hand_pose, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 45:
        raise ValueError(f"base_hand_pose shape must be [N, 45], got {base.shape}")
    out = np.zeros((base.shape[0], 5), dtype=np.float32)
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        dims = [3 * joint + axis for joint in joints for axis in range(3)]
        out[:, finger_idx] = np.linalg.norm(base[:, dims], axis=1)
    return out
