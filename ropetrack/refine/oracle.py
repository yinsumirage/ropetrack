"""Oracle objectives for rope optimization ceiling probes.

The oracle swaps the rope-only objective for direct GT joint supervision
while keeping the action space fixed. This measures how much error the
action space itself can remove (its ceiling), independent of how much the
5 rope scalars observe. See docs/2026-07-06-rope-refinement-next-plan.md.

The predicted joints are decoded with the exact scoring convention
(``ropetrack.eval.protocols``): vertices -> eval frame -> J_regressor
16 joints + tip vertices (+ FreiHAND reorder). The loss is wrist-relative
because the action spaces cannot change global orientation or translation,
and reported in cm^2 so its magnitude is comparable to the rope-norm loss
(the published lr regime stays a sane starting point).
"""

from __future__ import annotations

import numpy as np
import torch

from ropetrack.eval.protocols import (
    FREIHAND_JOINT_ORDER,
    FREIHAND_TIP_VERTEX_IDS,
    DEXYCB_TIP_VERTEX_IDS,
    HO3D_TIP_VERTEX_IDS,
    canonical_dataset,
)

# Fingertip ids in the 21-joint eval convention (thumb..pinky order).
FREIHAND_TIP_JOINT_IDS = (4, 8, 12, 16, 20)
HO3D_TIP_JOINT_IDS = (16, 17, 18, 19, 20)
WRIST_JOINT_ID = 0

ORACLE_OBJECTIVES = ("oracle_tip", "oracle_chain")


def oracle_joint_ids(dataset: str, objective: str) -> list[int]:
    ds = canonical_dataset(dataset)
    if objective == "oracle_tip":
        return list(FREIHAND_TIP_JOINT_IDS if ds in {"freihand", "egodex", "arctic", "hot3d", "dexycb", "interhand26m"} else HO3D_TIP_JOINT_IDS)
    if objective == "oracle_chain":
        return [joint for joint in range(21) if joint != WRIST_JOINT_ID]
    raise ValueError(f"unsupported oracle objective: {objective}")


def torch_eval_points_from_model(dataset: str, points: torch.Tensor, cam_t: torch.Tensor) -> torch.Tensor:
    """Differentiable version of protocols.eval_points_from_model (units=m).

    points: [B, V, 3] model-frame meters; cam_t: [B, 3].
    """
    pts = points + cam_t[:, None, :]
    if canonical_dataset(dataset) == "ho3d":
        flip = torch.tensor([1.0, -1.0, -1.0], device=pts.device, dtype=pts.dtype)
        pts = pts * flip
    return pts


def torch_eval_joints_from_vertices(dataset: str, verts_eval: torch.Tensor, j_regressor: torch.Tensor) -> torch.Tensor:
    """Differentiable version of protocols.joints_from_vertices.

    verts_eval: [B, 778, 3]; j_regressor: [16, 778]. Returns [B, 21, 3].
    """
    joints16 = torch.einsum("jv,bvc->bjc", j_regressor, verts_eval)
    ds = canonical_dataset(dataset)
    if ds in {"arctic", "hot3d", "interhand26m"}:
        raise ValueError(f"{ds.upper()} joints require MANO kinematic model_keypoints")
    if ds == "dexycb":
        tip_ids = DEXYCB_TIP_VERTEX_IDS
    else:
        tip_ids = FREIHAND_TIP_VERTEX_IDS if ds in {"freihand", "egodex"} else HO3D_TIP_VERTEX_IDS
    tips = verts_eval[:, torch.as_tensor(np.asarray(tip_ids), device=verts_eval.device)]
    joints = torch.cat([joints16, tips], dim=1)
    if ds in {"freihand", "egodex", "dexycb"}:
        order = torch.as_tensor(np.asarray(FREIHAND_JOINT_ORDER), device=joints.device)
        joints = joints[:, order]
    return joints


def wrist_relative(joints21: torch.Tensor, wrist_id: int = WRIST_JOINT_ID) -> torch.Tensor:
    return joints21 - joints21[:, wrist_id : wrist_id + 1]


def oracle_loss_cm2(pred_joints21: torch.Tensor, gt_joints21: torch.Tensor, joint_ids: list[int]) -> torch.Tensor:
    """Mean squared wrist-relative joint error over joint_ids, in cm^2."""
    ids = torch.as_tensor(joint_ids, device=pred_joints21.device)
    diff = (wrist_relative(pred_joints21) - wrist_relative(gt_joints21))[:, ids] * 100.0
    return (diff * diff).mean()
