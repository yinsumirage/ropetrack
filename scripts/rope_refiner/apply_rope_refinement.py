from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.eval.pipeline import load_mano_j_regressor
from ropetrack.eval.protocols import eval_points_from_model, joints_from_vertices
from ropetrack.io import read_jsonl
from ropetrack.refine.actions import (
    ACTION_SPACES,
    FINGER_POSE_GROUPS,
    FLEX_ACTION_SPACES,
    JOINT_TO_FINGER,
    alpha_dim,
    apply_action_np,
    apply_action_torch,
    per_finger_alpha_abs,
)
from ropetrack.refine.analysis import json_sanitize, rope_abs_residual, summarize_rope_residuals
from ropetrack.refine.cache import make_refiner_features, validate_cache
from ropetrack.refine.oracle import (
    ORACLE_OBJECTIVES,
    oracle_joint_ids,
    oracle_loss_cm2,
    torch_eval_joints_from_vertices,
    torch_eval_points_from_model,
)
from ropetrack.refine.rope_refiner import RopePoseRefiner
from ropetrack.rope import FINGER_CHAINS, FINGER_ORDER

from scripts.rope_refiner.build_freihand_refiner_cache import (
    dense_rope,
    load_prediction_joints,
    load_sample_order,
)


REPO = Path(__file__).resolve().parents[2]
OBJECTIVES = ("rope",) + ORACLE_OBJECTIVES


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pred_rope_norm_for_dataset(dataset: str, joints, chain_m, valid, fist_ratio: float, clamp: bool = True) -> list[float]:
    from ropetrack.rope import normalize_rope_distance, rope_distances_for_joints

    distances = rope_distances_for_joints(dataset, joints)
    return [
        float(normalize_rope_distance(distance, chain, fist_ratio=fist_ratio, clamp=clamp)) if is_valid and chain is not None else 0.0
        for distance, chain, is_valid in zip(distances, chain_m, valid, strict=True)
    ]


def build_inference_cache(dataset: str, rope_labels: Path, pred_dir: Path, run_meta: Path, mano_cache: Path, output: Path) -> Path:
    rope_rows = {row["sample_id"]: row for row in read_jsonl(rope_labels)}
    order = load_sample_order(run_meta, list(rope_rows))
    pred_joints = load_prediction_joints(pred_dir)
    with np.load(mano_cache) as cache:
        cache_ids = [str(sid) for sid in cache["sample_id"]]
        pose_by_id = {sid: pose for sid, pose in zip(cache_ids, cache["base_hand_pose"], strict=True)}

    if len(pred_joints) != len(order):
        raise ValueError(f"prediction/order length mismatch: pred={len(pred_joints)} order={len(order)}")

    sample_id, base_pose, base_rope, input_rope, chain_rows, valid_rows, fist_ratios = [], [], [], [], [], [], []
    for sid, joints in zip(order, pred_joints, strict=True):
        if sid not in rope_rows:
            raise ValueError(f"rope label missing sample_id: {sid}")
        if sid not in pose_by_id:
            raise ValueError(f"MANO cache missing sample_id: {sid}")
        row = rope_rows[sid]
        valid = [bool(v) for v in row["rope_valid"]]
        fist_ratio = float(row.get("normalization", {}).get("fist_ratio", 0.5))
        sample_id.append(sid)
        base_pose.append(np.asarray(pose_by_id[sid], dtype=np.float32).reshape(45))
        base_rope.append(pred_rope_norm_for_dataset(dataset, joints, row["rope_chain_m"], valid, fist_ratio))
        input_rope.append(dense_rope(row["rope_norm"], valid))
        chain_rows.append(dense_rope(row["rope_chain_m"], valid))
        valid_rows.append(valid)
        fist_ratios.append(fist_ratio)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        sample_id=np.asarray(sample_id),
        base_hand_pose=np.asarray(base_pose, dtype=np.float32),
        base_rope_norm=np.asarray(base_rope, dtype=np.float32),
        input_rope_norm=np.asarray(input_rope, dtype=np.float32),
        gt_rope_norm=np.asarray(input_rope, dtype=np.float32),
        rope_chain_m=np.asarray(chain_rows, dtype=np.float32),
        rope_valid=np.asarray(valid_rows, dtype=bool),
        fist_ratio=np.asarray(fist_ratios, dtype=np.float32),
        finger_order=np.asarray(FINGER_ORDER),
    )
    return output


def perturb_rope_cache(cache: dict[str, np.ndarray], noise_std: float, dropout: float, seed: int) -> dict[str, np.ndarray]:
    """Simulate an imperfect rope sensor on the cache, in place.

    - ``noise_std``: gaussian noise on ``input_rope_norm`` (normalized rope
      units; with FreiHAND chain lengths ~0.10 m, 0.05 is roughly +/-2.5 mm of
      distance noise), clamped back to [0, 1] like real labels.
    - ``dropout``: per-finger probability of marking the reading invalid
      (``rope_valid`` False), which excludes it from the loss and the gate.

    ``gt_rope_norm`` keeps the clean labels for later analysis. The gate and
    the optimizer both consume the perturbed reading — a noisy sensor must be
    noisy everywhere, not only in the loss.
    """
    if noise_std <= 0.0 and dropout <= 0.0:
        return cache
    rng = np.random.default_rng(seed)
    rope = np.asarray(cache["input_rope_norm"], dtype=np.float32)
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    if noise_std > 0.0:
        rope = rope + rng.normal(scale=noise_std, size=rope.shape).astype(np.float32)
        rope = np.clip(rope, 0.0, 1.0)
        rope[~valid] = 0.0
    if dropout > 0.0:
        dropped = rng.uniform(size=valid.shape) < dropout
        valid = valid & ~dropped
        rope[~valid] = 0.0
    cache["input_rope_norm"] = rope
    cache["rope_valid"] = valid
    return cache


def refined_pose(cache: dict[str, np.ndarray], checkpoint: Path, device: str) -> np.ndarray:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    rope_key = ckpt.get("config", {}).get("rope_key", "input_rope_norm")
    validate_cache(cache, require_target_hand_pose=False)
    arrays = make_refiner_features(cache, rope_key)
    model = RopePoseRefiner(hidden_dim=ckpt.get("config", {}).get("hidden_dim", 128)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        refined, _ = model(
            torch.from_numpy(arrays["base_hand_pose"]).to(device),
            torch.from_numpy(arrays["base_rope_norm"]).to(device),
            torch.from_numpy(arrays["input_rope_norm"]).to(device),
            torch.from_numpy(arrays["rope_valid"]).to(device),
        )
    return refined.cpu().numpy().astype(np.float32)


def mano_layer(device: str):
    sys.path.insert(0, str(REPO / "third_party" / "wilor"))
    from wilor.models.mano_wrapper import MANO

    mano = MANO(
        model_path=str(REPO / "mano_data"),
        gender="neutral",
        num_hand_joints=15,
        mean_params=str(REPO / "mano_data" / "mano_mean_params.npz"),
        create_body_pose=False,
        use_pca=False,
    )
    return mano.to(device).eval()


def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    arr = np.asarray(aa, dtype=np.float32)
    return Rotation.from_rotvec(arr.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(*arr.shape[:-1], 3, 3)


def torch_aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    vec = aa.reshape(-1, 3)
    angle = torch.linalg.norm(vec, dim=1, keepdim=True).clamp_min(1e-8)
    axis = vec / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    k = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(vec.shape[0], 3, 3)
    sin = torch.sin(angle).reshape(-1, 1, 1)
    cos = torch.cos(angle).reshape(-1, 1, 1)
    rot = eye + sin * k + (1.0 - cos) * (k @ k)
    return rot.reshape(*aa.shape[:-1], 3, 3)


def apply_finger_curl_alpha(base_hand_pose: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Legacy mult5 apply, kept as the reference implementation for tests."""
    refined = np.asarray(base_hand_pose, dtype=np.float32).copy()
    alpha = np.asarray(alpha, dtype=np.float32)
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        dims = np.asarray([3 * joint + axis for joint in joints for axis in range(3)])
        refined[:, dims] += alpha[:, finger_idx:finger_idx + 1] * refined[:, dims]
    return refined


def load_mano_globals(mano_cache: Path, sample_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load global_orient/betas/cam_t reordered to sample_ids.

    build_inference_cache tolerates a --mano-cache whose row order differs
    from run_meta sample_order (it joins base_hand_pose by id), so the
    extrinsics must be joined by id as well — positional reads would pair
    reordered poses with the wrong camera/orientation.
    """
    with np.load(mano_cache) as loaded:
        index = {str(sid): i for i, sid in enumerate(loaded["sample_id"])}
        missing = [str(sid) for sid in sample_ids if str(sid) not in index]
        if missing:
            raise ValueError(f"MANO cache missing sample_ids: {missing[:5]}")
        perm = np.asarray([index[str(sid)] for sid in sample_ids])
        return (
            np.asarray(loaded["base_global_orient"], dtype=np.float32)[perm],
            np.asarray(loaded["base_betas"], dtype=np.float32)[perm],
            np.asarray(loaded["base_cam_t"], dtype=np.float32)[perm],
        )


def torch_rope_norm(joints: torch.Tensor, chains, chain_m: torch.Tensor, fist_ratio: torch.Tensor) -> torch.Tensor:
    """Normalized rope values from 21-joint positions, differentiable.

    Unclamped on purpose: clamping to [0, 1] would kill gradients at the
    boundary. Matches the label normalization for in-range values.
    """
    preds = []
    for chain_idx, chain in enumerate(chains):
        dist = torch.linalg.norm(joints[:, chain[-1]] - joints[:, chain[0]], dim=1)
        lmin = fist_ratio * chain_m[:, chain_idx]
        denom = ((1.0 - fist_ratio) * chain_m[:, chain_idx]).clamp_min(1e-9)
        preds.append((dist - lmin) / denom)
    return torch.stack(preds, dim=1)


def compute_flex_directions(
    base_pose: torch.Tensor,
    global_orient: torch.Tensor,
    betas: torch.Tensor,
    mano,
    batch_size: int,
    per_finger: bool = False,
) -> torch.Tensor:
    """Per-sample frozen flexion directions [N, 15, 3].

    Direction of MANO joint j = gradient of its finger's rope distance
    w.r.t. that joint's axis-angle components, evaluated at the base pose.
    Positive alpha therefore opens the finger (increases rope distance),
    negative alpha closes it. This defines "flexion" through the rope geometry
    itself rather than an assumed fixed anatomical axis; it is an
    approximation and is frozen before optimization so the flex spaces are
    well-defined linear subspaces per sample.

    Normalization:

    - ``per_finger=False`` (flex15): each joint's 3-vector is normalized to
      unit length independently.
    - ``per_finger=True`` (flex5): the finger's concatenated 9-dim gradient
      is normalized as one unit vector, so a single per-finger alpha
      distributes the correction across joints proportionally to their
      leverage on the rope distance.
    """
    # The WiLoR MANO wrapper always emits OpenPose/FreiHAND-ordered joints
    # (mano_wrapper.py joint_map), regardless of dataset. Dataset-specific
    # chains only apply to eval-decoded joints (joints_from_vertices).
    # Indexing out.joints with FINGER_CHAINS["ho3d"] would measure the wrong
    # fingers; chain slot i still matches FINGER_ORDER finger i.
    chains = FINGER_CHAINS["freihand"]
    num = base_pose.shape[0]
    directions = torch.zeros((num, 15, 3), dtype=torch.float32, device=base_pose.device)
    orient_rot = torch_aa_to_rotmat(global_orient)[:, None]
    for start in range(0, num, batch_size):
        end = min(start + batch_size, num)
        pose = base_pose[start:end].detach().clone().requires_grad_(True)
        out = mano(
            global_orient=orient_rot[start:end],
            hand_pose=torch_aa_to_rotmat(pose.reshape(-1, 15, 3)),
            betas=betas[start:end],
            pose2rot=False,
        )
        joints = out.joints
        for finger_idx, chain in enumerate(chains):
            dist = torch.linalg.norm(joints[:, chain[-1]] - joints[:, chain[0]], dim=1).sum()
            (grad,) = torch.autograd.grad(dist, pose, retain_graph=finger_idx < len(chains) - 1)
            with torch.no_grad():
                finger_joints = FINGER_POSE_GROUPS[finger_idx]
                if per_finger:
                    dims = [3 * joint + axis for joint in finger_joints for axis in range(3)]
                    finger_vec = grad[:, dims]
                    norm = torch.linalg.norm(finger_vec, dim=1, keepdim=True)
                    unit = torch.where(norm > 1e-8, finger_vec / norm.clamp_min(1e-8), torch.zeros_like(finger_vec))
                    for slot, joint in enumerate(finger_joints):
                        directions[start:end, joint] = unit[:, 3 * slot : 3 * slot + 3]
                else:
                    for joint in finger_joints:
                        vec = grad[:, 3 * joint : 3 * joint + 3]
                        norm = torch.linalg.norm(vec, dim=1, keepdim=True)
                        directions[start:end, joint] = torch.where(norm > 1e-8, vec / norm.clamp_min(1e-8), torch.zeros_like(vec))
    return directions


def gate_from_cache(cache: dict[str, np.ndarray], gate_threshold: float) -> np.ndarray:
    """Per-finger gate [N, 5]: only fingers whose base rope residual exceeds
    the threshold (normalized rope units) are allowed to move.

    Low-residual fingers carry little correction signal — letting them
    participate spreads gradient onto fingers that should stay put and risks
    breaking clean predictions when the optimization is driven hard.
    """
    base = np.asarray(cache["base_rope_norm"], dtype=np.float32)
    target = np.asarray(cache["input_rope_norm"], dtype=np.float32)
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    return (np.abs(base - target) > float(gate_threshold)) & valid


def expand_gate_to_alpha(gate: np.ndarray, action_space: str) -> np.ndarray:
    """[N, 5] finger gate -> [N, alpha_dim] column mask."""
    gate = np.asarray(gate, dtype=np.float32)
    if alpha_dim(action_space) == 5:
        return gate
    return gate[:, JOINT_TO_FINGER]


def optimize_alpha(
    cache: dict[str, np.ndarray],
    mano_cache: Path,
    device: str,
    steps: int,
    lr: float,
    alpha_l2: float,
    max_alpha: float,
    batch_size: int,
    dataset: str,
    action_space: str = "mult5",
    objective: str = "rope",
    gt_xyz: np.ndarray | None = None,
    j_regressor: np.ndarray | None = None,
    gate_threshold: float | None = None,
    mano_module=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Per-sample alpha optimization; returns (refined_pose, alpha, flex_directions)."""
    if action_space not in ACTION_SPACES:
        raise ValueError(f"unsupported action space: {action_space}")
    if objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    validate_cache(cache, require_target_hand_pose=False)
    orient_np, betas_np, cam_t_np = load_mano_globals(mano_cache, cache["sample_id"])
    global_orient = torch.from_numpy(orient_np).to(device)
    betas = torch.from_numpy(betas_np).to(device)
    cam_t = torch.from_numpy(cam_t_np).to(device)
    base_pose = torch.from_numpy(np.asarray(cache["base_hand_pose"], dtype=np.float32)).to(device)
    target_rope = torch.from_numpy(np.asarray(cache["input_rope_norm"], dtype=np.float32)).to(device)
    chain = torch.from_numpy(np.asarray(cache["rope_chain_m"], dtype=np.float32)).to(device).clamp_min(1e-8)
    valid = torch.from_numpy(np.asarray(cache["rope_valid"], dtype=np.float32)).to(device)
    fist_ratio_np = np.asarray(cache.get("fist_ratio", np.full(base_pose.shape[0], 0.5)), dtype=np.float32)
    fist_ratio = torch.from_numpy(fist_ratio_np).to(device)

    gt_joints = None
    j_regressor_t = None
    joint_ids: list[int] = []
    if objective in ORACLE_OBJECTIVES:
        if gt_xyz is None or j_regressor is None:
            raise ValueError(f"{objective} requires gt_xyz and j_regressor")
        gt = np.asarray(gt_xyz, dtype=np.float32)
        if gt.shape != (base_pose.shape[0], 21, 3):
            raise ValueError(f"gt_xyz shape must be {(base_pose.shape[0], 21, 3)}, got {gt.shape}")
        gt_joints = torch.from_numpy(gt).to(device)
        j_regressor_t = torch.from_numpy(np.asarray(j_regressor, dtype=np.float32)).to(device)
        joint_ids = oracle_joint_ids(dataset, objective)

    alpha = torch.zeros((base_pose.shape[0], alpha_dim(action_space)), dtype=torch.float32, device=device, requires_grad=True)
    mano = mano_module if mano_module is not None else mano_layer(device)
    # out.joints is always OpenPose/FreiHAND-ordered (see compute_flex_directions).
    chains = FINGER_CHAINS["freihand"]
    orient_rot = torch_aa_to_rotmat(global_orient)[:, None]

    gate = None
    gate_alpha = None
    if gate_threshold is not None:
        gate_np = gate_from_cache(cache, gate_threshold)
        gate = torch.from_numpy(gate_np.astype(np.float32)).to(device)
        gate_alpha = torch.from_numpy(expand_gate_to_alpha(gate_np, action_space)).to(device)

    directions = None
    if action_space in FLEX_ACTION_SPACES:
        directions = compute_flex_directions(
            base_pose, global_orient, betas, mano, batch_size, per_finger=(action_space == "flex5")
        )

    for _ in range(steps):
        if alpha.grad is not None:
            alpha.grad.zero_()
        for start in range(0, base_pose.shape[0], batch_size):
            end = min(start + batch_size, base_pose.shape[0])
            alpha_batch = max_alpha * torch.tanh(alpha[start:end])
            if gate_alpha is not None:
                alpha_batch = alpha_batch * gate_alpha[start:end]
            dirs_batch = directions[start:end] if directions is not None else None
            hand_pose = apply_action_torch(base_pose[start:end], alpha_batch, action_space, dirs_batch)
            out = mano(
                global_orient=orient_rot[start:end],
                hand_pose=torch_aa_to_rotmat(hand_pose.reshape(-1, 15, 3)),
                betas=betas[start:end],
                pose2rot=False,
            )
            if objective == "rope":
                pred_rope = torch_rope_norm(out.joints, chains, chain[start:end], fist_ratio[start:end])
                weight = valid[start:end] if gate is None else valid[start:end] * gate[start:end]
                diff = (pred_rope - target_rope[start:end]) * weight
                denom = weight.sum().clamp_min(1.0)
                data_loss = (diff * diff).sum() / denom
            else:
                verts_eval = torch_eval_points_from_model(dataset, out.vertices, cam_t[start:end])
                pred_joints = torch_eval_joints_from_vertices(dataset, verts_eval, j_regressor_t)
                data_loss = oracle_loss_cm2(pred_joints, gt_joints[start:end], joint_ids)
            reg_loss = alpha_l2 * (alpha_batch * alpha_batch).mean()
            loss = data_loss + reg_loss
            loss.backward()
        with torch.no_grad():
            alpha -= lr * alpha.grad
    final_alpha = (max_alpha * torch.tanh(alpha)).detach().cpu().numpy().astype(np.float32)
    if gate_alpha is not None:
        final_alpha = final_alpha * gate_alpha.cpu().numpy()
    directions_np = directions.detach().cpu().numpy().astype(np.float32) if directions is not None else None
    refined = apply_action_np(cache["base_hand_pose"], final_alpha, action_space, directions_np)
    return refined, final_alpha, directions_np


def mano_predictions(dataset: str, hand_pose: np.ndarray, sample_ids, mano_cache: Path, device: str, batch_size: int, mano_module=None) -> tuple[list, list]:
    global_orient, betas, cam_t = load_mano_globals(mano_cache, sample_ids)
    j_regressor = load_mano_j_regressor(REPO / "mano_data" / "MANO_RIGHT.pkl")
    mano = mano_module if mano_module is not None else mano_layer(device)
    xyz_rows, vert_rows = [], []
    with torch.no_grad():
        for start in range(0, len(hand_pose), batch_size):
            end = min(start + batch_size, len(hand_pose))
            out = mano(
                global_orient=torch.from_numpy(aa_to_rotmat(global_orient[start:end])[:, None]).to(device),
                hand_pose=torch.from_numpy(aa_to_rotmat(hand_pose[start:end].reshape(-1, 15, 3))).to(device),
                betas=torch.from_numpy(betas[start:end]).to(device),
                pose2rot=False,
            )
            verts_model = out.vertices.cpu().numpy().astype(np.float32)
            for verts, trans in zip(verts_model, cam_t[start:end], strict=True):
                verts_eval = eval_points_from_model(dataset, verts, trans, "m")
                xyz_rows.append(joints_from_vertices(dataset, verts_eval, j_regressor).tolist())
                vert_rows.append(verts_eval.tolist())
    return xyz_rows, vert_rows


def decoded_rope_norm(dataset: str, xyz_rows: list, cache: dict[str, np.ndarray]) -> np.ndarray:
    """Rope values recomputed from decoded eval joints, one row per sample.

    Unclamped so the reported residual is exactly the quantity the optimizer
    minimizes (labels are clamped to [0, 1]; predictions may leave that range).
    """
    chain_m = np.asarray(cache["rope_chain_m"], dtype=np.float32)
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    fist_ratio = np.asarray(cache.get("fist_ratio", np.full(len(xyz_rows), 0.5)), dtype=np.float32)
    rows = []
    for idx, joints in enumerate(xyz_rows):
        rows.append(
            pred_rope_norm_for_dataset(
                dataset, joints, chain_m[idx].tolist(), valid[idx].tolist(), float(fist_ratio[idx]), clamp=False
            )
        )
    return np.asarray(rows, dtype=np.float32)


def rope_residual_report(dataset: str, base_xyz: list, refined_xyz: list, cache: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    """Residual-closure summary plus raw arrays, from same-decoder joints."""
    target = np.asarray(cache["input_rope_norm"], dtype=np.float32)
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    base_rope = decoded_rope_norm(dataset, base_xyz, cache)
    refined_rope = decoded_rope_norm(dataset, refined_xyz, cache)
    base_residual = rope_abs_residual(base_rope, target, valid)
    refined_residual = rope_abs_residual(refined_rope, target, valid)
    summary = summarize_rope_residuals(base_residual, refined_residual, valid)
    arrays = {
        "sample_id": np.asarray(cache["sample_id"]),
        "base_rope_norm_decoded": base_rope,
        "refined_rope_norm_decoded": refined_rope,
        "input_rope_norm": target,
        "rope_valid": valid,
        "base_rope_residual": base_residual.astype(np.float32),
        "refined_rope_residual": refined_residual.astype(np.float32),
        "finger_order": np.asarray(FINGER_ORDER),
    }
    return summary, arrays


def alpha_summary(alpha: np.ndarray, action_space: str) -> dict:
    per_finger = per_finger_alpha_abs(alpha, action_space).mean(axis=0)
    return {
        "mean_abs": float(np.mean(np.abs(alpha))),
        "max_abs": float(np.max(np.abs(alpha))),
        "per_finger_mean_abs": {finger: float(value) for finger, value in zip(FINGER_ORDER, per_finger, strict=True)},
    }


def write_pred(path: Path, xyz_rows: list, vert_rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([xyz_rows, vert_rows]), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a cached rope refiner or rope/oracle optimizer to eval MANO cache.")
    parser.add_argument("--dataset", choices=["freihand", "ho3d"], default="freihand")
    parser.add_argument("--rope-labels", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["checkpoint", "optimize", "student"], default="checkpoint",
                        help="checkpoint = legacy 45-dim pose refiner; optimize = per-sample teacher; "
                             "student = distilled one-pass alpha predictor (train_alpha_student.py checkpoint).")
    parser.add_argument("--objective", choices=list(OBJECTIVES), default="rope",
                        help="optimize-mode data term: rope label MSE, or GT-joint oracle ceiling probes.")
    parser.add_argument("--action-space", choices=list(ACTION_SPACES), default="mult5",
                        help="mult5 = original per-finger curl scale; mult15 = per-joint scale; "
                             "flex15 = additive per-joint flexion; flex5 = additive per-finger coupled flexion (matched capacity).")
    parser.add_argument("--gate-residual-threshold", type=float, default=None,
                        help="Optimize-mode per-finger gating: only fingers with base rope residual above this "
                             "threshold (normalized rope units, e.g. 0.1) may move; others keep alpha=0 and are "
                             "excluded from the rope loss.")
    parser.add_argument("--rope-noise-std", type=float, default=0.0,
                        help="Simulated sensor noise: gaussian std on input_rope_norm in normalized units "
                             "(~0.05 is +/-2.5 mm of distance noise on FreiHAND-scale hands).")
    parser.add_argument("--rope-dropout", type=float, default=0.0,
                        help="Simulated sensor dropout: per-finger probability of marking the rope reading invalid.")
    parser.add_argument("--rope-noise-seed", type=int, default=0,
                        help="Seed for the simulated sensor perturbation (deterministic reruns).")
    parser.add_argument("--gt-xyz", type=Path, default=None,
                        help="GT xyz json ([N, 21, 3] eval-frame meters, same row order as run_meta sample_order). Required for oracle objectives.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    # Defaults are the published working recipe from experience/0027 (aggressive
    # run). The old conservative defaults (80/0.05/0.1/0.25) provably do nothing.
    parser.add_argument("--opt-steps", type=int, default=120)
    parser.add_argument("--opt-lr", type=float, default=2.0)
    parser.add_argument("--opt-alpha-l2", type=float, default=0.001)
    parser.add_argument("--opt-max-alpha", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.mode != "optimize" and args.objective != "rope":
        raise ValueError("oracle objectives require --mode optimize")
    if args.objective in ORACLE_OBJECTIVES and args.gt_xyz is None:
        raise ValueError(f"--gt-xyz is required for --objective {args.objective}")
    if args.mode == "student" and args.checkpoint is None:
        raise ValueError("--checkpoint is required in student mode")

    cache_path = args.out_dir / "refiner_eval_cache.npz"
    build_inference_cache(args.dataset, args.rope_labels, args.pred_dir, args.run_meta, args.mano_cache, cache_path)
    with np.load(cache_path) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    if args.rope_noise_std > 0.0 or args.rope_dropout > 0.0:
        perturb_rope_cache(cache, args.rope_noise_std, args.rope_dropout, args.rope_noise_seed)
        # rewrite so downstream analysis (sliced/deadzone) sees the same
        # sensor readings the optimizer consumed; gt_rope_norm stays clean
        np.savez(cache_path, **cache)

    alpha = None
    resolved_action_space = args.action_space
    resolved_gate_threshold = args.gate_residual_threshold
    if args.mode == "checkpoint":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in checkpoint mode")
        refined = refined_pose(cache, args.checkpoint, args.device)
    elif args.mode == "student":
        from ropetrack.refine.alpha_student import student_alpha

        alpha, student_config = student_alpha(cache, args.checkpoint, args.device)
        # the checkpoint is authoritative for how its alphas must be applied
        resolved_action_space = student_config["action_space"]
        if resolved_gate_threshold is None:
            resolved_gate_threshold = student_config.get("gate_threshold")
        directions = None
        if resolved_action_space in FLEX_ACTION_SPACES:
            orient_np, betas_np, _ = load_mano_globals(args.mano_cache, cache["sample_id"])
            directions_t = compute_flex_directions(
                torch.from_numpy(np.asarray(cache["base_hand_pose"], dtype=np.float32)).to(args.device),
                torch.from_numpy(orient_np).to(args.device),
                torch.from_numpy(betas_np).to(args.device),
                mano_layer(args.device),
                args.batch_size,
                per_finger=(resolved_action_space == "flex5"),
            )
            directions = directions_t.cpu().numpy().astype(np.float32)
            np.save(args.out_dir / "flex_directions.npy", directions)
        if resolved_gate_threshold is not None:
            gate = gate_from_cache(cache, resolved_gate_threshold)
            alpha = alpha * expand_gate_to_alpha(gate, resolved_action_space)
        refined = apply_action_np(cache["base_hand_pose"], alpha, resolved_action_space, directions)
        np.save(args.out_dir / "alpha.npy", alpha)
    else:
        gt_xyz = None
        j_regressor = None
        if args.objective in ORACLE_OBJECTIVES:
            gt_xyz = np.asarray(read_json(args.gt_xyz), dtype=np.float32)
            j_regressor = load_mano_j_regressor(REPO / "mano_data" / "MANO_RIGHT.pkl")
        refined, alpha, directions = optimize_alpha(
            cache,
            args.mano_cache,
            args.device,
            args.opt_steps,
            args.opt_lr,
            args.opt_alpha_l2,
            args.opt_max_alpha,
            args.batch_size,
            args.dataset,
            action_space=args.action_space,
            objective=args.objective,
            gt_xyz=gt_xyz,
            j_regressor=j_regressor,
            gate_threshold=args.gate_residual_threshold,
        )
        np.save(args.out_dir / "alpha.npy", alpha)
        if directions is not None:
            np.save(args.out_dir / "flex_directions.npy", directions)
    np.save(args.out_dir / "refined_hand_pose.npy", refined)
    np.save(args.out_dir / "sample_id.npy", cache["sample_id"])

    base_xyz, base_verts = mano_predictions(args.dataset, cache["base_hand_pose"], cache["sample_id"], args.mano_cache, args.device, args.batch_size)
    refined_xyz, refined_verts = mano_predictions(args.dataset, refined, cache["sample_id"], args.mano_cache, args.device, args.batch_size)
    write_pred(args.out_dir / "base_pred.json", base_xyz, base_verts)
    write_pred(args.out_dir / "pred.json", refined_xyz, refined_verts)

    residual_summary, residual_arrays = rope_residual_report(args.dataset, base_xyz, refined_xyz, cache)
    np.savez(args.out_dir / "rope_residuals.npz", **residual_arrays)

    summary = {
        "num_samples": int(refined.shape[0]),
        "mean_abs_delta": float(np.mean(np.abs(refined - cache["base_hand_pose"]))),
        "mode": args.mode,
        "objective": args.objective,
        "action_space": resolved_action_space,
        "rope_sensor": {
            "noise_std": args.rope_noise_std,
            "dropout": args.rope_dropout,
            "seed": args.rope_noise_seed,
            "frac_valid_after": float(np.asarray(cache["rope_valid"], dtype=bool).mean()),
        },
        "rope_residual": residual_summary,
    }
    if args.mode == "optimize":
        summary["optimization"] = {
            "steps": args.opt_steps,
            "lr": args.opt_lr,
            "alpha_l2": args.opt_alpha_l2,
            "max_alpha": args.opt_max_alpha,
            "batch_size": args.batch_size,
            "gate_residual_threshold": resolved_gate_threshold,
        }
        if args.objective in ORACLE_OBJECTIVES:
            summary["oracle_joint_ids"] = oracle_joint_ids(args.dataset, args.objective)
    if args.mode == "student":
        summary["student_checkpoint"] = str(args.checkpoint)
    if alpha is not None:
        summary["alpha"] = alpha_summary(alpha, resolved_action_space)
        if resolved_gate_threshold is not None:
            gate = gate_from_cache(cache, resolved_gate_threshold)
            valid = np.asarray(cache["rope_valid"], dtype=bool)
            summary["gating"] = {
                "threshold": resolved_gate_threshold,
                "frac_fingers_gated": float(gate.sum() / max(valid.sum(), 1)),
                "frac_samples_any_gated": float(gate.any(axis=1).mean()),
                "per_finger_frac_gated": {
                    finger: float(gate[:, idx].mean()) for idx, finger in enumerate(FINGER_ORDER)
                },
            }
    (args.out_dir / "summary.json").write_text(json.dumps(json_sanitize(summary), indent=2), encoding="utf-8")
    return args.out_dir


if __name__ == "__main__":
    main()
