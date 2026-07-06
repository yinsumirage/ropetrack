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
from ropetrack.refine.cache import make_refiner_features, validate_cache
from ropetrack.refine.rope_refiner import RopePoseRefiner
from ropetrack.rope import FINGER_CHAINS, FINGER_ORDER

from scripts.rope_refiner.build_freihand_refiner_cache import (
    dense_rope,
    load_prediction_joints,
    load_sample_order,
)


REPO = Path(__file__).resolve().parents[2]
FINGER_POSE_GROUPS = (
    (12, 13, 14),  # thumb
    (0, 1, 2),    # index
    (3, 4, 5),    # middle
    (9, 10, 11),  # ring
    (6, 7, 8),    # pinky
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pred_rope_norm_for_dataset(dataset: str, joints, chain_m, valid, fist_ratio: float) -> list[float]:
    from ropetrack.rope import normalize_rope_distance, rope_distances_for_joints

    distances = rope_distances_for_joints(dataset, joints)
    return [
        float(normalize_rope_distance(distance, chain, fist_ratio=fist_ratio)) if is_valid and chain is not None else 0.0
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

    sample_id, base_pose, base_rope, input_rope, chain_rows, valid_rows = [], [], [], [], [], []
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
        finger_order=np.asarray(FINGER_ORDER),
    )
    return output


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
    refined = np.asarray(base_hand_pose, dtype=np.float32).copy()
    alpha = np.asarray(alpha, dtype=np.float32)
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        dims = np.asarray([3 * joint + axis for joint in joints for axis in range(3)])
        refined[:, dims] += alpha[:, finger_idx:finger_idx + 1] * refined[:, dims]
    return refined


def torch_apply_finger_curl_alpha(base_hand_pose: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    refined = base_hand_pose.clone()
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        dims = torch.tensor([3 * joint + axis for joint in joints for axis in range(3)], device=base_hand_pose.device)
        refined[:, dims] = refined[:, dims] + alpha[:, finger_idx:finger_idx + 1] * refined[:, dims]
    return refined


def optimize_finger_curl(
    cache: dict[str, np.ndarray],
    mano_cache: Path,
    device: str,
    steps: int,
    lr: float,
    alpha_l2: float,
    max_alpha: float,
    batch_size: int,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    validate_cache(cache, require_target_hand_pose=False)
    with np.load(mano_cache) as loaded:
        global_orient = torch.from_numpy(np.asarray(loaded["base_global_orient"], dtype=np.float32)).to(device)
        betas = torch.from_numpy(np.asarray(loaded["base_betas"], dtype=np.float32)).to(device)
    base_pose = torch.from_numpy(np.asarray(cache["base_hand_pose"], dtype=np.float32)).to(device)
    target_rope = torch.from_numpy(np.asarray(cache["input_rope_norm"], dtype=np.float32)).to(device)
    chain = torch.from_numpy(np.asarray(cache["rope_chain_m"], dtype=np.float32)).to(device).clamp_min(1e-8)
    valid = torch.from_numpy(np.asarray(cache["rope_valid"], dtype=np.float32)).to(device)
    alpha = torch.zeros((base_pose.shape[0], 5), dtype=torch.float32, device=device, requires_grad=True)
    mano = mano_layer(device)
    chains = [chain_ids for chain_ids in FINGER_CHAINS[dataset]]

    for _ in range(steps):
        total_loss = torch.zeros((), device=device)
        if alpha.grad is not None:
            alpha.grad.zero_()
        for start in range(0, base_pose.shape[0], batch_size):
            end = min(start + batch_size, base_pose.shape[0])
            alpha_batch = max_alpha * torch.tanh(alpha[start:end])
            hand_pose = torch_apply_finger_curl_alpha(base_pose[start:end], alpha_batch)
            out = mano(
                global_orient=torch_aa_to_rotmat(global_orient[start:end])[:, None],
                hand_pose=torch_aa_to_rotmat(hand_pose.reshape(-1, 15, 3)),
                betas=betas[start:end],
                pose2rot=False,
            )
            joints = out.joints
            pred = []
            for rope_idx, finger_chain in enumerate(chains):
                dist = torch.linalg.norm(joints[:, finger_chain[-1]] - joints[:, finger_chain[0]], dim=1)
                norm = (dist - 0.5 * chain[start:end, rope_idx]) / (0.5 * chain[start:end, rope_idx])
                pred.append(norm)
            pred_rope = torch.stack(pred, dim=1)
            diff = (pred_rope - target_rope[start:end]) * valid[start:end]
            denom = valid[start:end].sum().clamp_min(1.0)
            rope_loss = (diff * diff).sum() / denom
            reg_loss = alpha_l2 * (alpha_batch * alpha_batch).mean()
            loss = rope_loss + reg_loss
            loss.backward()
            total_loss = total_loss + loss.detach()
        with torch.no_grad():
            alpha -= lr * alpha.grad
    final_alpha = (max_alpha * torch.tanh(alpha)).detach().cpu().numpy().astype(np.float32)
    return apply_finger_curl_alpha(cache["base_hand_pose"], final_alpha), final_alpha


def mano_predictions(dataset: str, hand_pose: np.ndarray, mano_cache: Path, device: str, batch_size: int) -> tuple[list, list]:
    with np.load(mano_cache) as cache:
        global_orient = np.asarray(cache["base_global_orient"], dtype=np.float32)
        betas = np.asarray(cache["base_betas"], dtype=np.float32)
        cam_t = np.asarray(cache["base_cam_t"], dtype=np.float32)
    j_regressor = load_mano_j_regressor(REPO / "mano_data" / "MANO_RIGHT.pkl")
    mano = mano_layer(device)
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


def write_pred(path: Path, xyz_rows: list, vert_rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([xyz_rows, vert_rows]), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a cached rope refiner or rope optimizer to eval MANO cache.")
    parser.add_argument("--dataset", choices=["freihand", "ho3d"], default="freihand")
    parser.add_argument("--rope-labels", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--mano-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=["checkpoint", "optimize"], default="checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--opt-steps", type=int, default=80)
    parser.add_argument("--opt-lr", type=float, default=0.05)
    parser.add_argument("--opt-alpha-l2", type=float, default=0.1)
    parser.add_argument("--opt-max-alpha", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    cache_path = args.out_dir / "refiner_eval_cache.npz"
    build_inference_cache(args.dataset, args.rope_labels, args.pred_dir, args.run_meta, args.mano_cache, cache_path)
    with np.load(cache_path) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    if args.mode == "checkpoint":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in checkpoint mode")
        refined = refined_pose(cache, args.checkpoint, args.device)
    else:
        refined, alpha = optimize_finger_curl(
            cache,
            args.mano_cache,
            args.device,
            args.opt_steps,
            args.opt_lr,
            args.opt_alpha_l2,
            args.opt_max_alpha,
            args.batch_size,
            args.dataset,
        )
        np.save(args.out_dir / "alpha.npy", alpha)
    np.save(args.out_dir / "refined_hand_pose.npy", refined)
    np.save(args.out_dir / "sample_id.npy", cache["sample_id"])

    base_xyz, base_verts = mano_predictions(args.dataset, cache["base_hand_pose"], args.mano_cache, args.device, args.batch_size)
    refined_xyz, refined_verts = mano_predictions(args.dataset, refined, args.mano_cache, args.device, args.batch_size)
    write_pred(args.out_dir / "base_pred.json", base_xyz, base_verts)
    write_pred(args.out_dir / "pred.json", refined_xyz, refined_verts)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_samples": int(refined.shape[0]),
                "mean_abs_delta": float(np.mean(np.abs(refined - cache["base_hand_pose"]))),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return args.out_dir


if __name__ == "__main__":
    main()
