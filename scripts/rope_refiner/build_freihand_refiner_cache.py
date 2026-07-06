from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.io import read_jsonl
from ropetrack.rope import FINGER_ORDER, normalize_rope_distance, rope_distances_for_joints


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_prediction_joints(pred_dir: Path) -> list:
    payload = read_json(pred_dir / "pred.json")
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("Expected pred.json to contain [xyz_predictions, vertex_predictions].")
    return payload[0]


def load_sample_order(run_meta: Path | None, fallback_ids: list[str]) -> list[str]:
    if run_meta is None:
        return fallback_ids
    meta = read_json(run_meta)
    if "sample_order" not in meta:
        raise ValueError(f"run_meta missing sample_order: {run_meta}")
    return list(meta["sample_order"])


def mano_pose_vector(entry) -> np.ndarray:
    if len(entry) == 1 and isinstance(entry[0], list):
        entry = entry[0]
    pose = np.asarray(entry, dtype=np.float32)
    if pose.shape[0] < 48:
        raise ValueError(f"expected MANO vector with at least 48 values, got {pose.shape[0]}")
    return pose[3:48]


def dense_rope(values, valid) -> list[float]:
    return [float(value) if is_valid and value is not None else 0.0 for value, is_valid in zip(values, valid, strict=True)]


def pred_rope_norm(joints, chain_m, valid, fist_ratio: float) -> list[float]:
    distances = rope_distances_for_joints("freihand", joints)
    return [
        float(normalize_rope_distance(distance, chain, fist_ratio=fist_ratio)) if is_valid and chain is not None else 0.0
        for distance, chain, is_valid in zip(distances, chain_m, valid, strict=True)
    ]


def build_cache(
    input_root: Path,
    rope_labels: Path,
    pred_dir: Path,
    run_meta: Path | None,
    output: Path,
    split: str = "training",
    limit: int | None = None,
    base_hand_pose_source: str = "target",
    base_mano_cache: Path | None = None,
) -> Path:
    if base_hand_pose_source not in {"target", "mano_cache"}:
        raise ValueError("--base-hand-pose-source must be 'target' or 'mano_cache'")
    if base_hand_pose_source == "mano_cache" and base_mano_cache is None:
        raise ValueError("--base-mano-cache is required when --base-hand-pose-source=mano_cache")

    mano_rows = read_json(input_root / f"{split}_mano.json")
    rope_rows = {row["sample_id"]: row for row in read_jsonl(rope_labels)}
    order = load_sample_order(run_meta, list(rope_rows))
    pred_joints = load_prediction_joints(pred_dir)
    cache_pose_by_id = {}
    if base_mano_cache is not None:
        with np.load(base_mano_cache) as cache:
            cache_pose_by_id = {
                str(sid): np.asarray(pose, dtype=np.float32).reshape(45)
                for sid, pose in zip(cache["sample_id"], cache["base_hand_pose"], strict=True)
            }
    if run_meta is not None and len(pred_joints) != len(order):
        raise ValueError(f"prediction/order length mismatch: pred={len(pred_joints)} order={len(order)}")
    if limit is not None and limit > 0:
        order = order[:limit]
        pred_joints = pred_joints[:limit]
    if len(pred_joints) < len(order):
        raise ValueError(f"not enough predictions for sample order: pred={len(pred_joints)} order={len(order)}")

    sample_id, target_pose, base_pose, base_rope, input_rope, valid_rows = [], [], [], [], [], []
    for idx, (sid, joints) in enumerate(zip(order, pred_joints, strict=True)):
        if sid not in rope_rows:
            raise ValueError(f"rope label missing sample_id: {sid}")
        if base_hand_pose_source == "mano_cache" and sid not in cache_pose_by_id:
            raise ValueError(f"MANO cache missing sample_id: {sid}")
        row = rope_rows[sid]
        mano_idx = int(sid) if sid.isdigit() else idx
        if mano_idx >= len(mano_rows):
            raise ValueError(f"MANO row missing for sample_id: {sid}")
        valid = [bool(v) for v in row["rope_valid"]]
        fist_ratio = float(row.get("normalization", {}).get("fist_ratio", 0.5))
        sample_id.append(sid)
        target_pose.append(mano_pose_vector(mano_rows[mano_idx]))
        base_pose.append(cache_pose_by_id[sid] if base_hand_pose_source == "mano_cache" else target_pose[-1])
        base_rope.append(pred_rope_norm(joints, row["rope_chain_m"], valid, fist_ratio))
        input_rope.append(dense_rope(row["rope_norm"], valid))
        valid_rows.append(valid)

    target = np.asarray(target_pose, dtype=np.float32)
    base = np.asarray(base_pose, dtype=np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        sample_id=np.asarray(sample_id),
        base_hand_pose=base,
        target_hand_pose=target,
        base_rope_norm=np.asarray(base_rope, dtype=np.float32),
        input_rope_norm=np.asarray(input_rope, dtype=np.float32),
        gt_rope_norm=np.asarray(input_rope, dtype=np.float32),
        rope_valid=np.asarray(valid_rows, dtype=bool),
        finger_order=np.asarray(FINGER_ORDER),
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the first FreiHAND cached rope-refiner training npz.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--rope-labels", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["training", "evaluation"], default="training")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-hand-pose-source", choices=["target", "mano_cache"], default="target")
    parser.add_argument("--base-mano-cache", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    limit = None if args.limit <= 0 else args.limit
    out = build_cache(
        args.input_root,
        args.rope_labels,
        args.pred_dir,
        args.run_meta,
        args.output,
        split=args.split,
        limit=limit,
        base_hand_pose_source=args.base_hand_pose_source,
        base_mano_cache=args.base_mano_cache,
    )
    print(f"Wrote refiner cache: {out}")
    return out


if __name__ == "__main__":
    main()
