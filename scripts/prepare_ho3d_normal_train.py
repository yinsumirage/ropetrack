#!/usr/bin/env python3
"""Build a sequence-balanced, unmodified HO3D v3 training subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


OPENPOSE_ORDER = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
HO3D_TO_OPENCV = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequence_balanced(ids: list[str], count: int, seed: int) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        grouped[sample_id.split("/", 1)[0]].append(sample_id)
    rng = np.random.default_rng(seed)
    pools = {}
    for sequence, rows in grouped.items():
        pools[sequence] = [rows[index] for index in rng.permutation(len(rows))]
    sequences = sorted(pools)
    selected = []
    while len(selected) < count:
        before = len(selected)
        for sequence in sequences:
            if pools[sequence]:
                selected.append(pools[sequence].pop())
                if len(selected) == count:
                    return selected
        if len(selected) == before:
            break
    raise ValueError(f"requested {count} rows but source contains only {len(selected)}")


def gt_openpose_opencv(joints: np.ndarray) -> np.ndarray:
    value = np.asarray(joints, dtype=np.float32)
    if value.shape != (21, 3) or not np.isfinite(value).all():
        raise ValueError(f"invalid HO3D joints: {value.shape}")
    return (value * HO3D_TO_OPENCV)[OPENPOSE_ORDER]


def resolve_image(root: Path, sample_id: str) -> Path:
    sequence, frame = sample_id.split("/")
    for suffix in (".jpg", ".png", ".jpeg"):
        path = root / "train" / sequence / "rgb" / f"{frame}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"HO3D image missing for {sample_id}")


def write_json_array(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for index, row in enumerate(rows):
            if index:
                handle.write(",")
            handle.write(json.dumps(np.asarray(row).tolist(), separators=(",", ":")))
        handle.write("]\n")


def prepare(input_root: Path, output_root: Path, count: int, seed: int) -> Path:
    train_file = input_root / "train.txt"
    eval_file = input_root / "evaluation.txt"
    train_ids, eval_ids = read_ids(train_file), read_ids(eval_file)
    train_sequences = {value.split("/", 1)[0] for value in train_ids}
    eval_sequences = {value.split("/", 1)[0] for value in eval_ids}
    if set(train_ids) & set(eval_ids) or train_sequences & eval_sequences:
        raise ValueError("HO3D v3 official train/evaluation split overlaps")
    selected = sequence_balanced(train_ids, count, seed)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "train").symlink_to(input_root / "train", target_is_directory=True)
    (output_root / "train.txt").write_text("".join(f"{value}\n" for value in selected), encoding="utf-8")

    xyz_rows, manifest_rows = [], []
    for sample_id in selected:
        sequence, frame = sample_id.split("/")
        meta_path = input_root / "train" / sequence / "meta" / f"{frame}.pkl"
        with meta_path.open("rb") as handle:
            meta = pickle.load(handle, encoding="latin1")
        xyz_rows.append(gt_openpose_opencv(meta["handJoints3D"]))
        manifest_rows.append({
            "sample_id": sample_id,
            "sequence": sequence,
            "image_path": str(resolve_image(input_root, sample_id)),
            "meta_path": str(meta_path),
            "source_split": "train",
            "image_transform": "none",
        })
    write_json_array(output_root / "training_xyz.json", xyz_rows)
    with (output_root / "selection.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    counts = Counter(value.split("/", 1)[0] for value in selected)
    protocol = {
        "dataset": "HO3D_v3",
        "split": "train",
        "source_root": str(input_root),
        "source_train_rows": len(train_ids),
        "source_eval_rows": len(eval_ids),
        "source_sample_overlap": 0,
        "source_sequence_overlap": 0,
        "selection_rule": "sequence-balanced deterministic sampling without replacement",
        "selection_uses_eval_error": False,
        "seed": seed,
        "num_samples": len(selected),
        "num_sequences": len(counts),
        "sequence_count_min_max": [min(counts.values()), max(counts.values())],
        "images": "official unmodified train RGB via symlink",
        "visibility": "official natural visibility; no artificial mask or occlusion",
        "gt_source": "official train meta handJoints3D",
        "gt_transform": "HO3D/MANO order -> OpenPose order; [x,y,z] -> [x,-y,-z] OpenCV camera metres",
        "sha256": {
            "source_train_txt": file_sha256(train_file),
            "source_evaluation_txt": file_sha256(eval_file),
            "selected_train_txt": file_sha256(output_root / "train.txt"),
            "selection_jsonl": file_sha256(output_root / "selection.jsonl"),
            "training_xyz_json": file_sha256(output_root / "training_xyz.json"),
        },
    }
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2))
    return output_root


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=27000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    prepare(args.input_root, args.output_root, args.count, args.seed)
