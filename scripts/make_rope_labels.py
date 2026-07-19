#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from ropetrack.datasets.hand_pose import (
    HO3D_TO_OPENCV_CAMERA,
    camera_matrix_from_meta,
    iter_ho3d_samples,
    iter_ho3d_samples_from_order,
    project_points,
    read_ho3d_train_ids,
    read_json,
    resolve_image_path,
    iter_hand_pose_samples,
)
from ropetrack.io import write_jsonl
from ropetrack.rope import FINGER_CHAINS, FINGER_COLORS, FINGER_ORDER, build_rope_row, canonical_rope_dataset


def freihand_items(root: Path, limit: int | None, split: str = "evaluation"):
    xyz = read_json(root / f"{split}_xyz.json")
    K_path = root / f"{split}_K.json"
    Ks = read_json(K_path) if K_path.exists() else [None] * len(xyz)
    count = len(xyz) if limit is None or limit <= 0 else min(limit, len(xyz))
    for idx in range(count):
        frame = f"{idx:08d}"
        yield {
            "sample_id": frame,
            "joints": xyz[idx],
            "image_path": resolve_image_path(root / split / "rgb", frame),
            "K": Ks[idx],
        }


def ho3d_items(root: Path, limit: int | None, sample_order_file: Path | None = None):
    xyz = read_json(root / "evaluation_xyz.json")
    if sample_order_file:
        canonical_ids = [sample.sample_id for sample in iter_ho3d_samples(root, None)]
        id_to_gt_idx = {sample_id: idx for idx, sample_id in enumerate(canonical_ids)}
        for sample in iter_ho3d_samples_from_order(root, sample_order_file, limit):
            if sample.sample_id not in id_to_gt_idx:
                raise ValueError(f"sample_order id not found in HO3D root: {sample.sample_id}")
            yield {
                "sample_id": sample.sample_id,
                "joints": xyz[id_to_gt_idx[sample.sample_id]],
                "image_path": sample.image_path,
                "K": camera_matrix_from_meta_path(sample.meta_path),
            }
        return

    samples = list(iter_ho3d_samples(root, limit))
    for idx, sample in enumerate(samples):
        yield {
            "sample_id": sample.sample_id,
            "joints": xyz[idx],
            "image_path": sample.image_path,
            "K": camera_matrix_from_meta_path(sample.meta_path),
        }


def camera_matrix_from_meta_path(meta_path: Path):
    if not meta_path.exists():
        return None
    with meta_path.open("rb") as f:
        meta = pickle.load(f, encoding="latin1")
    return camera_matrix_from_meta(meta)


def ho3d_train_items(root: Path, limit: int | None, stride: int = 1, split_dir: str = "train"):
    """Training-split rope items straight from the meta pkls.

    handJoints3D is in the HO3D camera frame; rope distances are invariant to
    the OpenCV sign flip, and the visualization projection applies the flip
    itself, so raw meta joints are correct here (audit: experience/0040).
    """
    for sample_id in read_ho3d_train_ids(root, stride=stride, limit=limit):
        seq, frame = sample_id.split("/")
        meta_path = root / split_dir / seq / "meta" / f"{frame}.pkl"
        with meta_path.open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        yield {
            "sample_id": sample_id,
            "joints": meta["handJoints3D"],
            "image_path": resolve_image_path(root / split_dir / seq / "rgb", frame),
            "K": meta.get("camMat"),
        }


def manifest_items(dataset: str, root: Path, limit: int | None, split: str):
    xyz = read_json(root / f"{split}_xyz.json")
    samples = list(iter_hand_pose_samples(dataset, root, limit, split))
    if len(xyz) < len(samples):
        raise ValueError(f"{dataset} {split} GT shorter than manifest: xyz={len(xyz)} samples={len(samples)}")
    for idx, sample in enumerate(samples):
        yield {
            "sample_id": sample.sample_id,
            "joints": xyz[idx],
            "image_path": sample.image_path,
            "K": sample.intrinsic,
        }


def iter_rope_items(dataset: str, root: Path, limit: int | None, sample_order_file: Path | None = None, split: str = "evaluation", stride: int = 1):
    name = canonical_rope_dataset(dataset)
    if name == "freihand":
        return freihand_items(root, limit, split=split)
    if name in {"egodex", "arctic", "hot3d", "dexycb"}:
        return manifest_items(name, root, limit, split)
    if split == "evaluation":
        return ho3d_items(root, limit, sample_order_file)
    if split == "training":
        return ho3d_train_items(root, limit, stride=stride)
    raise ValueError(f"unsupported HO3D rope-label split: {split}")


def write_rope_labels(
    dataset: str,
    input_root: Path,
    output: Path,
    limit: int | None = None,
    sample_order_file: Path | None = None,
    fist_ratio: float = 0.5,
    viz_dir: Path | None = None,
    viz_count: int = 0,
    split: str = "evaluation",
    stride: int = 1,
) -> list[dict]:
    rows = []
    for idx, item in enumerate(iter_rope_items(dataset, input_root, limit, sample_order_file, split=split, stride=stride)):
        row = build_rope_row(dataset, item["sample_id"], item["joints"], fist_ratio=fist_ratio)
        rows.append(row)
        if viz_dir is not None and idx < viz_count:
            write_visualization(viz_dir / safe_png_name(item["sample_id"]), dataset, item, row)
    write_jsonl(output, rows)
    return rows


def safe_png_name(sample_id: str) -> str:
    return sample_id.replace("/", "__") + ".png"


def project_rope_chains(dataset: str, joints, K):
    if K is None:
        return {}
    name = canonical_rope_dataset(dataset)
    pts = joints
    if name == "ho3d":
        pts = [[p[0] * HO3D_TO_OPENCV_CAMERA[0], p[1] * HO3D_TO_OPENCV_CAMERA[1], p[2] * HO3D_TO_OPENCV_CAMERA[2]] for p in joints]
    out = {}
    for finger, chain in zip(FINGER_ORDER, FINGER_CHAINS[name], strict=True):
        try:
            uv = project_points([pts[i] for i in chain], K)
        except Exception:
            continue
        out[finger] = [tuple(map(float, point)) for point in uv]
    return out


def write_visualization(path: Path, dataset: str, item: dict, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if item["image_path"].exists():
        image = Image.open(item["image_path"]).convert("RGB")
    else:
        image = Image.new("RGB", (224, 224), (255, 255, 255))

    panel_w = 260
    canvas = Image.new("RGB", (image.width + panel_w, max(image.height, 224)), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for color, finger in zip(FINGER_COLORS, FINGER_ORDER, strict=True):
        pts = project_rope_chains(dataset, item["joints"], item["K"]).get(finger, [])
        for a, b in zip(pts, pts[1:], strict=False):
            draw.line([a, b], fill=color, width=3)
        for x, y in pts:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    x0 = image.width + 18
    draw.text((x0, 14), row["sample_id"], fill=(0, 0, 0), font=ImageFont.load_default())
    for idx, finger in enumerate(FINGER_ORDER):
        y = 42 + idx * 34
        norm = row["rope_norm"][idx]
        draw.text((x0, y), finger, fill=(0, 0, 0), font=ImageFont.load_default())
        draw.rectangle((x0 + 70, y + 2, x0 + 220, y + 14), outline=(120, 120, 120))
        if norm is not None:
            draw.rectangle((x0 + 70, y + 2, x0 + 70 + int(150 * norm), y + 14), fill=FINGER_COLORS[idx])
            draw.text((x0 + 70, y + 18), f"{norm:.3f}", fill=(0, 0, 0), font=ImageFont.load_default())
        else:
            draw.text((x0 + 70, y + 18), "invalid", fill=(150, 0, 0), font=ImageFont.load_default())
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fingertip-to-wrist rope labels from evaluation GT joints.")
    parser.add_argument("--dataset", choices=["freihand", "ho3d", "egodex", "arctic", "hot3d", "dexycb"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Number of samples; <=0 means all.")
    parser.add_argument("--sample-order-file", type=Path, default=None)
    parser.add_argument("--split", choices=["evaluation", "training"], default="evaluation")
    parser.add_argument("--stride", type=int, default=1,
                        help="HO3D training only: keep every k-th train.txt frame; must match the hard root stride.")
    parser.add_argument("--fist-ratio", type=float, default=0.5)
    parser.add_argument("--viz-dir", type=Path, default=None)
    parser.add_argument("--viz-count", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit <= 0 else args.limit
    if args.stride != 1 and not (args.dataset == "ho3d" and args.split == "training"):
        raise ValueError("--stride is only supported for --dataset ho3d --split training")
    rows = write_rope_labels(
        args.dataset,
        args.input_root,
        args.output,
        limit=limit,
        sample_order_file=args.sample_order_file,
        fist_ratio=args.fist_ratio,
        viz_dir=args.viz_dir,
        viz_count=args.viz_count,
        split=args.split,
        stride=args.stride,
    )
    print(f"Wrote {len(rows)} rope labels: {args.output}")


if __name__ == "__main__":
    main()
