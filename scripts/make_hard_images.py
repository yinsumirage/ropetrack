from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ropetrack.datasets.hand_pose import (  # noqa: E402
    Ho3dSample,
    bbox_from_projected_points,
    hand_bbox_from_meta,
    iter_ho3d_samples,
    project_points,
    read_json,
    resolve_image_path,
)

FREIHAND_FINGERTIP_JOINT_IDS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
FREIHAND_FINGER_END_SEGMENT_JOINT_IDS = np.asarray(
    [[4, 1], [8, 6], [6, 5], [12, 10], [10, 9], [16, 14], [14, 13], [20, 18], [18, 17]],
    dtype=np.int64,
)
HO3D_FINGERTIP_JOINT_IDS = np.asarray([16, 17, 18, 19, 20], dtype=np.int64)
HO3D_FINGER_END_SEGMENT_JOINT_IDS = np.asarray(
    [[16, 13], [17, 2], [2, 1], [18, 5], [5, 4], [19, 11], [11, 10], [20, 8], [8, 7]],
    dtype=np.int64,
)
HO3D_TO_OPENCV_CAMERA = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def clamp_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def centered_rect(x1: int, y1: int, x2: int, y2: int, severity: float) -> tuple[int, int, int, int]:
    severity = max(0.05, min(0.95, float(severity)))
    w = max(1, int(round((x2 - x1) * severity)))
    h = max(1, int(round((y2 - y1) * severity)))
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h


def fingertip_radius(bbox_xyxy, severity: float) -> int:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    base = max(1.0, min(x2 - x1, y2 - y1))
    return max(3, int(round(base * max(0.05, min(0.95, float(severity))) * 0.15)))


def project_fingertips_from_joints(joints3d, K, tip_ids=FREIHAND_FINGERTIP_JOINT_IDS) -> list[tuple[float, float]]:
    joints = np.asarray(joints3d, dtype=np.float32)
    tip_ids = np.asarray(tip_ids, dtype=np.int64)
    if joints.shape[0] <= int(tip_ids.max()):
        return []
    uv = project_points(joints[tip_ids], K)
    return [tuple(map(float, point)) for point in uv if np.isfinite(point).all()]


def project_finger_end_segments_from_joints(joints3d, K, segment_ids=FREIHAND_FINGER_END_SEGMENT_JOINT_IDS):
    joints = np.asarray(joints3d, dtype=np.float32)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if joints.shape[0] <= int(segment_ids.max()):
        return []
    uv = project_points(joints[segment_ids.reshape(-1)], K).reshape(-1, 2, 2)
    return [
        (tuple(map(float, segment[0])), tuple(map(float, segment[1])))
        for segment in uv
        if np.isfinite(segment).all()
    ]


def project_ho3d_fingertips_from_joints(joints3d, K) -> list[tuple[float, float]]:
    joints = np.asarray(joints3d, dtype=np.float32) * HO3D_TO_OPENCV_CAMERA
    return project_fingertips_from_joints(joints, K, tip_ids=HO3D_FINGERTIP_JOINT_IDS)


def project_ho3d_finger_end_segments_from_joints(joints3d, K):
    joints = np.asarray(joints3d, dtype=np.float32) * HO3D_TO_OPENCV_CAMERA
    return project_finger_end_segments_from_joints(joints, K, segment_ids=HO3D_FINGER_END_SEGMENT_JOINT_IDS)


def camera_matrix_from_meta(meta: dict):
    for key in ("camMat", "K", "camera_matrix", "intrinsics"):
        if key in meta:
            return meta[key]
    return None


def fingertip_points_from_meta(meta: dict) -> list[tuple[float, float]]:
    K = camera_matrix_from_meta(meta)
    if K is None or "handJoints3D" not in meta:
        return []
    return project_ho3d_fingertips_from_joints(meta["handJoints3D"], K)


def draw_tip_mask(out: Image.Image, bbox, points_xy, shape: str, severity: float, seed: int) -> None:
    import random

    rng = random.Random(seed)
    x1, y1, x2, y2 = clamp_bbox(bbox, out.width, out.height)
    points = list(points_xy or [((x1 + x2) / 2.0, (y1 + y2) / 2.0)])
    radius = fingertip_radius((x1, y1, x2, y2), severity)
    draw = ImageDraw.Draw(out)
    for px, py in points:
        cx = int(round(px))
        cy = int(round(py))
        tip_shape = rng.choice(["tip_circle", "tip_square", "tip_blur"]) if shape == "tip_mixed" else shape
        rect = (cx - radius, cy - radius, cx + radius, cy + radius)
        if tip_shape == "tip_circle":
            draw.ellipse(rect, fill=(0, 0, 0))
        elif tip_shape == "tip_square":
            draw.rectangle(rect, fill=(0, 0, 0))
        elif tip_shape == "tip_blur":
            patch_box = (
                max(0, cx - radius),
                max(0, cy - radius),
                min(out.width, cx + radius),
                min(out.height, cy + radius),
            )
            if patch_box[2] > patch_box[0] and patch_box[3] > patch_box[1]:
                patch = out.crop(patch_box).filter(ImageFilter.GaussianBlur(radius=max(2.0, severity * 12.0)))
                out.paste(patch, patch_box[:2])
        else:
            raise ValueError(f"unsupported tip effect: {shape}")


def draw_finger_end_mask(out: Image.Image, bbox, segments_xy, severity: float) -> None:
    x1, y1, x2, y2 = clamp_bbox(bbox, out.width, out.height)
    segments = list(segments_xy or [((x1, (y1 + y2) / 2.0), (x2, (y1 + y2) / 2.0))])
    half_width = fingertip_radius((x1, y1, x2, y2), severity)
    draw = ImageDraw.Draw(out)
    for (ax, ay), (bx, by) in segments:
        dx = float(bx) - float(ax)
        dy = float(by) - float(ay)
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 1e-6:
            draw.rectangle((ax - half_width, ay - half_width, ax + half_width, ay + half_width), fill=(0, 0, 0))
            continue
        nx = -dy / length * half_width
        ny = dx / length * half_width
        draw.polygon([
            (ax + nx, ay + ny),
            (bx + nx, by + ny),
            (bx - nx, by - ny),
            (ax - nx, ay - ny),
        ], fill=(0, 0, 0))


def apply_hard_effect(image: Image.Image, bbox, effect: str, severity: float, seed: int, points_xy=None, segments_xy=None) -> Image.Image:
    import random

    rng = random.Random(seed)
    image = image.convert("RGB")
    out = image.copy()
    x1, y1, x2, y2 = clamp_bbox(bbox, out.width, out.height)
    if effect == "mixed":
        effect = rng.choice(["mask", "blur", "crop"])
    elif effect == "tip_mixed":
        effect = rng.choice(["tip_circle", "tip_square", "tip_blur"])

    if effect == "mask":
        rect = centered_rect(x1, y1, x2, y2, severity)
        ImageDraw.Draw(out).rectangle(rect, fill=(0, 0, 0))
    elif effect == "blur":
        patch = out.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(radius=max(1.0, severity * 8.0)))
        out.paste(patch, (x1, y1))
    elif effect == "crop":
        draw = ImageDraw.Draw(out)
        side = rng.choice(["left", "right", "top", "bottom"])
        if side in {"left", "right"}:
            strip = max(1, int(round((x2 - x1) * max(0.05, min(0.95, severity)))))
            rect = (x1, y1, x1 + strip, y2) if side == "left" else (x2 - strip, y1, x2, y2)
        else:
            strip = max(1, int(round((y2 - y1) * max(0.05, min(0.95, severity)))))
            rect = (x1, y1, x2, y1 + strip) if side == "top" else (x1, y2 - strip, x2, y2)
        draw.rectangle(rect, fill=(0, 0, 0))
    elif effect in {"tip_circle", "tip_square", "tip_blur"}:
        draw_tip_mask(out, (x1, y1, x2, y2), points_xy, effect, severity, seed)
    elif effect == "finger_end":
        draw_finger_end_mask(out, (x1, y1, x2, y2), segments_xy, severity)
    else:
        raise ValueError(f"unsupported effect: {effect}")
    return out


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def save_hard_image(src_image: Path, dst_image: Path, bbox, effect: str, severity: float, seed: int, points_xy=None, segments_xy=None) -> None:
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    hard = apply_hard_effect(Image.open(src_image), bbox, effect, severity, seed, points_xy=points_xy, segments_xy=segments_xy)
    hard.save(dst_image)


def build_freihand_hard_root(
    input_root: Path,
    output_root: Path,
    effect: str,
    severity: float,
    limit: int | None,
    seed: int,
    split: str = "evaluation",
) -> Path:
    Ks = read_json(input_root / f"{split}_K.json")
    verts = read_json(input_root / f"{split}_verts.json")
    xyz = read_json(input_root / f"{split}_xyz.json")
    count = len(Ks) if limit is None or limit <= 0 else min(limit, len(Ks))
    rows = []

    for idx in range(count):
        frame = f"{idx:08d}"
        src_image = resolve_image_path(input_root / split / "rgb", frame)
        dst_image = output_root / split / "rgb" / src_image.name
        bbox = bbox_from_projected_points(verts[idx], Ks[idx]).tolist()
        points_xy = project_fingertips_from_joints(xyz[idx], Ks[idx])
        segments_xy = project_finger_end_segments_from_joints(xyz[idx], Ks[idx])
        sample_seed = seed + idx
        save_hard_image(src_image, dst_image, bbox, effect, severity, sample_seed, points_xy=points_xy, segments_xy=segments_xy)
        rows.append({
            "sample_id": frame,
            "dataset": "freihand",
            "source_image": str(src_image),
            "hard_image": str(dst_image),
            "bbox_xyxy": bbox,
            "points_xy": points_xy,
            "segments_xy": segments_xy,
            "effect": effect,
            "severity": severity,
            "seed": sample_seed,
        })

    write_json(output_root / f"{split}_K.json", Ks[:count])
    write_json(output_root / f"{split}_verts.json", verts[:count])
    write_json(output_root / f"{split}_xyz.json", xyz[:count])
    write_manifest(output_root / "hard_manifest.jsonl", rows)
    return output_root


def build_ho3d_hard_root(
    input_root: Path,
    output_root: Path,
    effect: str,
    severity: float,
    limit: int | None,
    seed: int,
    sample_order_file: Path | None = None,
) -> Path:
    samples = list(iter_ho3d_samples_from_order(input_root, sample_order_file, limit)
                   if sample_order_file else iter_ho3d_samples(input_root, limit))
    xyz = read_json(input_root / "evaluation_xyz.json")
    verts = read_json(input_root / "evaluation_verts.json")
    rows = []

    for idx, sample in enumerate(samples):
        seq, frame = sample.sample_id.split("/")
        dst_image = output_root / "evaluation" / seq / "rgb" / sample.image_path.name
        dst_meta = output_root / "evaluation" / seq / "meta" / f"{frame}.pkl"
        with sample.meta_path.open("rb") as f:
            meta = pickle.load(f, encoding="latin1")
        bbox = hand_bbox_from_meta(meta)[0].tolist()
        K = camera_matrix_from_meta(meta)
        points_xy = project_ho3d_fingertips_from_joints(xyz[idx], K) if K is not None else fingertip_points_from_meta(meta)
        segments_xy = project_ho3d_finger_end_segments_from_joints(xyz[idx], K) if K is not None else []
        sample_seed = seed + idx
        save_hard_image(sample.image_path, dst_image, bbox, effect, severity, sample_seed, points_xy=points_xy, segments_xy=segments_xy)
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.meta_path, dst_meta)
        rows.append({
            "sample_id": sample.sample_id,
            "dataset": "ho3d",
            "source_image": str(sample.image_path),
            "hard_image": str(dst_image),
            "bbox_xyxy": bbox,
            "points_xy": points_xy,
            "segments_xy": segments_xy,
            "effect": effect,
            "severity": severity,
            "seed": sample_seed,
        })

    write_json(output_root / "evaluation_xyz.json", xyz[:len(samples)])
    write_json(output_root / "evaluation_verts.json", verts[:len(samples)])
    (output_root / "evaluation.txt").write_text("".join(f"{sample.sample_id}\n" for sample in samples), encoding="utf-8")
    write_manifest(output_root / "hard_manifest.jsonl", rows)
    return output_root


def iter_ho3d_samples_from_order(root: Path, sample_order_file: Path, limit: int | None):
    payload = read_json(sample_order_file)
    ids = payload["sample_order"] if isinstance(payload, dict) else payload
    if limit is not None and limit > 0:
        ids = ids[:limit]
    eval_dir = root / "evaluation"
    for sample_id in ids:
        seq, frame = sample_id.split("/")
        yield Ho3dSample(
            sample_id=sample_id,
            image_path=resolve_image_path(eval_dir / seq / "rgb", frame),
            meta_path=eval_dir / seq / "meta" / f"{frame}.pkl",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small hard-image benchmark root.")
    parser.add_argument("--dataset", choices=["freihand", "ho3d"], required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--effect",
        choices=["mask", "blur", "crop", "mixed", "tip_circle", "tip_square", "tip_blur", "tip_mixed", "finger_end"],
        default="mask",
    )
    parser.add_argument("--severity", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=32, help="Number of samples; <=0 means all.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", choices=["evaluation", "training"], default="evaluation")
    parser.add_argument("--sample-order-file", type=Path, default=None,
                        help="Optional HO3D run_meta.json or JSON list with sample_order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit <= 0 else args.limit
    if args.dataset == "freihand":
        build_freihand_hard_root(args.input_root, args.output_root, args.effect, args.severity, limit, args.seed, split=args.split)
    else:
        if args.split != "evaluation":
            raise ValueError("HO3D hard-image generation only supports --split evaluation")
        build_ho3d_hard_root(
            args.input_root,
            args.output_root,
            args.effect,
            args.severity,
            limit,
            args.seed,
            sample_order_file=args.sample_order_file,
        )
    print(f"Wrote hard root: {args.output_root}")


if __name__ == "__main__":
    main()
