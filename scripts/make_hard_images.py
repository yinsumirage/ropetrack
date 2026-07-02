from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_freihand import bbox_from_projected_points, read_json  # noqa: E402
from bench_ho3d import Ho3dSample, hand_bbox_from_meta, iter_ho3d_samples, resolve_image_path  # noqa: E402


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


def apply_hard_effect(image: Image.Image, bbox, effect: str, severity: float, seed: int) -> Image.Image:
    import random

    rng = random.Random(seed)
    image = image.convert("RGB")
    out = image.copy()
    x1, y1, x2, y2 = clamp_bbox(bbox, out.width, out.height)
    if effect == "mixed":
        effect = rng.choice(["mask", "blur", "crop"])

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


def save_hard_image(src_image: Path, dst_image: Path, bbox, effect: str, severity: float, seed: int) -> None:
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    hard = apply_hard_effect(Image.open(src_image), bbox, effect, severity, seed)
    hard.save(dst_image)


def build_freihand_hard_root(
    input_root: Path,
    output_root: Path,
    effect: str,
    severity: float,
    limit: int | None,
    seed: int,
) -> Path:
    Ks = read_json(input_root / "evaluation_K.json")
    verts = read_json(input_root / "evaluation_verts.json")
    xyz = read_json(input_root / "evaluation_xyz.json")
    count = len(Ks) if limit is None or limit <= 0 else min(limit, len(Ks))
    rows = []

    for idx in range(count):
        frame = f"{idx:08d}"
        src_image = resolve_image_path(input_root / "evaluation" / "rgb", frame)
        dst_image = output_root / "evaluation" / "rgb" / src_image.name
        bbox = bbox_from_projected_points(verts[idx], Ks[idx]).tolist()
        sample_seed = seed + idx
        save_hard_image(src_image, dst_image, bbox, effect, severity, sample_seed)
        rows.append({
            "sample_id": frame,
            "dataset": "freihand",
            "source_image": str(src_image),
            "hard_image": str(dst_image),
            "bbox_xyxy": bbox,
            "effect": effect,
            "severity": severity,
            "seed": sample_seed,
        })

    write_json(output_root / "evaluation_K.json", Ks[:count])
    write_json(output_root / "evaluation_verts.json", verts[:count])
    write_json(output_root / "evaluation_xyz.json", xyz[:count])
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
        sample_seed = seed + idx
        save_hard_image(sample.image_path, dst_image, bbox, effect, severity, sample_seed)
        dst_meta.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.meta_path, dst_meta)
        rows.append({
            "sample_id": sample.sample_id,
            "dataset": "ho3d",
            "source_image": str(sample.image_path),
            "hard_image": str(dst_image),
            "bbox_xyxy": bbox,
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
    parser.add_argument("--effect", choices=["mask", "blur", "crop", "mixed"], default="mask")
    parser.add_argument("--severity", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=32, help="Number of samples; <=0 means all.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-order-file", type=Path, default=None,
                        help="Optional HO3D run_meta.json or JSON list with sample_order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit <= 0 else args.limit
    if args.dataset == "freihand":
        build_freihand_hard_root(args.input_root, args.output_root, args.effect, args.severity, limit, args.seed)
    else:
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
