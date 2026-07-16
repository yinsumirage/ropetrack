#!/usr/bin/env python3
"""Audit EgoDex confidence, temporal geometry, and low-confidence overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ropetrack.datasets.hand_pose import project_points  # noqa: E402
from ropetrack.io import load_pred_json, read_json  # noqa: E402
from ropetrack.rope import FINGER_CHAINS, FINGER_COLORS  # noqa: E402


TIP_IDS = np.asarray([4, 8, 12, 16, 20])
EDGES = tuple((a, b) for chain in FINGER_CHAINS["egodex"] for a, b in zip(chain, chain[1:]))
QUANTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


def percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {f"p{q}": float(v) for q, v in zip(QUANTILES, np.percentile(values, QUANTILES), strict=True)}


def pa_errors_mm(gt: np.ndarray, pred: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Vectorized version of score_predictions.align_w_scale, returned per row."""
    if gt.shape != pred.shape or gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError(f"GT/pred shapes must match [N, J, 3], got {gt.shape} and {pred.shape}")
    errors = np.empty(len(gt), dtype=np.float64)
    for start in range(0, len(gt), batch_size):
        end = min(start + batch_size, len(gt))
        g = np.asarray(gt[start:end], dtype=np.float64)
        p = np.asarray(pred[start:end], dtype=np.float64)
        g_mean = g.mean(axis=1, keepdims=True)
        p_mean = p.mean(axis=1, keepdims=True)
        g0 = g - g_mean
        p0 = p - p_mean
        g_scale = np.linalg.norm(g0, axis=(1, 2)).clip(1e-8)
        p_scale = np.linalg.norm(p0, axis=(1, 2)).clip(1e-8)
        gn = g0 / g_scale[:, None, None]
        pn = p0 / p_scale[:, None, None]
        covariance = np.matmul(np.swapaxes(gn, 1, 2), pn)
        u, singular_values, vt = np.linalg.svd(covariance, full_matrices=False)
        rotation = np.matmul(u, vt)
        aligned = np.matmul(pn, np.swapaxes(rotation, 1, 2))
        aligned *= (singular_values.sum(axis=1) * g_scale)[:, None, None]
        aligned += g_mean
        errors[start:end] = np.linalg.norm(g - aligned, axis=2).mean(axis=1) * 1000.0
    return errors


def select_diverse(order: np.ndarray, rows: list[dict], count: int) -> list[int]:
    selected, episodes = [], set()
    for idx in np.asarray(order, dtype=np.int64).tolist():
        episode = str(rows[idx]["episode_id"])
        if episode in episodes:
            continue
        episodes.add(episode)
        selected.append(idx)
        if len(selected) == count:
            break
    return selected


def confidence_color(value: float) -> tuple[int, int, int]:
    if value < 0.25:
        return (220, 45, 45)
    if value < 0.5:
        return (240, 145, 25)
    if value < 0.75:
        return (235, 205, 40)
    return (35, 180, 80)


def draw_skeleton(image: Image.Image, row: dict, joints: np.ndarray) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    uv = project_points(joints, row["intrinsic"])
    confidence = np.asarray(row["joint_confidence"], dtype=np.float64)
    for chain, color in zip(FINGER_CHAINS["egodex"], FINGER_COLORS, strict=True):
        points = [tuple(map(float, uv[idx])) for idx in chain]
        draw.line(points, fill=color, width=5)
    for idx, (x, y) in enumerate(uv):
        radius = 7 if idx in TIP_IDS else 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=confidence_color(confidence[idx]), outline="black", width=2)
    draw.rectangle(tuple(map(float, row["bbox_xyxy"])), outline="white", width=3)
    return out


def fit_panel(image: Image.Image, width: int = 720, height: int = 480) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "#202020")
    panel.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return panel


def render_overlay(output: Path, row: dict, joints: np.ndarray, category: str) -> None:
    image = Image.open(row["absolute_image_path"]).convert("RGB")
    overlay = draw_skeleton(image, row, joints)
    x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
    size = max(x2 - x1, y2 - y1, 260.0) * 1.8
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_box = (
        max(0, int(cx - size / 2)),
        max(0, int(cy - size / 2)),
        min(image.width, int(cx + size / 2)),
        min(image.height, int(cy + size / 2)),
    )
    canvas = Image.new("RGB", (1440, 570), "white")
    canvas.paste(fit_panel(overlay), (0, 60))
    canvas.paste(fit_panel(overlay.crop(crop_box)), (720, 60))
    confidence = np.asarray(row["joint_confidence"], dtype=np.float64)
    title = (
        f"{category} | {row['sample_id']} | {'right' if row['is_right'] else 'left'} | "
        f"mean={confidence.mean():.3f} tip_mean={confidence[TIP_IDS].mean():.3f} min={confidence.min():.3f}"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), title, fill="black", font=ImageFont.load_default())
    draw.text((18, 548), "joint confidence: red < .25, orange < .50, yellow < .75, green >= .75", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    images = [Image.open(path).convert("RGB") for path in paths]
    width = 900
    thumbs = []
    for image in images:
        image.thumbnail((width, 340), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    sheet = Image.new("RGB", (width, sum(image.height for image in thumbs)), "white")
    y = 0
    for image in thumbs:
        sheet.paste(image, (0, y))
        y += image.height
    sheet.save(output)


def score_by_confidence(
    errors: dict[str, np.ndarray], tip_mean: np.ndarray, has_native_confidence: np.ndarray
) -> dict:
    bounds = (0.0, 0.25, 0.5, 0.75, 0.9, 1.000001)
    rows = []
    for low, high in zip(bounds, bounds[1:]):
        mask = has_native_confidence & (tip_mean >= low) & (tip_mean < high)
        item = {"tip_confidence": f"[{low:.2f},{min(high, 1.0):.2f})", "count": int(mask.sum())}
        for name, values in errors.items():
            item[f"{name}_pa_mm"] = float(values[mask].mean()) if mask.any() else None
        if "base" in errors:
            for name, values in errors.items():
                if name != "base":
                    item[f"{name}_delta_vs_base_mm"] = float((values[mask] - errors["base"][mask]).mean()) if mask.any() else None
        rows.append(item)
    missing = ~has_native_confidence
    if missing.any():
        item = {"tip_confidence": "missing", "count": int(missing.sum())}
        for name, values in errors.items():
            item[f"{name}_pa_mm"] = float(values[missing].mean())
        if "base" in errors:
            for name, values in errors.items():
                if name != "base":
                    item[f"{name}_delta_vs_base_mm"] = float((values[missing] - errors["base"][missing]).mean())
        rows.append(item)
    native_count = int(has_native_confidence.sum())
    correlation = None
    if "base" in errors and native_count >= 2:
        native_tip = tip_mean[has_native_confidence]
        native_error = errors["base"][has_native_confidence]
        if native_tip.std() > 0.0 and native_error.std() > 0.0:
            correlation = float(np.corrcoef(native_tip, native_error)[0, 1])
    return {
        "bins": rows,
        "correlation_tip_confidence_base_error": correlation,
    }


def parse_prediction(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError("prediction must be name=/path/to/pred.json")
    return name, Path(path)


def run(args: argparse.Namespace) -> Path:
    manifest_path = args.root / f"{args.split}.jsonl"
    rows = [json.loads(line) for line in manifest_path.open("r", encoding="utf-8")]
    joints = np.asarray(read_json(args.root / f"{args.split}_xyz.json"), dtype=np.float32)
    if joints.shape != (len(rows), 21, 3):
        raise ValueError(f"manifest/xyz mismatch: rows={len(rows)} xyz={joints.shape}")
    for row in rows:
        row["absolute_image_path"] = str(args.root / row["image_path"])

    confidence = np.asarray([row["joint_confidence"] for row in rows], dtype=np.float32)
    mean_conf = confidence.mean(axis=1)
    tip_mean = confidence[:, TIP_IDS].mean(axis=1)
    min_conf = confidence.min(axis=1)
    bones = np.stack([np.linalg.norm(joints[:, b] - joints[:, a], axis=1) for a, b in EDGES], axis=1)

    temporal_speed = np.full(len(rows), np.nan, dtype=np.float32)
    temporal_bone_change = np.full(len(rows), np.nan, dtype=np.float32)
    previous: dict[tuple[str, bool], tuple[int, np.ndarray, np.ndarray]] = {}
    for idx, row in enumerate(rows):
        key = (str(row["episode_id"]), bool(row["is_right"]))
        frame = int(row["frame_index"])
        if key in previous:
            prev_frame, prev_joints, prev_bones = previous[key]
            dt = (frame - prev_frame) / 30.0
            if 0.0 < dt <= 1.0:
                relative = joints[idx] - joints[idx, :1]
                prev_relative = prev_joints - prev_joints[:1]
                temporal_speed[idx] = float(np.linalg.norm(relative - prev_relative, axis=1).max() / dt)
                temporal_bone_change[idx] = float(np.max(np.abs(bones[idx] - prev_bones) / np.maximum(prev_bones, 1e-3)))
        previous[key] = (frame, joints[idx], bones[idx])

    missing_confidence_files = []
    if args.raw_root is not None:
        import h5py

        for source in sorted({row["source_hdf5"] for row in rows}):
            with h5py.File(args.raw_root / source, "r") as h5:
                if "confidences" not in h5:
                    missing_confidence_files.append(source)
    missing_confidence_sources = set(missing_confidence_files)
    has_native_confidence = np.asarray(
        [row["source_hdf5"] not in missing_confidence_sources for row in rows], dtype=bool
    )

    prediction_errors = {}
    for name, path in args.prediction:
        xyz, _ = load_pred_json(path)
        pred = np.asarray(xyz, dtype=np.float32)
        prediction_errors[name] = pa_errors_mm(joints, pred)

    valid_speed = np.isfinite(temporal_speed)
    valid_bone_change = np.isfinite(temporal_bone_change)
    audit = {
        "samples": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "confidence": {
            "joint": percentiles(confidence),
            "sample_mean": percentiles(mean_conf),
            "sample_tip_mean": percentiles(tip_mean),
            "sample_min": percentiles(min_conf),
            "fraction_joint_below": {str(t): float((confidence < t).mean()) for t in (0.01, 0.25, 0.5, 0.75)},
            "fraction_tip_below": {str(t): float((confidence[:, TIP_IDS] < t).mean()) for t in (0.01, 0.25, 0.5, 0.75)},
            "fraction_samples_all_one": float(np.all(confidence == 1.0, axis=1).mean()),
            "fraction_native_samples_all_one": float(
                np.all(confidence[has_native_confidence] == 1.0, axis=1).mean()
            ),
            "missing_confidence_hdf5": len(missing_confidence_files),
            "rows_missing_native_confidence": int((~has_native_confidence).sum()),
            "fraction_rows_missing_native_confidence": float((~has_native_confidence).mean()),
            "missing_confidence_sources": missing_confidence_files,
        },
        "geometry": {
            "bone_length_mm": percentiles(bones * 1000.0),
            "fraction_nonfinite_rows": float((~np.isfinite(joints).all(axis=(1, 2))).mean()),
            "fraction_any_joint_behind_camera": float((joints[:, :, 2] <= 0.0).any(axis=1).mean()),
            "max_wrist_relative_speed_m_s": percentiles(temporal_speed),
            "adjacent_bone_relative_change": percentiles(temporal_bone_change),
            "temporal_pairs": int(valid_speed.sum()),
            "fraction_temporal_speed_over_0.5_m_s": float((temporal_speed[valid_speed] > 0.5).mean()),
            "fraction_bone_change_over_10pct": float((temporal_bone_change[valid_bone_change] > 0.10).mean()),
            "fraction_bone_change_over_25pct": float((temporal_bone_change[valid_bone_change] > 0.25).mean()),
        },
        "prediction": (
            score_by_confidence(prediction_errors, tip_mean, has_native_confidence)
            if prediction_errors else None
        ),
    }

    selections = {
        "low_tip": select_diverse(np.argsort(tip_mean), rows, args.low_count),
        "low_mean": select_diverse(np.argsort(mean_conf), rows, args.low_count),
        "temporal_jump": select_diverse(np.argsort(np.nan_to_num(temporal_speed, nan=-1.0))[::-1], rows, args.jump_count),
        "high_tip": select_diverse(
            np.argsort(np.where(has_native_confidence, tip_mean, -np.inf))[::-1], rows, args.high_count
        ),
    }
    visual_manifest = []
    for category, indices in selections.items():
        paths = []
        for rank, idx in enumerate(indices):
            path = args.output_dir / category / f"{rank:02d}_{idx:06d}.png"
            render_overlay(path, rows[idx], joints[idx], category)
            paths.append(path)
            visual_manifest.append({
                "category": category,
                "rank": rank,
                "sample_index": idx,
                "sample_id": rows[idx]["sample_id"],
                "episode_id": rows[idx]["episode_id"],
                "is_right": bool(rows[idx]["is_right"]),
                "mean_confidence": float(mean_conf[idx]),
                "tip_mean_confidence": float(tip_mean[idx]),
                "min_confidence": float(min_conf[idx]),
                "temporal_speed_m_s": float(temporal_speed[idx]) if np.isfinite(temporal_speed[idx]) else None,
                "path": str(path.relative_to(args.output_dir)),
            })
        make_contact_sheet(paths, args.output_dir / f"contact_{category}.jpg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "visual_manifest.json").write_text(json.dumps(visual_manifest, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False))
    return args.output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--prediction", type=parse_prediction, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-count", type=int, default=8)
    parser.add_argument("--jump-count", type=int, default=6)
    parser.add_argument("--high-count", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
