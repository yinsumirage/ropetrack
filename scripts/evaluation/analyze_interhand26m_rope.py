#!/usr/bin/env python3
"""Post-hoc InterHand rope-effect slices and readable diagnostic figures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.analysis import json_sanitize
from ropetrack.rope import FINGER_CHAINS
from ropetrack.eval.interhand26m import bootstrap_delta, procrustes, read_ground_truth, read_prediction
from ropetrack.datasets.interhand26m_validation import INTERHAND_TO_OPENPOSE, load_mano_layers


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bbox_iou(left, right) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
    return intersection / union if union > 0 else 0.0


def project(points: np.ndarray, intrinsic) -> np.ndarray:
    homogeneous = (np.asarray(intrinsic, dtype=np.float64) @ np.asarray(points, dtype=np.float64).T).T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = points[:, 2] > 1e-9
    result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return result


def pair_geometry(rows: list[dict], xyz: np.ndarray) -> dict[str, np.ndarray]:
    count = len(rows)
    iou = np.full(count, np.nan, dtype=np.float64)
    other_inside = np.full(count, np.nan, dtype=np.float64)
    root_distance = np.full(count, np.nan, dtype=np.float64)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["frame_group_id"]].append(index)
    for indices in groups.values():
        if len(indices) != 2 or {rows[index]["mano_side"] for index in indices} != {"left", "right"}:
            continue
        for index, other in (indices, indices[::-1]):
            row = rows[index]
            iou[index] = bbox_iou(row["bbox_xyxy"], rows[other]["bbox_xyxy"])
            points = project(xyz[other], row["intrinsic"])
            valid = np.asarray(rows[other]["joint_valid"], dtype=bool) & np.isfinite(points).all(axis=1)
            x1, y1, x2, y2 = map(float, row["bbox_xyxy"])
            inside = valid & (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
            other_inside[index] = float(inside.sum() / valid.sum()) if valid.any() else np.nan
            roots = project(np.stack((xyz[index, 0], xyz[other, 0])), row["intrinsic"])
            size = max(x2 - x1, y2 - y1, 1.0)
            root_distance[index] = float(np.linalg.norm(roots[0] - roots[1]) / size)
    return {"bbox_iou": iou, "other_joint_inside_fraction": other_inside, "root_distance_bbox_units": root_distance}


def per_sample_metrics(rows: list[dict], gt: np.ndarray, pred: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    count = len(rows)
    metrics = {name: np.full(count, np.nan, dtype=np.float64) for name in ("pa_mm", "root_mm", "camera_mm")}
    pa_joint = np.full((count, 21), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        valid = np.asarray(row["joint_valid"], dtype=bool)
        if valid.sum() < 3:
            continue
        aligned = procrustes(pred[index, valid], gt[index, valid])
        pa_joint[index, valid] = np.linalg.norm(aligned - gt[index, valid], axis=1) * 1000.0
        metrics["pa_mm"][index] = float(np.nanmean(pa_joint[index]))
        metrics["camera_mm"][index] = float(np.linalg.norm(pred[index, valid] - gt[index, valid], axis=1).mean() * 1000.0)
        pred_root = pred[index] - pred[index, :1]
        gt_root = gt[index] - gt[index, :1]
        metrics["root_mm"][index] = float(np.linalg.norm(pred_root[valid] - gt_root[valid], axis=1).mean() * 1000.0)
    return metrics, pa_joint


def align_cache(path: Path, sample_ids: list[str]) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        source_ids = loaded["sample_id"].astype(str).tolist()
        by_id = {sample_id: index for index, sample_id in enumerate(source_ids)}
        if set(by_id) != set(sample_ids) or len(by_id) != len(source_ids):
            raise ValueError("rope cache sample IDs differ from the val manifest")
        order = np.asarray([by_id[sample_id] for sample_id in sample_ids])
        return {
            key: np.asarray(loaded[key])[order] if len(np.asarray(loaded[key])) == len(source_ids) else np.asarray(loaded[key])
            for key in loaded.files if key != "sample_id"
        }


def rope_residual(cache: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.asarray(cache["rope_valid"], dtype=bool)
    residual = np.where(valid, np.abs(cache["base_rope_norm"] - cache["input_rope_norm"]), 0.0)
    counts = valid.sum(axis=1)
    return np.where(counts, residual.sum(axis=1) / np.maximum(counts, 1), np.nan)


def native_mano_mismatch(
    rows: list[dict], gt: np.ndarray, mano: dict[str, np.ndarray], mano_root: Path, official_root: Path, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    layers = load_mano_layers(mano_root)
    regressor = np.load(official_root / "tool" / "MANO_world_to_camera" / "J_regressor_mano_ih26m.npy").astype(np.float32)
    camera_error = np.full(len(rows), np.nan, dtype=np.float64)
    root_error = np.full(len(rows), np.nan, dtype=np.float64)
    valid_mano = np.asarray(mano["mano_valid"], dtype=bool)
    is_right = np.asarray(mano["is_right"], dtype=bool)
    for side, right in (("right", True), ("left", False)):
        indices = np.flatnonzero(valid_mano & (is_right == right))
        for start in range(0, len(indices), batch_size):
            chosen = indices[start:start + batch_size]
            pose = np.asarray(mano["pose"][chosen], dtype=np.float32)
            shape = np.asarray(mano["shape"][chosen], dtype=np.float32)
            trans = np.asarray(mano["trans_world_m"][chosen], dtype=np.float32)
            with torch.no_grad():
                output = layers[side](
                    global_orient=torch.from_numpy(pose[:, :3]), hand_pose=torch.from_numpy(pose[:, 3:]),
                    betas=torch.from_numpy(shape), transl=torch.from_numpy(trans),
                )
            world = output.vertices.numpy().astype(np.float32)
            rotation = np.asarray(mano["camrot"][chosen], dtype=np.float32)
            position = np.asarray(mano["campos_world_mm"][chosen], dtype=np.float32) / 1000.0
            camera = np.einsum("bij,bkj->bki", rotation, world - position[:, None, :])
            decoded = np.einsum("jv,bvc->bjc", regressor, camera)[:, INTERHAND_TO_OPENPOSE]
            target = gt[chosen]
            valid = np.stack([np.asarray(rows[index]["joint_valid"], dtype=bool) for index in chosen])
            distance = np.linalg.norm(decoded - target, axis=2) * 1000.0
            relative = np.linalg.norm((decoded - decoded[:, :1]) - (target - target[:, :1]), axis=2) * 1000.0
            for local, index in enumerate(chosen):
                camera_error[index] = float(distance[local, valid[local]].mean())
                root_error[index] = float(relative[local, valid[local]].mean())
    return camera_error, root_error


def quantile_labels(values: np.ndarray, buckets: int = 5) -> tuple[np.ndarray, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.full(len(values), "missing", dtype=object), []
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, buckets + 1)[1:-1]))
    labels = np.full(len(values), "missing", dtype=object)
    indices = np.searchsorted(edges, values, side="right")
    labels[np.isfinite(values)] = [f"Q{index + 1}" for index in indices[np.isfinite(values)]]
    return labels, edges.tolist()


def fixed_labels(values: np.ndarray, edges: list[float], prefix: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.full(len(values), "not_paired", dtype=object)
    indices = np.searchsorted(edges, values, side="right")
    labels[np.isfinite(values)] = [f"{prefix}{index}" for index in indices[np.isfinite(values)]]
    return labels


def quantile_representatives(indices: np.ndarray, values: np.ndarray, quantiles=(0.1, 0.5, 0.9)) -> list[int]:
    """Return deterministic observed samples nearest requested rank quantiles."""
    ordered = np.asarray(indices)[np.argsort(np.asarray(values)[indices], kind="stable")]
    if not len(ordered):
        return []
    positions = np.rint(np.asarray(quantiles) * (len(ordered) - 1)).astype(int)
    return ordered[positions].tolist()


def bucket_report(
    labels: np.ndarray, rows: list[dict], candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray], iterations: int, seed: int
) -> list[dict]:
    result = []
    groups = np.asarray([row["frame_group_id"] for row in rows])
    for label in sorted(set(labels.tolist())):
        selected = labels == label
        entry = {"label": label, "count": int(selected.sum())}
        for metric in ("pa_mm", "root_mm", "camera_mm"):
            left, right = candidate[metric][selected], reference[metric][selected]
            finite = np.isfinite(left) & np.isfinite(right)
            entry[f"delta_{metric}"] = float(np.mean(left[finite] - right[finite])) if finite.any() else None
            entry[f"ci95_{metric}"] = bootstrap_delta(left, right, groups[selected].tolist(), iterations, seed)
        result.append(entry)
    return result


def composition(rows: list[dict]) -> dict:
    sizes = np.asarray([max(row["bbox_xyxy"][2] - row["bbox_xyxy"][0], row["bbox_xyxy"][3] - row["bbox_xyxy"][1]) for row in rows])
    return {
        "samples": len(rows),
        "frames": len({row["frame_group_id"] for row in rows}),
        "episodes": len({row["episode_id"] for row in rows}),
        "subjects": len({str(row["subject_id"]) for row in rows}),
        "side": dict(sorted(Counter(row["mano_side"] for row in rows).items())),
        "hand_type": dict(sorted(Counter("interacting" if row["is_interacting"] else "single" for row in rows).items())),
        "capture": dict(sorted(Counter(str(row["capture_id"]) for row in rows).items())),
        "camera_count": len({str(row["camera_id"]) for row in rows}),
        "bbox_size_px": {key: float(value) for key, value in zip(("min", "q25", "median", "q75", "max"), np.quantile(sizes, [0, .25, .5, .75, 1]), strict=True)},
        "projected_joint_count": dict(sorted(Counter(int(row["projected_in_frame_joint_count"]) for row in rows).items())),
    }


def plot_distribution(train_rows: list[dict], val_rows: list[dict], validation_episodes: set[str], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_fit = [row for row in train_rows if row["episode_id"] not in validation_episodes]
    train_dev = [row for row in train_rows if row["episode_id"] in validation_episodes]
    datasets = (("train fit", train_fit), ("internal val", train_dev), ("official val", val_rows))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [name for name, _ in datasets]
    single = [sum(not row["is_interacting"] for row in rows) for _, rows in datasets]
    interacting = [sum(bool(row["is_interacting"]) for row in rows) for _, rows in datasets]
    axes[0, 0].bar(labels, single, label="single")
    axes[0, 0].bar(labels, interacting, bottom=single, label="interacting")
    axes[0, 0].set_title("Sample composition")
    axes[0, 0].legend()
    for name, rows in datasets:
        sizes = [max(row["bbox_xyxy"][2] - row["bbox_xyxy"][0], row["bbox_xyxy"][3] - row["bbox_xyxy"][1]) for row in rows]
        axes[0, 1].hist(sizes, bins=30, density=True, histtype="step", linewidth=2, label=name)
    axes[0, 1].set_title("Per-hand bbox size")
    axes[0, 1].set_xlabel("max(width, height), px")
    axes[0, 1].legend()
    for name, rows in datasets:
        counts = Counter(int(row["projected_in_frame_joint_count"]) for row in rows)
        x = np.arange(4, 22)
        axes[1, 0].plot(x, [counts.get(int(value), 0) / max(len(rows), 1) for value in x], marker="o", label=name)
    axes[1, 0].set_title("Projected joints inside image")
    axes[1, 0].set_xlabel("joint count")
    axes[1, 0].set_ylabel("sample fraction")
    axes[1, 0].legend()
    captures = sorted({str(row["capture_id"]) for row in train_rows + val_rows}, key=lambda value: int(value))
    x = np.arange(len(captures))
    train_counts, val_counts = Counter(str(row["capture_id"]) for row in train_rows), Counter(str(row["capture_id"]) for row in val_rows)
    axes[1, 1].bar(x - .2, [train_counts.get(value, 0) for value in captures], width=.4, label="train27k")
    axes[1, 1].bar(x + .2, [val_counts.get(value, 0) for value in captures], width=.4, label="official val")
    axes[1, 1].set_xticks(x, captures, rotation=90)
    axes[1, 1].set_title("Capture distribution")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_effects(tables: dict[str, list[dict]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, rows) in zip(axes.flat, tables.items(), strict=True):
        shown = [row for row in rows if row["label"] != "not_paired"]
        x = np.arange(len(shown))
        ax.axhline(0, color="black", linewidth=1)
        ax.bar(x - .18, [row["delta_pa_mm"] for row in shown], width=.36, label="PA")
        ax.bar(x + .18, [row["delta_root_mm"] for row in shown], width=.36, label="root-relative")
        ax.set_xticks(x, [f"{row['label']}\n(n={row['count']})" for row in shown], rotation=25)
        ax.set_ylabel("RGB+rope - RGB-only, mm")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_per_finger(rows: list[dict], rgb_pa: np.ndarray, rope_pa: np.ndarray, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("thumb", "index", "middle", "ring", "pinky")
    chains = FINGER_CHAINS["freihand"]
    groups = {
        "single": np.asarray([not row["is_interacting"] for row in rows]),
        "interacting": np.asarray([bool(row["is_interacting"]) for row in rows]),
    }
    table = {}
    for label, selected in groups.items():
        table[label] = [float(np.nanmean(rope_pa[selected][:, chain[1:]] - rgb_pa[selected][:, chain[1:]])) for chain in chains]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(x - .2, table["single"], width=.4, label="single")
    ax.bar(x + .2, table["interacting"], width=.4, label="interacting")
    ax.set_xticks(x, names)
    ax.set_ylabel("RGB+rope - RGB-only PA joint error, mm")
    ax.set_title("Rope effect by finger")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return table


def draw_skeleton(draw, points: np.ndarray, color, y_offset: int, width: int, height: int) -> None:
    inside = (
        np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    for chain in FINGER_CHAINS["freihand"]:
        for left, right in zip(chain, chain[1:], strict=False):
            if inside[left] and inside[right]:
                draw.line((points[left, 0], points[left, 1] + y_offset, points[right, 0], points[right, 1] + y_offset), fill=color, width=2)
    for x, y in points[inside]:
        draw.ellipse((x - 2, y + y_offset - 2, x + 2, y + y_offset + 2), fill=color)


def qualitative_grid(
    rows: list[dict], gt: np.ndarray, rgb: np.ndarray, rope: np.ndarray, delta: np.ndarray, raw_root: Path, output: Path
) -> list[str]:
    from PIL import Image, ImageDraw, ImageFont

    selected = []
    for interacting in (False, True):
        indices = np.flatnonzero(np.asarray([
            bool(row["is_interacting"]) == interacting and row["projected_in_frame_joint_count"] == 21
            for row in rows
        ]) & np.isfinite(delta))
        order = indices[np.argsort(delta[indices])]
        selected.extend(order[:3].tolist())
        selected.extend(order[-3:].tolist())
    panels = []
    ids = []
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["frame_group_id"]].append(index)
    paired = {index: next((other for other in indices if other != index), None) for indices in groups.values() for index in indices if len(indices) == 2}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index in selected:
        row = rows[index]
        image = Image.open(raw_root / row["image_path"]).convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 46), "white")
        panel.paste(image, (0, 46))
        draw = ImageDraw.Draw(panel)
        draw.text((4, 3), f"{'interacting' if row['is_interacting'] else 'single'}  rope-RGB PA {delta[index]:+.2f} mm", fill="black", font=font)
        draw.text((4, 23), "GT green | RGB blue | rope red | other bbox cyan", fill="black", font=font)
        x1, y1, x2, y2 = row["bbox_xyxy"]
        draw.rectangle((x1, y1 + 46, x2, y2 + 46), outline="yellow", width=2)
        if index in paired:
            ox1, oy1, ox2, oy2 = rows[paired[index]]["bbox_xyxy"]
            draw.rectangle((ox1, oy1 + 46, ox2, oy2 + 46), outline="cyan", width=2)
        draw_skeleton(draw, project(gt[index], row["intrinsic"]), (0, 255, 0), 46, image.width, image.height)
        draw_skeleton(draw, project(rgb[index], row["intrinsic"]), (0, 128, 255), 46, image.width, image.height)
        draw_skeleton(draw, project(rope[index], row["intrinsic"]), (255, 0, 0), 46, image.width, image.height)
        panels.append(panel)
        ids.append(row["sample_id"])
    columns = 3
    rows_count = (len(panels) + columns - 1) // columns
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    grid = Image.new("RGB", (columns * cell_w, rows_count * cell_h), "white")
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * cell_w, (index // columns) * cell_h))
    grid.save(output, quality=92)
    return ids


def qualitative_metric_boundary(
    rows: list[dict], gt: np.ndarray, rgb: np.ndarray, rope: np.ndarray,
    metrics: dict[str, dict[str, np.ndarray]], raw_root: Path, output: Path,
) -> list[str]:
    """Show why low PA error can coexist with visibly wrong camera placement."""
    from PIL import Image, ImageDraw, ImageFont

    delta = metrics["rope"]["pa_mm"] - metrics["rgb"]["pa_mm"]
    selected = []
    for interacting in (False, True):
        indices = np.flatnonzero(np.asarray([
            bool(row["is_interacting"]) == interacting
            and row["projected_in_frame_joint_count"] == 21
            and sum(row["joint_valid"]) == 21
            for row in rows
        ]) & np.isfinite(delta))
        selected.extend(quantile_representatives(indices, delta))

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["frame_group_id"]].append(index)
    paired = {index: next((other for other in indices if other != index), None)
              for indices in groups.values() for index in indices if len(indices) == 2}
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    panels = []
    for index in selected:
        row = rows[index]
        image = Image.open(raw_root / row["image_path"]).convert("RGB")
        aligned_rgb = procrustes(rgb[index], gt[index])
        aligned_rope = procrustes(rope[index], gt[index])
        for mode, rgb_points, rope_points in (
            ("RAW CAMERA (real placement)", rgb[index], rope[index]),
            ("PA-ALIGNED (diagnostic only)", aligned_rgb, aligned_rope),
        ):
            panel = Image.new("RGB", (image.width, image.height + 62), "white")
            panel.paste(image, (0, 62))
            draw = ImageDraw.Draw(panel)
            draw.text((4, 3), f"{'interacting' if row['is_interacting'] else 'single'} | {mode}", fill="black", font=font)
            if mode.startswith("RAW"):
                values = f"RGB {metrics['rgb']['camera_mm'][index]:.1f} | rope {metrics['rope']['camera_mm'][index]:.1f} mm"
            else:
                values = f"RGB {metrics['rgb']['pa_mm'][index]:.1f} | rope {metrics['rope']['pa_mm'][index]:.1f} mm"
            draw.text((4, 23), values, fill="black", font=font)
            draw.text((4, 43), "GT green | RGB blue | rope red", fill="black", font=font)
            x1, y1, x2, y2 = row["bbox_xyxy"]
            draw.rectangle((x1, y1 + 62, x2, y2 + 62), outline="yellow", width=2)
            if index in paired:
                ox1, oy1, ox2, oy2 = rows[paired[index]]["bbox_xyxy"]
                draw.rectangle((ox1, oy1 + 62, ox2, oy2 + 62), outline="cyan", width=2)
            draw_skeleton(draw, project(gt[index], row["intrinsic"]), (0, 255, 0), 62, image.width, image.height)
            draw_skeleton(draw, project(rgb_points, row["intrinsic"]), (0, 128, 255), 62, image.width, image.height)
            draw_skeleton(draw, project(rope_points, row["intrinsic"]), (255, 0, 0), 62, image.width, image.height)
            panels.append(panel)

    columns = 4  # two adjacent raw/aligned sample pairs per row
    rows_count = (len(panels) + columns - 1) // columns
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    grid = Image.new("RGB", (columns * cell_w, rows_count * cell_h), "white")
    for index, panel in enumerate(panels):
        grid.paste(panel, ((index % columns) * cell_w, (index // columns) * cell_h))
    grid.save(output, quality=92)
    return [rows[index]["sample_id"] for index in selected]


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# InterHand rope-effect diagnostic", "",
        "Post-hoc official-val analysis. Negative delta means RGB+rope is better than matched RGB-only.", "",
        "## Population", "",
        "| Split | Samples | Frames | Episodes | Subjects | Cameras |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["population"].items():
        lines.append(f"| {name} | {row['samples']} | {row['frames']} | {row['episodes']} | {row['subjects']} | {row['camera_count']} |")
    lines.extend(["", "## Rope minus RGB-only", "", "| Slice | Bin | n | PA delta | Root delta | Camera delta |", "|---|---|---:|---:|---:|---:|"])
    for name, rows in report["effect_buckets"].items():
        for row in rows:
            lines.append(f"| {name} | {row['label']} | {row['count']} | {row['delta_pa_mm']:+.3f} | {row['delta_root_mm']:+.3f} | {row['delta_camera_mm']:+.3f} |")
    lines.extend(["", "## Figures", "", "- `data_distribution.png`", "- `rope_effect_buckets.png`", "- `per_finger_effect.png`", "- `qualitative_extremes.jpg`", "- `qualitative_metric_boundary.jpg` (raw camera vs PA-aligned; aligned panels are diagnostic only)", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--base", nargs=2, required=True, metavar=("PRED", "ORDER"))
    parser.add_argument("--rgb", nargs=2, required=True, metavar=("PRED", "ORDER"))
    parser.add_argument("--rope", nargs=2, required=True, metavar=("PRED", "ORDER"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    args = parser.parse_args(argv)

    train_rows = read_jsonl(args.train_root / "training.jsonl")
    val_rows, gt, mano, _ = read_ground_truth(args.val_root)
    sample_ids = [row["sample_id"] for row in val_rows]
    predictions, metrics, pa_joint = {}, {}, {}
    for name, spec in (("base", args.base), ("rgb", args.rgb), ("rope", args.rope)):
        predictions[name], vertices = read_prediction(Path(spec[0]), Path(spec[1]), sample_ids)
        del vertices
        metrics[name], pa_joint[name] = per_sample_metrics(val_rows, gt, predictions[name])
    cache = align_cache(args.cache, sample_ids)
    residual = rope_residual(cache)
    pair = pair_geometry(val_rows, gt)
    mano_camera, mano_root = native_mano_mismatch(val_rows, gt, mano, args.mano_root, args.official_root, args.batch_size)
    residual_labels, residual_edges = quantile_labels(residual)
    mano_labels, mano_edges = quantile_labels(mano_root)
    effect_labels = {
        "rope residual quintile": residual_labels,
        "paired bbox IoU": fixed_labels(pair["bbox_iou"], [0.01, 0.10, 0.30], "B"),
        "other-hand joints in crop": fixed_labels(pair["other_joint_inside_fraction"], [0.25, 0.50, 0.75], "B"),
        "native MANO mismatch quintile": mano_labels,
    }
    tables = {
        name: bucket_report(labels, val_rows, metrics["rope"], metrics["rgb"], args.bootstrap_iterations, args.bootstrap_seed)
        for name, labels in effect_labels.items()
    }
    protocol = json.loads((args.train_root / "protocol.json").read_text(encoding="utf-8"))
    validation_episodes = set(protocol["internal_validation"]["validation_episodes"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    plot_distribution(train_rows, val_rows, validation_episodes, args.output_root / "data_distribution.png")
    plot_effects(tables, args.output_root / "rope_effect_buckets.png")
    finger = plot_per_finger(val_rows, pa_joint["rgb"], pa_joint["rope"], args.output_root / "per_finger_effect.png")
    qualitative_ids = qualitative_grid(
        val_rows, gt, predictions["rgb"], predictions["rope"], metrics["rope"]["pa_mm"] - metrics["rgb"]["pa_mm"],
        args.raw_root, args.output_root / "qualitative_extremes.jpg",
    )
    metric_boundary_ids = qualitative_metric_boundary(
        val_rows, gt, predictions["rgb"], predictions["rope"], metrics,
        args.raw_root, args.output_root / "qualitative_metric_boundary.jpg",
    )
    report = {
        "dataset": "InterHand2.6M v1.0 30fps",
        "protocol": "interhand26m_v1_30fps_oneview_v1 official-val post-hoc diagnostic",
        "sample_count": len(val_rows),
        "population": {"train27k": composition(train_rows), "official_val": composition(val_rows)},
        "internal_validation": {"episodes": len(validation_episodes), "samples": sum(row["episode_id"] in validation_episodes for row in train_rows)},
        "effect_buckets": tables,
        "bucket_edges": {"rope_residual": residual_edges, "mano_root_mismatch_mm": mano_edges},
        "fixed_bucket_definitions": {
            "paired_bbox_iou": {"B0": "<=0.01", "B1": "(0.01,0.10]", "B2": "(0.10,0.30]", "B3": ">0.30"},
            "other_joint_inside_fraction": {"B0": "<=0.25", "B1": "(0.25,0.50]", "B2": "(0.50,0.75]", "B3": ">0.75"},
        },
        "pair_geometry": {
            "paired_samples": int(np.isfinite(pair["bbox_iou"]).sum()),
            "bbox_iou_quantiles": np.nanquantile(pair["bbox_iou"], [0, .25, .5, .75, .95, 1]).tolist(),
            "other_joint_inside_fraction_quantiles": np.nanquantile(pair["other_joint_inside_fraction"], [0, .25, .5, .75, .95, 1]).tolist(),
        },
        "native_mano_joint_fit_mm": {
            "camera_mean": float(np.nanmean(mano_camera)), "camera_p95": float(np.nanpercentile(mano_camera, 95)),
            "root_mean": float(np.nanmean(mano_root)), "root_p95": float(np.nanpercentile(mano_root, 95)),
        },
        "per_finger_pa_delta_mm": finger,
        "qualitative_sample_ids": qualitative_ids,
        "metric_boundary_sample_ids": metric_boundary_ids,
        "interpretation_boundary": "Post-hoc mechanism analysis only; no checkpoint selection and no test access.",
    }
    output = args.output_root / "report.json"
    output.write_text(json.dumps(json_sanitize(report), indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.output_root / "report.md")
    print(json.dumps({"output": str(output), "population": report["population"], "pair_geometry": report["pair_geometry"]}, indent=2))
    return output


if __name__ == "__main__":
    main()
