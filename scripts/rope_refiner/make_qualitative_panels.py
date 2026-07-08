#!/usr/bin/env python3
"""Qualitative report panels: base vs named variants (teacher/student) vs GT.

Selects the samples where the ranking variant improves most over the base
prediction, then renders one figure per sample:

- if the dataset is FreiHAND and the hard root provides images + K, the first
  panel shows the hard image with the GT skeleton projected on it;
- one 3D skeleton panel per prediction source (base + each variant), GT in
  green underneath, per-panel PA error in the title.

Skeletons come from the dataset's FINGER_CHAINS, PA alignment is identical to
scripts/score_predictions.py, so panel titles match the report tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.datasets.hand_pose import project_points, resolve_image_path  # noqa: E402
from ropetrack.io import load_pred_json, read_json  # noqa: E402
from ropetrack.rope import FINGER_CHAINS, FINGER_COLORS, canonical_rope_dataset  # noqa: E402

from scripts.score_predictions import align_w_scale  # noqa: E402


def load_pred_xyz(path: Path) -> np.ndarray:
    xyz, _ = load_pred_json(path)
    return np.asarray(xyz, dtype=np.float64)


def pa_errors_mm(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample mean PA error (mm) and aligned predictions."""
    errors = np.zeros(gt_xyz.shape[0], dtype=np.float64)
    aligned = np.zeros_like(pred_xyz)
    for idx in range(gt_xyz.shape[0]):
        aligned[idx] = align_w_scale(gt_xyz[idx], pred_xyz[idx])
        errors[idx] = float(np.linalg.norm(gt_xyz[idx] - aligned[idx], axis=1).mean() * 1000.0)
    return errors, aligned


def draw_skeleton_3d(ax, joints: np.ndarray, dataset: str, color, label: str, linewidth: float = 1.6) -> None:
    for chain, finger_color in zip(FINGER_CHAINS[canonical_rope_dataset(dataset)], FINGER_COLORS):
        pts = joints[list(chain)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color or finger_color, linewidth=linewidth, label=None)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color or "#333333", s=6, label=label)


def render_sample(
    output: Path,
    dataset: str,
    sample_index: int,
    sample_id: str,
    gt: np.ndarray,
    variants: list[tuple[str, np.ndarray, float]],
    image_path: Path | None,
    K: np.ndarray | None,
) -> None:
    num_panels = len(variants) + (1 if image_path is not None and image_path.exists() else 0)
    fig = plt.figure(figsize=(3.2 * num_panels, 3.6))
    panel = 1

    if image_path is not None and image_path.exists():
        ax = fig.add_subplot(1, num_panels, panel)
        panel += 1
        from PIL import Image

        ax.imshow(np.asarray(Image.open(image_path).convert("RGB")))
        if K is not None:
            uv = project_points(gt, K)
            for chain, color in zip(FINGER_CHAINS[canonical_rope_dataset(dataset)], FINGER_COLORS):
                pts = uv[list(chain)]
                ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)
        ax.set_title("hard image + GT", fontsize=9)
        ax.axis("off")

    for name, aligned, error_mm in variants:
        ax = fig.add_subplot(1, num_panels, panel, projection="3d")
        panel += 1
        draw_skeleton_3d(ax, gt, dataset, color="#20a050", label="GT", linewidth=2.2)
        draw_skeleton_3d(ax, aligned, dataset, color="#d14b45", label=name, linewidth=1.4)
        ax.set_title(f"{name}: {error_mm:.2f} mm", fontsize=9)
        ax.set_axis_off()
        ax.view_init(elev=-80, azim=-90)

    fig.suptitle(f"#{sample_index} {sample_id}", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"variant must be name=path, got {value!r}")
    name, _, path = value.partition("=")
    return name, Path(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render base/variant/GT qualitative skeleton panels.")
    parser.add_argument("--dataset", choices=["freihand", "ho3d"], required=True)
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory with {set_name}_xyz.json (and _K.json for freihand overlays).")
    parser.add_argument("--set-name", default="evaluation")
    parser.add_argument("--base", type=Path, required=True, help="base_pred.json path.")
    parser.add_argument("--variant", type=parse_variant, action="append", required=True,
                        help="Repeatable name=pred.json (e.g. teacher=.../pred.json student=.../pred.json).")
    parser.add_argument("--hard-root", type=Path, default=None, help="Hard root for image panels (freihand: {set_name}/rgb).")
    parser.add_argument("--rank-variant", default=None, help="Variant name used for improvement ranking; defaults to the last one.")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    gt_xyz = np.asarray(read_json(args.gt_dir / f"{args.set_name}_xyz.json"), dtype=np.float64)
    base_xyz = load_pred_xyz(args.base)
    if base_xyz.shape != gt_xyz.shape:
        raise ValueError(f"base shape {base_xyz.shape} != gt shape {gt_xyz.shape}")

    variant_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    base_err, base_aligned = pa_errors_mm(gt_xyz, base_xyz)
    for name, path in args.variant:
        pred = load_pred_xyz(path)
        if pred.shape != gt_xyz.shape:
            raise ValueError(f"variant {name} shape {pred.shape} != gt shape {gt_xyz.shape}")
        variant_data[name] = pa_errors_mm(gt_xyz, pred)

    rank_name = args.rank_variant or args.variant[-1][0]
    if rank_name not in variant_data:
        raise ValueError(f"--rank-variant {rank_name!r} not among variants {sorted(variant_data)}")
    improvement = base_err - variant_data[rank_name][0]
    top = np.argsort(-improvement)[: args.top_k]

    Ks = None
    if canonical_rope_dataset(args.dataset) == "freihand":
        k_path = args.gt_dir / f"{args.set_name}_K.json"
        if k_path.exists():
            Ks = read_json(k_path)

    manifest = []
    for rank, sample_index in enumerate(top.tolist()):
        image_path = None
        K = None
        if args.hard_root is not None and canonical_rope_dataset(args.dataset) == "freihand":
            image_path = resolve_image_path(args.hard_root / args.set_name / "rgb", f"{sample_index:08d}")
            K = np.asarray(Ks[sample_index]) if Ks is not None else None
        variants = [("base", base_aligned[sample_index], float(base_err[sample_index]))]
        for name, (errors, aligned) in variant_data.items():
            variants.append((name, aligned[sample_index], float(errors[sample_index])))
        output = args.output_dir / f"panel_{rank:02d}_sample{sample_index:08d}.png"
        render_sample(output, args.dataset, sample_index, f"{sample_index:08d}", gt_xyz[sample_index], variants, image_path, K)
        manifest.append({
            "rank": rank,
            "sample_index": int(sample_index),
            "base_pa_mm": float(base_err[sample_index]),
            "improvement_mm": float(improvement[sample_index]),
            "variant_pa_mm": {name: float(errors[sample_index]) for name, (errors, _) in variant_data.items()},
            "panel": output.name,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "panels_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} panels to {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
