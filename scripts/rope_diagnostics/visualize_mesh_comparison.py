from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.visualize.mesh_compare import (  # noqa: E402
    load_mano_faces,
    load_triplets,
    select_triplets,
)


def resolve_sample_image(root: Path, sample_id: str) -> Path:
    if "/" in sample_id:
        seq, frame = sample_id.split("/")
        rgb_dir = root / "evaluation" / seq / "rgb"
    else:
        frame = sample_id
        rgb_dir = root / "evaluation" / "rgb"
    for suffix in (".png", ".jpg", ".jpeg"):
        path = rgb_dir / f"{frame}{suffix}"
        if path.exists():
            return path
    return rgb_dir / f"{frame}.png"


def set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-70)


def draw_mesh(ax, verts: np.ndarray, faces: np.ndarray, color: str, title: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = Poly3DCollection(verts[faces], linewidths=0.02, alpha=0.92)
    mesh.set_facecolor(color)
    mesh.set_edgecolor((0.2, 0.2, 0.2, 0.08))
    ax.add_collection3d(mesh)
    set_equal_axes(ax, verts)
    ax.set_title(title, fontsize=9)


def draw_image(fig, nrows: int, row: int, col: int, columns: int, path: Path, title: str) -> None:
    import matplotlib.image as mpimg

    ax = fig.add_subplot(nrows, columns, row * columns + col + 1)
    ax.imshow(mpimg.imread(path))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def draw_sample(fig, row: int, columns: int, triplet, faces: np.ndarray, image_root: Path | None, hard_image_root: Path | None) -> None:
    nrows = len(fig._triplets)
    if image_root is not None:
        draw_image(fig, nrows, row, 0, columns, resolve_sample_image(image_root, triplet.sample_id), "clean image")
    if hard_image_root is not None:
        draw_image(fig, nrows, row, 1, columns, resolve_sample_image(hard_image_root, triplet.sample_id), "hard image")

    meshes = [
        ("GT", triplet.gt, "#9a9a9a"),
        (f"clean pred\n{triplet.clean_error * 1000:.1f}mm", triplet.clean, "#3f7fca"),
        (f"hard pred\n{triplet.hard_error * 1000:.1f}mm", triplet.hard, "#d14b45"),
        (f"overlay\n+{triplet.degradation * 1000:.1f}mm", None, ""),
    ]
    mesh_col_offset = 2 if image_root is not None and hard_image_root is not None else 0
    for col, (title, verts, color) in enumerate(meshes):
        ax = fig.add_subplot(nrows, columns, row * columns + mesh_col_offset + col + 1, projection="3d")
        if verts is None:
            draw_mesh(ax, triplet.clean, faces, "#3f7fca", title)
            draw_mesh(ax, triplet.hard, faces, "#d14b45", title)
            set_equal_axes(ax, np.concatenate([triplet.clean, triplet.hard], axis=0))
        else:
            draw_mesh(ax, verts, faces, color, title)
        if col == 0:
            ax.text2D(0.0, 1.05, f"{triplet.index} {triplet.sample_id}", transform=ax.transAxes, fontsize=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GT, clean pred, and hard pred meshes from eval runs.")
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--hard-run", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--mano-right", type=Path, default=Path("mano_data/MANO_RIGHT.pkl"))
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--hard-image-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument(
        "--select",
        choices=["first", "worst", "degradation", "middle_degradation", "low_degradation"],
        default="degradation",
    )
    parser.add_argument("--indices", default=None, help="Comma-separated sample indices. Skips full-run ranking.")
    parser.add_argument("--face-step", type=int, default=3, help="Draw every Nth MANO face for faster preview.")
    return parser.parse_args()


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    indices = None
    if args.indices:
        indices = [int(item) for item in args.indices.split(",") if item.strip()]
    triplets = select_triplets(load_triplets(args.clean_run, args.hard_run, args.gt_root, indices=indices), args.count, args.select)
    faces = load_mano_faces(args.mano_right)[::max(1, args.face_step)]

    columns = 6 if args.image_root is not None and args.hard_image_root is not None else 4
    fig = plt.figure(figsize=(18 if columns == 6 else 12, max(2.5, 2.5 * len(triplets))), dpi=160)
    fig._triplets = triplets
    for row, triplet in enumerate(triplets):
        draw_sample(fig, row, columns, triplet, faces, args.image_root, args.hard_image_root)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
