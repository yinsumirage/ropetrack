#!/usr/bin/env python3
"""Report figures from the aggregated runs table (summarize_runs.py output).

Two figures, both consuming runs_summary.json rows:

- ``dose_response``: rope residual closure (x) vs all-joint PA gain in mm (y).
  The P1 story in one picture: the default recipe sits near the origin, the
  strong recipes climb monotonically.
- ``noise``: sensor noise std (x) vs all-joint PA gain in mm (y), one line
  per dropout level. Shows where the sensor-realism claim holds.

Rows are selected with a substring filter on the cell name so the same
summary file can feed multiple figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(summary_path: Path, cell_filter: str) -> list[dict]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = [row for row in rows if cell_filter in (row.get("cell") or "")]
    if not selected:
        raise ValueError(f"no rows match filter '{cell_filter}' in {summary_path}")
    return selected


def gain_mm(row: dict) -> float | None:
    delta = row.get("all_joints_delta_cm")
    if delta is None:
        return None
    return -10.0 * float(delta)  # improvement as positive mm


def plot_dose_response(rows: list[dict], output: Path, title: str) -> None:
    points = [(row["closure"], gain_mm(row), row["cell"]) for row in rows
              if row.get("closure") is not None and gain_mm(row) is not None]
    if not points:
        raise ValueError("no rows with both closure and all_joints_delta_cm")
    points.sort()
    closures = [p[0] for p in points]
    gains = [p[1] for p in points]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(closures, gains, "o-", color="#1f77b4")
    for closure, gain, cell in points:
        ax.annotate(Path(cell).name, (closure, gain), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("rope residual closure")
    ax.set_ylabel("all-joint PA improvement (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_noise(rows: list[dict], output: Path, title: str) -> None:
    by_dropout: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        gain = gain_mm(row)
        noise = row.get("noise_std")
        if gain is None or noise is None:
            continue
        by_dropout.setdefault(float(row.get("dropout") or 0.0), []).append((float(noise), gain))
    if not by_dropout:
        raise ValueError("no rows with noise_std and all_joints_delta_cm")
    fig, ax = plt.subplots(figsize=(6, 4))
    for dropout, points in sorted(by_dropout.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], "o-", label=f"dropout={dropout:g}")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("rope sensor noise std (normalized units)")
    ax.set_ylabel("all-joint PA improvement (mm)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report figures from runs_summary.json.")
    parser.add_argument("--summary", type=Path, required=True, help="runs_summary.json from summarize_runs.py.")
    parser.add_argument("--figure", choices=["dose_response", "noise"], required=True)
    parser.add_argument("--cell-filter", default="", help="Substring filter on cell names (empty = all rows).")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output image path (.png).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    rows = read_rows(args.summary, args.cell_filter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or args.figure.replace("_", " ")
    if args.figure == "dose_response":
        plot_dose_response(rows, args.output, title)
    else:
        plot_noise(rows, args.output, title)
    print(f"Wrote figure: {args.output}")
    return args.output


if __name__ == "__main__":
    main()
