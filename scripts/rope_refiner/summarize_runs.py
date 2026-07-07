#!/usr/bin/env python3
"""Aggregate apply/score outputs from many run cells into one table.

Walks the given roots for directories containing a ``summary.json`` written
by apply_rope_refinement.py, then merges (when present alongside it):

- ``summary.json``: mode/objective/action space, gate, sensor perturbation,
  rope residual closure, alpha stats;
- ``sliced/sliced_scores.json`` (or ``sliced_scores.json``): same-decoder
  base/refined/delta for all_joints, fingertips, occluded/clean slices;
- ``scores/scores.json``/``scores/scores.txt`` (or root-level equivalents): absolute benchmark metrics
  (PA joints/mesh, F-scores) for the refined prediction.

Emits a TSV and a Markdown table so report tables are generated, not
hand-copied (transcription slips in earlier notes motivated this).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.refine.analysis import json_sanitize  # noqa: E402

COLUMNS = (
    "cell",
    "mode",
    "objective",
    "action_space",
    "gate_threshold",
    "noise_std",
    "dropout",
    "closure",
    "alpha_mean_abs",
    "pa_cm",
    "mesh_al_cm",
    "f_al_score_5",
    "all_joints_base_cm",
    "all_joints_delta_cm",
    "occluded_tip_delta_cm",
    "clean_tip_delta_cm",
    "frac_fingers_gated",
    "num_samples",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_scores(path: Path) -> dict:
    if path.suffix == ".json":
        return read_json(path)
    scores = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            scores[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return scores


def find_first(cell_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = cell_dir / name
        if path.is_file():
            return path
    return None


def collect_cell(cell_dir: Path, root: Path) -> dict:
    summary = read_json(cell_dir / "summary.json")
    row: dict = {key: None for key in COLUMNS}
    row["cell"] = cell_dir.relative_to(root).as_posix() if cell_dir != root else cell_dir.name
    row["mode"] = summary.get("mode")
    row["objective"] = summary.get("objective")
    row["action_space"] = summary.get("action_space")
    row["num_samples"] = summary.get("num_samples")
    row["closure"] = summary.get("rope_residual", {}).get("closure_frac")
    row["alpha_mean_abs"] = summary.get("alpha", {}).get("mean_abs")
    row["gate_threshold"] = (
        summary.get("gating", {}).get("threshold")
        if "gating" in summary
        else summary.get("optimization", {}).get("gate_residual_threshold")
    )
    row["frac_fingers_gated"] = summary.get("gating", {}).get("frac_fingers_gated")
    sensor = summary.get("rope_sensor", {})
    row["noise_std"] = sensor.get("noise_std")
    row["dropout"] = sensor.get("dropout")

    sliced_path = find_first(cell_dir, ("sliced/sliced_scores.json", "sliced_scores.json"))
    if sliced_path is not None:
        slices = read_json(sliced_path).get("slices", {})
        all_joints = slices.get("all_joints", {})
        row["all_joints_base_cm"] = all_joints.get("base_cm")
        row["all_joints_delta_cm"] = all_joints.get("delta_cm")
        row["occluded_tip_delta_cm"] = slices.get("occluded_fingertips", {}).get("delta_cm")
        row["clean_tip_delta_cm"] = slices.get("clean_fingertips", {}).get("delta_cm")

    scores_path = find_first(cell_dir, ("scores/scores.json", "scores.json", "scores/scores.txt", "scores.txt"))
    if scores_path is not None:
        scores = read_scores(scores_path)
        row["pa_cm"] = scores.get("xyz_procrustes_al_mean3d")
        row["mesh_al_cm"] = scores.get("mesh_al_mean3d")
        row["f_al_score_5"] = scores.get("f_al_score_5")

    return row


def discover_cells(roots: list[Path]) -> list[tuple[Path, Path]]:
    """(cell_dir, owning_root) pairs for every summary.json below the roots."""
    pairs = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"run root does not exist: {root}")
        for summary in sorted(root.rglob("summary.json")):
            pairs.append((summary.parent, root))
    if not pairs:
        raise ValueError(f"no summary.json found under: {', '.join(str(r) for r in roots)}")
    return pairs


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_tsv(path: Path, rows: list[dict]) -> None:
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        lines.append("\t".join(format_value(row[key]) for key in COLUMNS))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[key]) for key in COLUMNS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate rope run cells into one report table.")
    parser.add_argument("roots", type=Path, nargs="+", help="Run roots to scan recursively for summary.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sort-by", default="cell", choices=list(COLUMNS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    rows = [collect_cell(cell_dir, root) for cell_dir, root in discover_cells(args.roots)]

    def sort_key(row: dict):
        value = row[args.sort_by]
        return (value is None, value if value is not None else "")

    rows.sort(key=sort_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "runs_summary.tsv", rows)
    write_markdown(args.output_dir / "runs_summary.md", rows)
    (args.output_dir / "runs_summary.json").write_text(
        json.dumps(json_sanitize(rows), indent=2), encoding="utf-8"
    )
    print(f"Aggregated {len(rows)} cells into {args.output_dir}")
    return args.output_dir


if __name__ == "__main__":
    main()
