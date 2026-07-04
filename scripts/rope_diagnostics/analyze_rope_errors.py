#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.io import read_jsonl


GT_BINS = (("closed", 0.0, 0.33), ("mid", 0.33, 0.67), ("open", 0.67, 1.01))


def parse_run_name(name: str) -> dict[str, str]:
    parts = name.split("_")
    if name.startswith("clean_"):
        return {"split": "clean", "dataset": "_".join(parts[1:-1]), "hard_kind": "clean", "backend": parts[-1]}
    if not name.startswith("hard_"):
        return {"split": "unknown", "dataset": "unknown", "hard_kind": "unknown", "backend": parts[-1]}
    backend = parts[-1]
    rest = name[len("hard_") : -len("_" + backend)]
    if rest.startswith("ho3d_v2_"):
        return {"split": "hard", "dataset": "ho3d_v2", "hard_kind": rest[len("ho3d_v2_") :], "backend": backend}
    dataset, hard_kind = rest.split("_", 1)
    return {"split": "hard", "dataset": dataset, "hard_kind": hard_kind, "backend": backend}


def mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def fmt(value) -> str:
    return "" if value is None else f"{float(value):.6f}"


def iter_finger_values(row: dict):
    fingers = row.get("finger_order") or [f"finger_{i}" for i in range(len(row["gt_rope_norm"]))]
    for idx, finger in enumerate(fingers):
        gt = row["gt_rope_norm"][idx]
        pred = row["pred_rope_norm"][idx]
        err = row["rope_norm_abs_error"][idx]
        if gt is not None and pred is not None and err is not None:
            yield finger, float(gt), float(pred), float(err)


def bin_name(gt: float) -> str:
    for name, lo, hi in GT_BINS:
        if lo <= gt < hi:
            return name
    return "out_of_range"


def summarize_run(run_name: str, rows: list[dict], top_k: int = 20):
    meta = parse_run_name(run_name)
    values = [(finger, gt, pred, err) for row in rows for finger, gt, pred, err in iter_finger_values(row)]
    errors = [err for _, _, _, err in values]
    biases = [pred - gt for _, gt, pred, _ in values]
    summary = {
        "run": run_name,
        **meta,
        "num_samples": len(rows),
        "num_valid_fingers": len(values),
        "rope_norm_mae": mean(errors),
        "rope_norm_bias": mean(biases),
    }

    fingers = []
    for row in rows:
        for finger in row.get("finger_order") or []:
            if finger not in fingers:
                fingers.append(finger)
    per_finger = []
    for finger in fingers:
        vals = [(gt, pred, err) for f, gt, pred, err in values if f == finger]
        per_finger.append({
            "run": run_name,
            **meta,
            "finger": finger,
            "count": len(vals),
            "mae": mean(err for _, _, err in vals),
            "bias": mean(pred - gt for gt, pred, _ in vals),
            "gt_mean": mean(gt for gt, _, _ in vals),
            "pred_mean": mean(pred for _, pred, _ in vals),
        })

    bins = []
    for name, _, _ in GT_BINS:
        vals = [(gt, pred, err) for _, gt, pred, err in values if bin_name(gt) == name]
        bins.append({
            "run": run_name,
            **meta,
            "gt_bin": name,
            "count": len(vals),
            "mae": mean(err for _, _, err in vals),
            "bias": mean(pred - gt for gt, pred, _ in vals),
        })

    worst = sorted(rows, key=lambda row: row.get("rope_norm_mae") or -1.0, reverse=True)[:top_k]
    worst_rows = []
    for rank, row in enumerate(worst, start=1):
        worst_rows.append({
            "run": run_name,
            **meta,
            "rank": rank,
            "sample_id": row["sample_id"],
            "rope_norm_mae": row.get("rope_norm_mae"),
            "gt_rope_norm": json.dumps(row["gt_rope_norm"], separators=(",", ":")),
            "pred_rope_norm": json.dumps(row["pred_rope_norm"], separators=(",", ":")),
            "rope_norm_abs_error": json.dumps(row["rope_norm_abs_error"], separators=(",", ":")),
        })
    return summary, per_finger, bins, worst_rows


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row[key]) if isinstance(row.get(key), float) else row.get(key, "") for key in fields})


def hard_clean_deltas(summaries: list[dict]) -> list[dict]:
    clean = {(r["dataset"], r["backend"]): r for r in summaries if r["split"] == "clean"}
    rows = []
    for row in summaries:
        base = clean.get((row["dataset"], row["backend"]))
        if row["split"] != "hard" or base is None:
            continue
        rows.append({
            "run": row["run"],
            "dataset": row["dataset"],
            "hard_kind": row["hard_kind"],
            "backend": row["backend"],
            "hard_rope_norm_mae": row["rope_norm_mae"],
            "clean_rope_norm_mae": base["rope_norm_mae"],
            "delta_rope_norm_mae": row["rope_norm_mae"] - base["rope_norm_mae"],
        })
    return sorted(rows, key=lambda r: r["delta_rope_norm_mae"], reverse=True)


def load_runs(scores_root: Path) -> dict[str, list[dict]]:
    return {
        run_dir.name: list(read_jsonl(run_dir / "rope_errors.jsonl"))
        for run_dir in sorted(scores_root.iterdir())
        if (run_dir / "rope_errors.jsonl").exists()
    }


def plot_analysis(output_dir: Path, runs: dict[str, list[dict]], summaries: list[dict], deltas: list[dict], top_k: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if deltas:
        labels = [r["run"] for r in deltas]
        values = [r["delta_rope_norm_mae"] for r in deltas]
        plt.figure(figsize=(max(8, len(labels) * 0.6), 4))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("hard - clean rope_norm_mae")
        plt.tight_layout()
        plt.savefig(fig_dir / "hard_clean_delta.png", dpi=160)
        plt.close()

    for run_name, rows in runs.items():
        gt, pred, err = [], [], []
        for row in rows:
            for _, g, p, e in iter_finger_values(row):
                gt.append(g)
                pred.append(p)
                err.append(e)
        if gt:
            plt.figure(figsize=(4.5, 4.5))
            plt.scatter(gt, pred, c=err, s=4, alpha=0.35, cmap="viridis")
            plt.plot([0, 1], [0, 1], color="black", linewidth=1)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("GT rope_norm")
            plt.ylabel("Pred rope_norm")
            plt.title(run_name)
            plt.colorbar(label="abs error")
            plt.tight_layout()
            plt.savefig(fig_dir / f"scatter_{run_name}.png", dpi=160)
            plt.close()

        worst = sorted(rows, key=lambda row: row.get("rope_norm_mae") or -1.0, reverse=True)[:top_k]
        if worst:
            labels = [row["sample_id"] for row in worst]
            values = [row.get("rope_norm_mae") or 0.0 for row in worst]
            plt.figure(figsize=(max(6, len(labels) * 0.5), 3.5))
            plt.bar(range(len(labels)), values)
            plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
            plt.ylabel("sample rope_norm_mae")
            plt.title(f"worst {run_name}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"worst_{run_name}.png", dpi=160)
            plt.close()


def write_analysis(scores_root: Path, output_dir: Path, top_k: int = 20, make_plots: bool = True) -> dict:
    runs = load_runs(scores_root)
    summaries, per_finger, bins, worst = [], [], [], []
    for run_name, rows in runs.items():
        summary, finger_rows, bin_rows, worst_rows = summarize_run(run_name, rows, top_k=top_k)
        summaries.append(summary)
        per_finger.extend(finger_rows)
        bins.extend(bin_rows)
        worst.extend(worst_rows)

    deltas = hard_clean_deltas(summaries)
    write_tsv(output_dir / "run_summary.tsv", summaries, ["run", "split", "dataset", "hard_kind", "backend", "num_samples", "num_valid_fingers", "rope_norm_mae", "rope_norm_bias"])
    write_tsv(output_dir / "per_finger.tsv", per_finger, ["run", "dataset", "hard_kind", "backend", "finger", "count", "mae", "bias", "gt_mean", "pred_mean"])
    write_tsv(output_dir / "gt_bin_summary.tsv", bins, ["run", "dataset", "hard_kind", "backend", "gt_bin", "count", "mae", "bias"])
    write_tsv(output_dir / "hard_clean_delta.tsv", deltas, ["run", "dataset", "hard_kind", "backend", "hard_rope_norm_mae", "clean_rope_norm_mae", "delta_rope_norm_mae"])
    write_tsv(output_dir / "worst_cases.tsv", worst, ["run", "dataset", "hard_kind", "backend", "rank", "sample_id", "rope_norm_mae", "gt_rope_norm", "pred_rope_norm", "rope_norm_abs_error"])
    if make_plots:
        plot_analysis(output_dir, runs, summaries, deltas, min(top_k, 12))
    return {"num_runs": len(runs), "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze rope diagnostic error rows for report-ready tables and plots.")
    parser.add_argument("scores_root", type=Path, help="Directory containing per-run score folders with rope_errors.jsonl.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_analysis(args.scores_root, args.output_dir, top_k=args.top_k, make_plots=not args.no_plots)
    print(f"Analyzed {result['num_runs']} runs: {result['output_dir']}")


if __name__ == "__main__":
    main()
