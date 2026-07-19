#!/usr/bin/env python3
"""Create raw-tree signatures and verify the completed DexYCB experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXCLUDED_RAW_DIRECTORIES = {"img_feats", "img_feats_dino", "videos_v4"}


def raw_signature(root: Path, output: Path) -> Path:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    stack = [root.resolve()]
    while stack:
        directory = stack.pop()
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        for entry in entries:
            relative = Path(entry.path).relative_to(root).as_posix()
            if entry.is_dir(follow_symlinks=False):
                if entry.name in EXCLUDED_RAW_DIRECTORIES:
                    continue
                stack.append(Path(entry.path))
                continue
            stat = entry.stat(follow_symlinks=False)
            record = f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            digest.update(record)
            file_count += 1
            total_bytes += stat.st_size
    payload = {
        "root": str(root.resolve()),
        "signature": "sha256 over sorted-per-directory relative_path, size, mtime_ns",
        "excluded_derived_directories": sorted(EXCLUDED_RAW_DIRECTORIES),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return output


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(args) -> Path:
    split = load(args.processed_root / "split_report.json")
    before, after = load(args.raw_before), load(args.raw_after)
    pretrain_gate, final_gate = load(args.pretrain_gate), load(args.final_gate)
    freeze = load(args.freeze)
    val, test = load(args.val_scores), load(args.test_scores)
    jobs = load(args.jobs)
    checks = {
        "official_counts": {name: split["splits"][name]["sample_count"] for name in ("train", "val", "test")} == {
            "train": 407088, "val": 58592, "test": 116288,
        },
        "subject_intersections_zero": not any(split["subject_intersections"].values()),
        "sample_intersections_zero": not any(split["sample_id_intersections"].values()),
        "episode_cross_split_zero": split["episode_cross_split_count"] == 0,
        "pretrain_coordinate_gate": pretrain_gate["status"] == "validated" and pretrain_gate["coverage"]["subject_count"] == 8,
        "final_coordinate_gate": final_gate["status"] == "validated" and final_gate["coverage"]["subject_count"] == 10,
        "recipe_frozen": freeze["status"] == "frozen" and freeze["test_score_reads_before_freeze"] == 0,
        "val_has_three_methods": {"wilor_base", "rgb_only", "rgb_rope"} <= set(val["metrics"]),
        "test_has_three_methods": {"wilor_base", "rgb_only", "rgb_rope"} <= set(test["metrics"]),
        "test_one_shot_completed": load(args.test_scores.parent / "test_score_access.json")["status"] == "completed",
        "raw_unchanged": all(before[key] == after[key] for key in ("root", "file_count", "total_bytes", "sha256")),
        "all_slurm_jobs_completed": bool(jobs["jobs"]) and all(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in jobs["jobs"]),
    }
    required = [
        args.processed_root / "train27k" / "training.jsonl",
        args.processed_root / "train27k" / "training_xyz.json",
        args.processed_root / "train27k" / "training_mano.npz",
        args.processed_root / "val" / "evaluation.jsonl",
        args.processed_root / "test" / "evaluation.jsonl",
        Path(freeze["checkpoint_paths"]["rgb_only"]),
        Path(freeze["checkpoint_paths"]["rgb_rope"]),
    ]
    checks["required_artifacts_exist"] = all(path.is_file() for path in required)
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw_signature_before": before,
        "raw_signature_after": after,
        "slurm_jobs": jobs,
        "val_scores": str(args.val_scores),
        "test_scores": str(args.test_scores),
        "recipe_freeze": str(args.freeze),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)
    return args.output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    signature = sub.add_parser("raw-signature")
    signature.add_argument("--root", type=Path, required=True)
    signature.add_argument("--output", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    for name in ("processed_root", "raw_before", "raw_after", "pretrain_gate", "final_gate", "freeze", "val_scores", "test_scores", "jobs", "output"):
        verify_parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.command == "raw-signature":
        raw_signature(args.root, args.output)
    else:
        verify(args)


if __name__ == "__main__":
    main()
