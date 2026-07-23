#!/usr/bin/env python3
"""Freeze DexYCB checkpoints and evaluation recipe before any S1 test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(args) -> Path:
    if args.output.exists():
        raise FileExistsError(f"recipe freeze already exists: {args.output}")
    gate = json.loads(args.coordinate_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "validated" or gate.get("mode") != "pretrain":
        raise ValueError("pretrain coordinate gate is not validated")
    val_scores = json.loads(args.val_scores.read_text(encoding="utf-8"))
    if val_scores.get("protocol_split") != "val":
        raise ValueError("checkpoint freeze requires official S1 val scores")
    checkpoints = {"rgb_only": args.rgb_checkpoint, "rgb_rope": args.rope_checkpoint}
    configs = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        train_log = json.loads((path.parent / "train_log.json").read_text(encoding="utf-8"))
        configs[name] = train_log["config"]
    if configs["rgb_only"]["num_train"] != configs["rgb_rope"]["num_train"] or \
            configs["rgb_only"]["num_val"] != configs["rgb_rope"]["num_val"]:
        raise ValueError("matched RGB-only/rope training split sizes differ")
    if configs["rgb_only"]["provenance"]["sample_id_sha256"] != configs["rgb_rope"]["provenance"]["sample_id_sha256"]:
        raise ValueError("matched RGB-only/rope sample identities differ")
    if args.test_root.exists():
        raise PermissionError(f"official test derivative exists before freeze: {args.test_root}")
    payload = {
        "status": "frozen",
        "dataset": "dexycb_s1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_selection": "internal_validation_only",
        "test_score_reads_before_freeze": 0,
        "checkpoint_sha256": {name: sha256(path) for name, path in checkpoints.items()},
        "checkpoint_paths": {name: str(path) for name, path in checkpoints.items()},
        "best_epochs": {name: int(config["best_epoch"]) for name, config in configs.items()},
        "matched_training": {
            "sample_id_sha256": configs["rgb_only"]["provenance"]["sample_id_sha256"],
            "train_sample_id_sha256": configs["rgb_only"]["provenance"]["train_sample_id_sha256"],
            "val_sample_id_sha256": configs["rgb_only"]["provenance"]["val_sample_id_sha256"],
            "num_train": configs["rgb_only"]["num_train"],
            "num_internal_val": configs["rgb_only"]["num_val"],
            "only_difference": "rope_mode zero versus correct",
        },
        "evaluation": {
            "bbox": "same frozen GT joint_2d bbox manifest for all methods",
            "methods": ["wilor_base", "rgb_only", "rgb_rope"],
            "visibility_thresholds": str(args.visibility_thresholds),
            "visibility_thresholds_sha256": sha256(args.visibility_thresholds),
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260720,
            "test_score_policy": "one shot; no epoch, bbox, coordinate, mixture, or method selection from test",
        },
        "external_transfer": {
            "old_direct_pose_folds": "fold0/fold1/fold2 all reported on S1 val as fixed mean and spread; no DexYCB fold selection",
        },
        "coordinate_gate": str(args.coordinate_gate),
        "coordinate_gate_sha256": sha256(args.coordinate_gate),
        "val_scores": str(args.val_scores),
        "val_scores_sha256": sha256(args.val_scores),
        "raw_signature_before": str(args.raw_signature_before),
        "raw_signature_before_sha256": sha256(args.raw_signature_before),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return args.output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinate-gate", type=Path, required=True)
    parser.add_argument("--val-scores", type=Path, required=True)
    parser.add_argument("--rgb-checkpoint", type=Path, required=True)
    parser.add_argument("--rope-checkpoint", type=Path, required=True)
    parser.add_argument("--visibility-thresholds", type=Path, required=True)
    parser.add_argument("--raw-signature-before", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    freeze(parse_args())
