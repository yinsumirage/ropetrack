#!/usr/bin/env python3
"""Freeze and verify the InterHand2.6M one-view experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ropetrack.datasets.dexycb_artifacts import raw_signature


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_pass(path: Path) -> dict:
    payload = load(path)
    if payload.get("status") != "PASS":
        raise ValueError(f"verification is not PASS: {path}")
    return payload


def repo_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
    ).strip()


def verified_checkpoint(checkpoint: Path, rope_mode: str, verification: Path, kind: str) -> dict:
    train_log = checkpoint.parent / "train_log.json"
    config = load(train_log)["config"]
    provenance = config["provenance"]
    return {
        "kind": kind,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "train_log": str(train_log),
        "train_log_sha256": sha256(train_log),
        "training_git_commit": provenance["git_commit"],
        "training_protocol_sha256": provenance["protocol_sha256"],
        "training_sample_id_sha256": provenance["sample_id_sha256"],
        "checkpoint_selection": f"internal validation PA-MPJPE; best_epoch={config['best_epoch']}",
        "rope_mode": rope_mode,
        "verification": str(verification),
        "verification_sha256": sha256(verification),
        "input_mode": "localized 4x3 WiLoR tokens" + ("; rope disabled" if rope_mode == "zero" else " plus correct normalized five-rope"),
        "apply_command": "python scripts/rope_refiner/direct_pose_head.py apply --dataset interhand26m --cache <CACHE> --mano-cache <MANO_CACHE> --feature-cache <TOKENS> --checkpoint <CHECKPOINT> --rope-mode " + rope_mode + " --out-dir <OUT>",
    }


def freeze_preval(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.normal_fold) != 3:
        raise ValueError(f"expected all three frozen normal folds, got {len(args.normal_fold)}")
    require_pass(args.normal_verification)
    require_pass(args.dexycb_verification)
    methods = {"wilor_base": {
        "kind": "base", "checkpoint": str(args.wilor_checkpoint), "checkpoint_sha256": sha256(args.wilor_checkpoint),
        "model_config": str(args.wilor_config), "model_config_sha256": sha256(args.wilor_config),
        "input_mode": "InterHand RGB with frozen GT-derived per-hand bbox; no rope",
        "apply_command": "python scripts/eval.py --dataset interhand26m_<split>_oneview --method wilor_original --root <ROOT> --out-dir <OUT> --save-mano-cache",
    }}
    for index, checkpoint in enumerate(args.normal_fold):
        methods[f"normal_fold{index}"] = verified_checkpoint(
            checkpoint, "correct", args.normal_verification, "frozen_normal_joint_direct_pose",
        )
    for name, checkpoint, rope_mode in (
        ("dexycb_rgb_only", args.dexycb_rgb_only, "zero"),
        ("dexycb_rgb_rope", args.dexycb_rgb_rope, "correct"),
    ):
        methods[name] = verified_checkpoint(
            checkpoint, rope_mode, args.dexycb_verification, "verified_dexycb_direct_pose",
        )
    payload = {
        "status": "frozen_before_interhand_val",
        "dataset": "interhand26m_v1_30fps",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_git_commit": repo_commit(),
        "selection_uses_interhand_val_or_test": False,
        "methods": methods,
        "apply_contract": "same InterHand manifest IDs, GT-derived per-side bbox, WiLoR base cache, normalized rope bundle, localized 4x3 tokens; every listed fold reported",
        "excluded_unverified_models": "older standalone ARCTIC RGB-only/RGB+rope checkpoints have no independent final-verification PASS and are not admitted",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return args.output


def matched_checkpoint_configs(rgb: Path, rope: Path) -> tuple[dict, dict]:
    configs = []
    for checkpoint in (rgb, rope):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        configs.append(load(checkpoint.parent / "train_log.json")["config"])
    left, right = configs
    comparable = (
        "num_train", "num_val", "token_dim", "hidden_dim", "max_delta", "weights",
    )
    differences = {key: (left.get(key), right.get(key)) for key in comparable if left.get(key) != right.get(key)}
    if left.get("training_recipe") != right.get("training_recipe"):
        differences["training_recipe"] = (left.get("training_recipe"), right.get("training_recipe"))
    if left["provenance"].get("protocol_sha256") != right["provenance"].get("protocol_sha256"):
        differences["protocol_sha256"] = (left["provenance"].get("protocol_sha256"), right["provenance"].get("protocol_sha256"))
    if left["provenance"].get("inputs") != right["provenance"].get("inputs"):
        differences["training_inputs"] = (left["provenance"].get("inputs"), right["provenance"].get("inputs"))
    for key in ("sample_id_sha256", "train_sample_id_sha256", "val_sample_id_sha256"):
        if left["provenance"][key] != right["provenance"][key]:
            differences[key] = (left["provenance"][key], right["provenance"][key])
    if differences or left["rope_mode"] != "zero" or right["rope_mode"] != "correct":
        raise ValueError(f"matched InterHand checkpoints differ beyond rope mode: {differences}")
    return left, right


def interhand_checkpoint(checkpoint: Path, config: dict, rope_mode: str) -> dict:
    train_log = checkpoint.parent / "train_log.json"
    return {
        "kind": "interhand_train27k_direct_pose",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "train_log": str(train_log), "train_log_sha256": sha256(train_log),
        "training_git_commit": config["provenance"]["git_commit"],
        "training_protocol_sha256": config["provenance"]["protocol_sha256"],
        "training_sample_id_sha256": config["provenance"]["sample_id_sha256"],
        "checkpoint_selection": f"internal validation PA-MPJPE; best_epoch={config['best_epoch']}",
        "training_recipe": config["training_recipe"],
        "rope_mode": rope_mode,
        "input_mode": "localized 4x3 WiLoR tokens" + ("; rope disabled" if rope_mode == "zero" else " plus correct normalized five-rope"),
        "apply_command": "python scripts/rope_refiner/direct_pose_head.py apply --dataset interhand26m --cache <CACHE> --mano-cache <MANO_CACHE> --feature-cache <TOKENS> --checkpoint <CHECKPOINT> --rope-mode " + rope_mode + " --out-dir <OUT>",
    }


def freeze_test(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.test_root.exists():
        raise PermissionError(f"test GT derivative exists before freeze: {args.test_root}")
    gate, val, candidate, preval = load(args.coordinate_gate), load(args.val_scores), load(args.test_candidate_protocol), load(args.preval_freeze)
    if gate.get("status") != "validated" or gate.get("mode") != "pretrain":
        raise ValueError("pretrain coordinate gate is not validated")
    if val.get("protocol_split") != "val":
        raise ValueError("official-val score is required before test freeze")
    if candidate.get("test_joint_3d_or_MANO_values_read") != 0:
        raise PermissionError("test candidate index crossed the GT boundary")
    if preval.get("status") != "frozen_before_interhand_val":
        raise ValueError("pre-val model list was not frozen")
    rgb, rope = matched_checkpoint_configs(args.rgb_checkpoint, args.rope_checkpoint)
    methods = dict(preval["methods"])
    methods.update({
        "rgb_only": interhand_checkpoint(args.rgb_checkpoint, rgb, "zero"),
        "rgb_rope": interhand_checkpoint(args.rope_checkpoint, rope, "correct"),
    })
    missing_val = sorted(set(methods) - set(val.get("metrics", {})))
    if missing_val:
        raise ValueError(f"val score missing frozen methods: {missing_val}")
    payload = {
        "status": "frozen",
        "dataset": "interhand26m_v1_30fps",
        "project_protocol": "interhand26m_v1_30fps_oneview_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_git_commit": repo_commit(),
        "checkpoint_selection": "internal_validation_only",
        "test_score_reads_before_freeze": 0,
        "test_candidates": str(args.test_candidate_protocol),
        "test_candidates_sha256": candidate["candidate_manifest_sha256"],
        "test_candidate_sample_id_sha256": candidate["candidate_sample_id_sha256"],
        "test_post_freeze_filter": candidate["post_freeze_rule"],
        "methods": methods,
        "matched_training": {
            "sample_id_sha256": rgb["provenance"]["sample_id_sha256"],
            "train_sample_id_sha256": rgb["provenance"]["train_sample_id_sha256"],
            "val_sample_id_sha256": rgb["provenance"]["val_sample_id_sha256"],
            "num_train": rgb["num_train"],
            "num_internal_val": rgb["num_val"],
            "best_epochs": {"rgb_only": rgb["best_epoch"], "rgb_rope": rope["best_epoch"]},
            "only_difference": "rope_mode zero versus correct",
        },
        "evaluation": {
            "methods": list(methods),
            "same_sample_ids_and_bbox": True,
            "bbox_thresholds": str(args.bucket_thresholds),
            "bbox_thresholds_sha256": sha256(args.bucket_thresholds),
            "bootstrap_unit": "underlying frame_group_id",
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 20260720,
            "score_policy": "one shot; no checkpoint, bbox, axis, sample, mixture, or method selection from test",
        },
        "coordinate_gate": str(args.coordinate_gate),
        "coordinate_gate_sha256": sha256(args.coordinate_gate),
        "preval_model_freeze": str(args.preval_freeze),
        "preval_model_freeze_sha256": sha256(args.preval_freeze),
        "val_scores": str(args.val_scores),
        "val_scores_sha256": sha256(args.val_scores),
        "raw_signature_before": str(args.raw_signature_before),
        "raw_signature_before_sha256": sha256(args.raw_signature_before),
        "rope_boundary": "test rope is GT-derived ideal geometry generated only after this freeze; not no-GT RGB inference or a validated physical sensor",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return args.output


def finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def manifest_ids(path: Path) -> list[str]:
    return [json.loads(line)["sample_id"] for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify(args: argparse.Namespace) -> Path:
    split, before, after = load(args.processed_root / "split_report.json"), load(args.raw_before), load(args.raw_after)
    pretrain, final, preval, freeze = load(args.pretrain_gate), load(args.final_gate), load(args.preval_freeze), load(args.test_freeze)
    val, test, jobs = load(args.val_scores), load(args.test_scores), load(args.jobs)
    expected_methods = set(freeze["methods"])
    train_protocol = load(args.processed_root / "train27k" / "protocol.json")
    val_protocol = load(args.processed_root / "val" / "protocol.json")
    test_protocol = load(args.processed_root / "test" / "protocol.json")
    candidate_ids = set()
    for row in [json.loads(line) for line in (args.processed_root / "test_candidates.jsonl").read_text(encoding="utf-8").splitlines() if line]:
        for side in row["candidate_sides"]:
            candidate_ids.add(f"{row['split']}/Capture{row['capture_id']}/{row['sequence_id']}/cam{row['camera_id']}/{row['frame_id']:06d}/{side}")
    test_ids = manifest_ids(args.processed_root / "test" / "evaluation.jsonl")
    capacity = train_protocol.get("selection", {}).get("capacity_balance", {})
    checks = {
        "version_exact": split["dataset"] == "InterHand2.6M v1.0 30fps",
        "protocol_exact": split["project_protocol"] == "interhand26m_v1_30fps_oneview_v1",
        "train27k_exact": train_protocol["num_samples"] == 27000,
        "train_capture_episode_capacity_balanced": capacity.get("status") == "PASS",
        "train_subjects_exclude_official_val_test_overlap": split["train_subjects_excluded_for_project_subject_disjointness"] == ["0", "1", "12"],
        "all_split_overlaps_zero": split["all_required_overlap_zero"],
        "oneview_frame_rule": all(protocol["distribution"]["frames"] <= protocol["num_samples"] for protocol in (train_protocol, val_protocol, test_protocol)),
        "pretrain_coordinate_gate": pretrain["status"] == "validated" and pretrain["mode"] == "pretrain",
        "final_coordinate_gate": final["status"] == "validated" and final["mode"] == "final",
        "preval_methods_frozen": preval["status"] == "frozen_before_interhand_val",
        "test_recipe_frozen": freeze["status"] == "frozen" and freeze["test_score_reads_before_freeze"] == 0,
        "test_ids_are_frozen_candidate_subset": bool(test_ids) and set(test_ids) <= candidate_ids,
        "val_all_frozen_methods": expected_methods == set(val.get("metrics", {})),
        "test_all_frozen_methods": expected_methods == set(test.get("metrics", {})),
        "test_one_shot_completed": load(args.test_scores.parent / "test_score_access.json")["status"] == "completed",
        "score_json_finite": finite_numbers(val) and finite_numbers(test),
        "raw_unchanged": all(before[key] == after[key] for key in ("root", "file_count", "total_bytes", "sha256")),
        "all_slurm_jobs_completed": bool(jobs.get("jobs")) and all(row["state"] == "COMPLETED" and row["exit_code"] == "0:0" for row in jobs["jobs"]),
        "ideal_rope_boundary_recorded": "GT-derived ideal" in freeze["rope_boundary"],
    }
    required = [
        args.processed_root / "protocol.json",
        args.processed_root / "split_report.json",
        args.processed_root / "train27k" / "training.jsonl",
        args.processed_root / "train27k" / "training_xyz.json",
        args.processed_root / "val" / "evaluation.jsonl",
        args.processed_root / "test" / "evaluation.jsonl",
        args.pretrain_gate, args.final_gate, args.preval_freeze, args.test_freeze,
        args.val_scores, args.test_scores, args.jobs,
    ] + [Path(row["checkpoint"]) for row in freeze["methods"].values()]
    checks["required_artifacts_exist"] = all(path.is_file() for path in required)
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw_signature_before": before,
        "raw_signature_after": after,
        "manifests": {
            "train": train_protocol["sha256"],
            "val": val_protocol["sha256"],
            "test": test_protocol["sha256"],
        },
        "slurm_jobs": jobs,
        "val_scores": str(args.val_scores),
        "test_scores": str(args.test_scores),
        "test_freeze": str(args.test_freeze),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)
    return args.output


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    signature = sub.add_parser("raw-signature")
    signature.add_argument("--root", type=Path, required=True)
    signature.add_argument("--output", type=Path, required=True)
    preval = sub.add_parser("freeze-preval")
    preval.add_argument("--wilor-checkpoint", type=Path, required=True)
    preval.add_argument("--wilor-config", type=Path, required=True)
    preval.add_argument("--normal-fold", type=Path, action="append", required=True)
    preval.add_argument("--normal-verification", type=Path, required=True)
    preval.add_argument("--dexycb-rgb-only", type=Path, required=True)
    preval.add_argument("--dexycb-rgb-rope", type=Path, required=True)
    preval.add_argument("--dexycb-verification", type=Path, required=True)
    preval.add_argument("--output", type=Path, required=True)
    test = sub.add_parser("freeze-test")
    for name in ("coordinate_gate", "val_scores", "rgb_checkpoint", "rope_checkpoint", "bucket_thresholds", "test_candidate_protocol", "raw_signature_before", "preval_freeze", "test_root", "output"):
        test.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    verification = sub.add_parser("verify")
    for name in ("processed_root", "raw_before", "raw_after", "pretrain_gate", "final_gate", "preval_freeze", "test_freeze", "val_scores", "test_scores", "jobs", "output"):
        verification.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "raw-signature":
        return raw_signature(args.root, args.output)
    if args.command == "freeze-preval":
        return freeze_preval(args)
    if args.command == "freeze-test":
        return freeze_test(args)
    return verify(args)


if __name__ == "__main__":
    main()
