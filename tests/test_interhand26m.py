import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ropetrack.datasets.hand_pose import iter_hand_pose_samples, load_gt_bbox_candidates
from ropetrack.eval.protocols import canonical_dataset
from ropetrack.eval.config import build_run_args
from ropetrack.rope import FINGER_CHAINS, canonical_rope_dataset


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InterHand26MTest(unittest.TestCase):
    def test_official_joint_mapping_and_coordinate_formula_are_explicit(self):
        script = load_script("prepare_interhand26m")
        self.assertEqual(script.INTERHAND_TO_OPENPOSE.tolist(), [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16])
        world = np.asarray([[11.0, 22.0, 33.0]])
        camera = script.world_to_camera(world, np.eye(3), np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(camera, [[10.0, 20.0, 30.0]])

    def test_oneview_is_deterministic_and_never_splits_candidate_hands(self):
        script = load_script("prepare_interhand26m")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "annotations" / "val"
            directory.mkdir(parents=True)
            images, annotations = [], []
            for image_id, camera in enumerate(("400001", "400002")):
                images.append({
                    "id": image_id, "file_name": f"Capture0/seq/cam{camera}/image00001.jpg",
                    "width": 334, "height": 512, "capture": 0, "subject": 9,
                    "seq_name": "seq", "camera": camera, "frame_idx": 1,
                })
                annotations.append({
                    "image_id": image_id, "hand_type": "interacting", "hand_type_valid": 1,
                    "joint_valid": [1] * 42,
                })
            (directory / "InterHand2.6M_val_data.json").write_text(json.dumps({"images": images, "annotations": annotations}))
            first, _ = script.select_oneview(root, "val")
            second, _ = script.select_oneview(root, "val")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["candidate_sides"], ["right", "left"])

    def test_group_selection_keeps_left_right_pair_and_exact_count(self):
        script = load_script("prepare_interhand26m")
        records = []
        for frame in range(4):
            sides = ("right", "left") if frame < 2 else ("right",)
            for side in sides:
                records.append({"row": {
                    "frame_group_id": f"train/Capture0/seq/{frame}",
                    "episode_id": "train/Capture0/seq",
                    "sample_id": f"sample/{frame}/{side}",
                }})
        selected = script.select_group_balanced(records, 5, 3)
        grouped = {}
        for record in selected:
            grouped.setdefault(record["row"]["frame_group_id"], 0)
            grouped[record["row"]["frame_group_id"]] += 1
        self.assertEqual(len(selected), 5)
        self.assertNotIn(1, [count for frame, count in grouped.items() if frame.endswith(("/0", "/1"))])

    def test_bbox_is_side_specific_margin_and_clipped(self):
        script = load_script("prepare_interhand26m")
        points = np.tile([100.0, 100.0], (21, 1))
        points[0], points[1], points[2], points[3] = [0, 0], [333, 511], [20, 30], [30, 40]
        self.assertEqual(script.bbox_from_points(points, np.ones(21, dtype=bool), 334, 512), [0.0, 0.0, 333.0, 511.0])

    def test_test_gt_requires_matching_candidate_freeze(self):
        script = load_script("prepare_interhand26m")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            candidate.write_text(json.dumps({"candidate_manifest_sha256": "abc"}))
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({
                "status": "frozen", "dataset": "interhand26m_v1_30fps",
                "test_score_reads_before_freeze": 0, "test_candidates_sha256": "abc",
            }))
            self.assertEqual(script.require_test_freeze(freeze, candidate)["path"], str(freeze))
            freeze.write_text(json.dumps({"status": "frozen"}))
            with self.assertRaises(PermissionError):
                script.require_test_freeze(freeze, candidate)

    def test_manifest_adapter_resolves_raw_image_and_left_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed, raw = Path(tmp) / "processed", Path(tmp) / "raw"
            processed.mkdir(); raw.mkdir()
            (processed / "protocol.json").write_text(json.dumps({"raw_root": str(raw)}))
            row = {
                "sample_id": "val/Capture0/seq/cam1/000001/left",
                "image_path": "images/val/Capture0/seq/cam1/image000001.jpg",
                "bbox_xyxy": [1, 2, 30, 40], "is_right": False,
                "episode_id": "val/Capture0/seq", "frame_index": 1,
                "intrinsic": np.eye(3).tolist(), "joint_valid": [1] * 21,
            }
            (processed / "evaluation.jsonl").write_text(json.dumps(row) + "\n")
            sample = next(iter(iter_hand_pose_samples("interhand26m", processed, None)))
            candidate = load_gt_bbox_candidates("interhand26m", [sample])[0]
        self.assertEqual(sample.image_path, raw / row["image_path"])
        self.assertFalse(candidate.is_right)

    def test_shared_protocol_and_rope_paths_accept_interhand(self):
        self.assertEqual(canonical_dataset("InterHand2.6M"), "interhand26m")
        self.assertEqual(canonical_rope_dataset("interhand26m"), "interhand26m")
        self.assertEqual(FINGER_CHAINS["interhand26m"], FINGER_CHAINS["freihand"])

    def test_eval_configs_use_external_manifest_and_kinematic_joints(self):
        train = build_run_args("interhand26m_train27k", "wilor_original", split="training")
        val = build_run_args("interhand26m_val_oneview", "wilor_original")
        self.assertEqual(train.adapter, "interhand26m")
        self.assertEqual(train.joint_source, "model_keypoints")
        self.assertEqual(train.root.name, "train27k")
        self.assertEqual(val.root.name, "val")

    def test_frame_bootstrap_resamples_paired_hands_together(self):
        scorer = load_script("score_interhand26m")
        candidate = np.asarray([1.0, 1.0, 3.0, 3.0])
        reference = np.zeros(4)
        groups = ["frame0", "frame0", "frame1", "frame1"]
        interval = scorer.bootstrap_delta(candidate, reference, groups, 100, 7)
        self.assertLessEqual(interval[0], 2.0)
        self.assertGreaterEqual(interval[1], 2.0)


if __name__ == "__main__":
    unittest.main()
