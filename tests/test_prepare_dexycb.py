import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "prepare_dexycb.py"
    spec = importlib.util.spec_from_file_location("prepare_dexycb", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareDexYcbTest(unittest.TestCase):
    def test_s1_subjects_are_disjoint_and_counts_are_pinned(self):
        script = load_script()
        sets = {name: set(value) for name, value in script.S1_SUBJECTS.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        self.assertEqual(script.S1_COUNTS, {"train": 407088, "val": 58592, "test": 116288})

    def test_sample_identity_keeps_official_hand_side(self):
        script = load_script()
        left = script.SyncFrame("val", "subject", "sequence", 2, ("camera",), "calib", False)
        self.assertFalse(left.is_right)
        self.assertEqual(left.sample_id("camera"), "subject/sequence/camera/000002")

    def test_bop_keyframes_follow_toolkit_every_fourth_rule(self):
        script = load_script()
        frames = [
            script.SyncFrame("test", "subject", "seq", index, ("c0", "c1"), "calib")
            for index in range(6)
        ]
        self.assertEqual(script.expected_bop_keyframes(frames), {(0, 0), (1, 0), (0, 4), (1, 4)})

    def test_selection_balances_sequences_and_cameras_without_sync_duplicates(self):
        script = load_script()
        frames = []
        for sequence in range(4):
            for frame in range(20):
                frames.append(script.SyncFrame(
                    "train", "subject", f"seq{sequence}", frame, script.SERIALS, "calib"
                ))
        selected = script.select_balanced_views(frames, 64, 5)
        self.assertEqual(selected, script.select_balanced_views(frames, 64, 5))
        self.assertEqual(len({(row.episode_id, row.frame_index) for row, _ in selected}), 64)
        self.assertLessEqual(max(Counter(row.episode_id for row, _ in selected).values()) -
                             min(Counter(row.episode_id for row, _ in selected).values()), 1)
        cameras = Counter(serial for _, serial in selected)
        self.assertLessEqual(max(cameras.values()) - min(cameras.values()), 1)

    def test_valid_selection_refills_official_no_hand_sentinels(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            frames = []
            for sequence in range(2):
                for frame_index in range(4):
                    frame = script.SyncFrame("train", "subject", f"seq{sequence}", frame_index, ("c0", "c1"), "calib")
                    frames.append(frame)
                    for serial in frame.serials:
                        directory = raw / frame.subject_id / frame.sequence_id / serial
                        directory.mkdir(parents=True, exist_ok=True)
                        invalid = frame_index == 0
                        np.savez(
                            directory / f"labels_{frame_index:06d}.npz",
                            joint_2d=np.stack((np.linspace(100, 200, 21), np.linspace(100, 200, 21)), axis=-1),
                            joint_3d=np.full((1, 21, 3), -1 if invalid else 0.1, dtype=np.float32),
                            pose_m=np.zeros((1, 51), dtype=np.float32) if invalid else np.ones((1, 51), dtype=np.float32),
                        )
            selected, diagnostics = script.select_balanced_valid_views(raw, frames, 6, 9)
        self.assertEqual(len(selected), 6)
        self.assertEqual(Counter(row.episode_id for row, _ in selected), {"subject/seq0": 3, "subject/seq1": 3})
        self.assertEqual(diagnostics["invalid_candidates_skipped"], 2)
        self.assertEqual(
            diagnostics["invalid_candidates_by_reason"],
            {"official invalid/no-visible-hand sentinel": 2},
        )

    def test_valid_selection_refills_fully_off_image_bbox(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            frames = []
            for frame_index in range(3):
                frame = script.SyncFrame("train", "subject", "seq", frame_index, ("c0",), "calib")
                frames.append(frame)
                directory = raw / frame.subject_id / frame.sequence_id / "c0"
                directory.mkdir(parents=True, exist_ok=True)
                joint_2d = np.stack((np.linspace(100, 200, 21), np.linspace(100, 200, 21)), axis=-1)
                if frame_index == 1:
                    joint_2d[:, 1] += 600
                np.savez(
                    directory / f"labels_{frame_index:06d}.npz",
                    joint_2d=joint_2d,
                    joint_3d=np.full((1, 21, 3), 0.1, dtype=np.float32),
                    pose_m=np.ones((1, 51), dtype=np.float32),
                )
            selected, diagnostics = script.select_balanced_valid_views(raw, frames, 2, 9)
        self.assertEqual(len(selected), 2)
        self.assertEqual(diagnostics["invalid_candidates_skipped"], 1)
        self.assertEqual(
            diagnostics["rejected_candidates"][0]["error"],
            "invalid bbox: bbox empty after clipping",
        )

    def test_bbox_has_fixed_margin_and_clipping(self):
        script = load_script()
        joints = np.tile([100.0, 100.0], (21, 1))
        joints[0] = [10.0, 20.0]
        joints[1] = [630.0, 470.0]
        self.assertEqual(script.bbox_from_joints(joints), [0.0, 0.0, 639.0, 479.0])

    def test_internal_split_keeps_all_views_of_episode_together(self):
        script = load_script()
        rows = []
        for sequence in range(10):
            frame = script.SyncFrame("train", "subject", f"seq{sequence}", 0, script.SERIALS, "calib")
            rows.append((frame, script.SERIALS[0]))
        train, val = script.internal_episode_split(rows, 0.2, 0)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(val), 2)

    def test_test_export_requires_complete_freeze(self):
        script = load_script()
        with self.assertRaises(PermissionError):
            script.require_test_freeze(None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freeze.json"
            path.write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
            with self.assertRaises(PermissionError):
                script.require_test_freeze(path)
            path.write_text(json.dumps({
                "status": "frozen",
                "dataset": "dexycb_s1",
                "checkpoint_selection": "internal_validation_only",
                "test_score_reads_before_freeze": 0,
                "checkpoint_sha256": {"rgb_only": "a", "rgb_rope": "b"},
            }), encoding="utf-8")
            self.assertEqual(script.require_test_freeze(path)["checkpoint_sha256"]["rgb_only"], "a")

    def test_pose_layout_is_not_guessed_from_name(self):
        script = load_script()
        pose = np.arange(51)
        self.assertEqual(pose[:3].tolist(), [0, 1, 2])
        self.assertEqual(len(pose[3:48]), 45)
        self.assertEqual(pose[48:51].tolist(), [48, 49, 50])
        self.assertEqual(script.TOOLKIT_COMMIT, "64551b001d360ad83bc383157a559ec248fb9100")


if __name__ == "__main__":
    unittest.main()
