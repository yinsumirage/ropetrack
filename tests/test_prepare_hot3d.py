import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "datasets" / "prepare_hot3d.py"
    spec = importlib.util.spec_from_file_location("prepare_hot3d", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Hot3dProtocolTest(unittest.TestCase):
    def test_bbox_clamps_to_raw_aria_image(self):
        script = load_script()
        np.testing.assert_allclose(script.clamp_bbox([1177, 739, 1504, 967], (1408, 1408)), [1177, 739, 1407, 967])

    def test_openpose_order_and_skeleton_are_complete(self):
        script = load_script()
        self.assertEqual(sorted(script.OPENPOSE_ORDER.tolist()), list(range(21)))
        self.assertEqual(len(script.SKELETON), 20)
        self.assertIn("mask_hand_pose_available.csv", script.REQUIRED_MASKS)

    def test_selection_filters_sequence_and_rejects_duplicates(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.jsonl"
            rows = [
                {"sequence": "seq0", "timestamp_ns": 1, "hand_index": 0},
                {"sequence": "seq1", "timestamp_ns": 2, "hand_index": 1},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            self.assertEqual(list(script.load_selection(path, "seq0")), [(1, 0)])
            path.write_text("\n".join(json.dumps(rows[0]) for _ in range(2)) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate HOT3D selection"):
                script.load_selection(path, "seq0")


if __name__ == "__main__":
    unittest.main()
