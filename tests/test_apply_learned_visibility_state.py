import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "apply_learned_visibility_state.py"
    spec = importlib.util.spec_from_file_location("apply_learned_visibility_state", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LearnedVisibilityStateTest(unittest.TestCase):
    def test_causal_state_updates_only_on_predicted_clean(self):
        script = load_script()
        pose = np.zeros((6, 45), dtype=np.float32)
        pose[:, 0] = np.arange(6)
        ids = np.asarray(["A/0000", "A/0001", "A/0002", "B/0000", "B/0001", "B/0002"])
        freeze = np.asarray([False, True, True, True, False, True])
        state = script.causal_trusted_pose(pose, ids, freeze)
        np.testing.assert_array_equal(state[:, 0], [0, 0, 0, 3, 4, 4])

    def test_shuffle_changes_only_selected_rows_inside_sequence(self):
        script = load_script()
        cache = {
            "sample_id": np.asarray([f"A/{i:04d}" for i in range(4)]),
            "base_global_orient": np.zeros((4, 3), dtype=np.float32),
            "base_hand_pose": np.zeros((4, 45), dtype=np.float32),
            "base_betas": np.zeros((4, 10), dtype=np.float32),
            "base_rope_norm": np.zeros((4, 5), dtype=np.float32),
            "input_rope_norm": np.arange(20, dtype=np.float32).reshape(4, 5),
            "gt_rope_norm": np.zeros((4, 5), dtype=np.float32),
            "rope_valid": np.ones((4, 5), dtype=bool),
        }
        selected = np.asarray([False, True, True, True])
        out = script.shuffle_selected_rope(cache, selected, seed=1)
        np.testing.assert_array_equal(out["input_rope_norm"][0], cache["input_rope_norm"][0])
        self.assertEqual(set(out["input_rope_norm"][1:, 0]), set(cache["input_rope_norm"][1:, 0]))


if __name__ == "__main__":
    unittest.main()
