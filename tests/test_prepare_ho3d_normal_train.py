import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "datasets" / "prepare_ho3d_normal_train.py"
    spec = importlib.util.spec_from_file_location("prepare_ho3d_normal_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareHo3dNormalTrainTest(unittest.TestCase):
    def test_selection_is_balanced_unique_and_seeded(self):
        script = load_script()
        ids = [f"S{sequence}/{frame:04d}" for sequence in range(3) for frame in range(8)]
        first = script.sequence_balanced(ids, 12, 7)
        second = script.sequence_balanced(ids, 12, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        counts = {name: sum(value.startswith(f"{name}/") for value in first) for name in ("S0", "S1", "S2")}
        self.assertEqual(counts, {"S0": 4, "S1": 4, "S2": 4})

    def test_gt_is_opencv_and_openpose_ordered(self):
        script = load_script()
        joints = np.arange(63, dtype=np.float32).reshape(21, 3)
        transformed = script.gt_openpose_opencv(joints)
        np.testing.assert_array_equal(transformed[0], joints[0] * [1, -1, -1])
        np.testing.assert_array_equal(transformed[1], joints[13] * [1, -1, -1])
        np.testing.assert_array_equal(transformed[5], joints[1] * [1, -1, -1])


if __name__ == "__main__":
    unittest.main()
