import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_hot3d.py"
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


if __name__ == "__main__":
    unittest.main()
