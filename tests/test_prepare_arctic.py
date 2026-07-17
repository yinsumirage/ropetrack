import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_arctic.py"
    spec = importlib.util.spec_from_file_location("prepare_arctic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArcticProtocolTest(unittest.TestCase):
    def test_frame_intrinsic_and_bbox_protocol(self):
        script = load_script()

        self.assertEqual(script.annotation_index(12, 2), 10)
        np.testing.assert_allclose(
            script.scaled_intrinsic([[100, 0, 50], [0, 120, 40], [0, 0, 1]]),
            [[30, 0, 15], [0, 36, 12], [0, 0, 1]],
        )
        np.testing.assert_allclose(
            script.hand_bbox([[100, 100], [200, 150]], margin=0.25),
            [75, 75, 225, 175],
        )

    def test_train_subjects_and_joint_only_verification(self):
        script = load_script()
        self.assertEqual(
            script.SPLIT_SUBJECTS["train"],
            ("s01", "s02", "s04", "s06", "s07", "s08", "s09", "s10"),
        )
        records = [{"manifest": {"sample_id": "s01/seq/right/00010"}, "side": "right"}]
        counts = script.verify_records(records, np.zeros((1, 21, 3), dtype=np.float32), None)
        self.assertEqual(dict(counts), {"right": 1})


if __name__ == "__main__":
    unittest.main()
