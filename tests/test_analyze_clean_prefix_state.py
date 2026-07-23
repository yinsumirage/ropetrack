import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "legacy" / "temporal" / "analyze_clean_prefix_state.py"
    spec = importlib.util.spec_from_file_location("analyze_clean_prefix_state", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CleanPrefixStateAnalysisTest(unittest.TestCase):
    def test_perfect_prefix_motion_is_bounded_and_changes_only_masked_rows(self):
        script = load_script()
        gt = np.zeros((120, 21, 3), dtype=np.float64)
        gt[:, 1, 0] = np.arange(120) * 0.01
        baseline = np.full_like(gt, -1.0)

        predictions = script.perfect_prefix_predictions(gt, (np.arange(120),), baseline)

        np.testing.assert_array_equal(predictions["perfect_prefix_last_clean"][:30], baseline[:30])
        np.testing.assert_array_equal(predictions["perfect_prefix_last_clean"][90:], baseline[90:])
        self.assertAlmostEqual(predictions["perfect_prefix_last_clean"][30, 1, 0], 0.29)
        self.assertAlmostEqual(predictions["perfect_prefix_constant_velocity"][30, 1, 0], 0.30)
        self.assertAlmostEqual(predictions["perfect_prefix_constant_velocity"][89, 1, 0], 0.34)
        self.assertAlmostEqual(predictions["perfect_prefix_damped_velocity"][89, 1, 0], 0.34, places=5)

        visual = gt.copy()
        visual[:, 1, 0] += 1.0
        visual_predictions = script.prefix_predictions(visual, (np.arange(120),), baseline, "visual_prefix")
        self.assertAlmostEqual(visual_predictions["visual_prefix_last_clean"][30, 1, 0], 1.29)


if __name__ == "__main__":
    unittest.main()
