import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "legacy" / "temporal" / "evaluate_visibility_shift.py"
    spec = importlib.util.spec_from_file_location("evaluate_visibility_shift", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VisibilityShiftTest(unittest.TestCase):
    def test_hard_metrics_count_sequence_runs(self):
        script = load_script()
        ids = ["A/0000", "A/0001", "A/0002", "B/0000", "B/0001"]
        scores = np.asarray([0.1, 0.2, 0.3, 0.9, 0.1])
        metrics = script.hard_metrics(ids, scores, 0.5)
        self.assertEqual(metrics["sequences_any_false_clean"], 2)
        self.assertEqual(metrics["sequences_three_consecutive_false_clean"], 1)
        self.assertEqual(metrics["max_consecutive_false_clean"], 3)


if __name__ == "__main__":
    unittest.main()
