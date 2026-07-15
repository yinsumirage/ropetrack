import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "compose_visibility_shift_scores.py"
    spec = importlib.util.spec_from_file_location("compose_visibility_shift_scores", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ComposeVisibilityShiftScoresTest(unittest.TestCase):
    def test_only_masked_rows_are_replaced(self):
        script = load_script()
        out = script.compose_scores(
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([0.8, 0.9, 1.0]),
            np.asarray([False, True, False]),
        )
        np.testing.assert_array_equal(out, [0.1, 0.9, 0.3])


if __name__ == "__main__":
    unittest.main()
