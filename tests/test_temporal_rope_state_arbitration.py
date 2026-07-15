import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "temporal_rope_state_arbitration.py"
    spec = importlib.util.spec_from_file_location("temporal_rope_state_arbitration", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TemporalRopeStateArbitrationTest(unittest.TestCase):
    def test_gate_uses_state_only_on_masked_rows_and_beats_margin(self):
        script = load_script()
        current = np.full((4, 5), 0.5, dtype=np.float32)
        state = np.full((4, 5), 0.4, dtype=np.float32)
        state[2, 0] = 0.49
        gate = script.arbitration_gate(current, state, np.asarray([1, 2]), 0.02)
        self.assertFalse(gate[0].any())
        self.assertTrue(gate[1].all())
        self.assertFalse(gate[2, 0])
        self.assertTrue(gate[2, 1:].all())

    def test_shuffle_stays_inside_each_masked_episode(self):
        script = load_script()
        values = np.arange(240 * 5).reshape(240, 5)
        episodes = (np.arange(120), np.arange(120, 240))
        shuffled = script.shuffled_episode_rows(values, episodes, seed=3)
        np.testing.assert_array_equal(shuffled[:30], values[:30])
        np.testing.assert_array_equal(shuffled[90:150], values[90:150])
        self.assertEqual(set(shuffled[30:90, 0]), set(values[30:90, 0]))
        self.assertEqual(set(shuffled[150:210, 0]), set(values[150:210, 0]))


if __name__ == "__main__":
    unittest.main()
