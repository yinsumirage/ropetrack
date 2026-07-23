from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "legacy" / "temporal" / "probe_hot3d_natural_gate.py"
    spec = importlib.util.spec_from_file_location("probe_hot3d_natural_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProbeHot3DNaturalGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()

    def test_participant_and_gate_metrics(self):
        self.assertEqual(self.script.participant("P0014_seq/right/1"), "P0014")
        labels = np.asarray([0, 0, 1, 1], dtype=bool)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        report = self.script.gate_metrics(labels, scores, np.arange(4))
        self.assertEqual(report["auc"], 1.0)
        self.assertEqual(report["balanced_accuracy"], 1.0)

    def test_causal_state_updates_only_when_not_frozen(self):
        pose = np.arange(4 * 45, dtype=np.float32).reshape(4, 45)
        state = self.script.causal_episode_state(
            pose, (np.arange(4, dtype=np.int64),), np.asarray([False, True, True, False])
        )
        np.testing.assert_array_equal(state[0], pose[0])
        np.testing.assert_array_equal(state[1], pose[0])
        np.testing.assert_array_equal(state[2], pose[0])
        np.testing.assert_array_equal(state[3], pose[3])


if __name__ == "__main__":
    unittest.main()
