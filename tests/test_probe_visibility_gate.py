import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "probe_visibility_gate.py"
    spec = importlib.util.spec_from_file_location("probe_visibility_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VisibilityGateProbeTest(unittest.TestCase):
    def test_features_reset_at_sequence_gap(self):
        script = load_script()
        pose = np.zeros((4, 45), dtype=np.float32)
        pose[1] = 1.0
        pose[2] = 100.0
        cache = {
            "sample_id": np.asarray(["A/0000", "A/0001", "B/0010", "B/0011"]),
            "base_hand_pose": pose,
            "base_rope_norm": np.zeros((4, 5), dtype=np.float32),
            "input_rope_norm": np.ones((4, 5), dtype=np.float32),
            "rope_valid": np.ones((4, 5), dtype=bool),
        }
        features = script.build_features(cache, np.zeros((4, 10), dtype=np.float32))
        self.assertEqual(features.shape, (4, len(script.FEATURE_NAMES)))
        self.assertEqual(float(features[2, 3]), 0.0)
        self.assertTrue(np.isfinite(features).all())

    def test_gate_metrics_counts_consecutive_false_updates(self):
        script = load_script()
        labels = np.zeros(120, dtype=bool)
        labels[30:90] = True
        scores = np.ones(120)
        scores[40:43] = 0.0
        metrics = script.gate_metrics(labels, scores, np.arange(120), (np.arange(120),), 0.5)
        self.assertEqual(metrics["episodes_any_false_clean"], 1)
        self.assertEqual(metrics["episodes_three_consecutive_false_clean"], 1)


if __name__ == "__main__":
    unittest.main()
