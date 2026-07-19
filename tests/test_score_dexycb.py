import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "score_dexycb.py"
    spec = importlib.util.spec_from_file_location("score_dexycb", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScoreDexYcbTest(unittest.TestCase):
    def test_procrustes_removes_similarity_transform(self):
        script = load_script()
        target = np.random.default_rng(4).normal(size=(21, 3))
        predicted = target * 2.3 + [3.0, -2.0, 4.0]
        np.testing.assert_allclose(script.procrustes(predicted, target), target, atol=1e-6)
        np.testing.assert_allclose(
            script.procrustes_batch(predicted[None], target[None])[0], target, atol=1e-6
        )

    def test_orientation_error_is_geodesic(self):
        script = load_script()
        predicted = np.asarray([[0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.0]])
        target = np.zeros((2, 3))
        np.testing.assert_allclose(script.orientation_error_deg(predicted, target), [90.0, 0.0], atol=1e-6)

    def test_bootstrap_is_episode_level_deterministic(self):
        script = load_script()
        candidate = np.asarray([1.0, 1.0, 3.0, 3.0])
        reference = np.zeros(4)
        episodes = ["a", "a", "b", "b"]
        first = script.bootstrap_delta(candidate, reference, episodes, iterations=50, seed=7)
        self.assertEqual(first, script.bootstrap_delta(candidate, reference, episodes, iterations=50, seed=7))
        self.assertLessEqual(first[0], 2.0)
        self.assertGreaterEqual(first[1], 2.0)

    def test_visibility_buckets_use_frozen_thresholds(self):
        script = load_script()
        rows = [{"hand_segmentation_pixels": value} for value in (10, 20, 30)]
        self.assertEqual(script.visibility_labels(rows, (10, 30)), ["low_visible", "mid_visible", "high_visible"])


if __name__ == "__main__":
    unittest.main()
