import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "ropetrack" / "eval" / "dexycb.py"
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

    def test_palm_orientation_is_side_agnostic_geometry(self):
        script = load_script()
        target = np.zeros((1, 21, 3), dtype=np.float64)
        target[0, 5] = [1.0, 0.0, 0.0]
        target[0, 9] = [0.0, 1.0, 0.0]
        target[0, 17] = [-1.0, 0.0, 0.0]
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        predicted = target @ rotation.T
        np.testing.assert_allclose(script.palm_orientation_error_deg(predicted, target), [90.0], atol=1e-6)

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

    def test_translation_diagnostic_uses_exported_base_cam_t_schema(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "mano_cache.npz",
                sample_id=np.asarray(["b", "a"]),
                base_cam_t=np.asarray([[0.2, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=np.float32),
            )
            pose = np.zeros((2, 51), dtype=np.float32)
            pose[:, 48:51] = [[0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]
            np.savez(root / "target.npz", sample_id=np.asarray(["a", "b"]), pose_m=pose)
            error = script.load_mano_translation_diagnostic(
                root / "mano_cache.npz", ["a", "b"], root / "target.npz"
            )
        np.testing.assert_allclose(error, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
