import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_parallel.py"
    spec = importlib.util.spec_from_file_location("eval_parallel", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ParallelEvalTest(unittest.TestCase):
    def test_measure_distances_matches_ho3d_evalutil_semantics(self):
        eval_parallel = load_script()
        distances = np.array(
            [
                [0.0, 0.01],
                [0.02, 0.03],
            ],
            dtype=np.float64,
        )

        mean, auc, pck, thresholds = eval_parallel.measure_distances(distances, 0.0, 0.05, 100)

        self.assertAlmostEqual(mean, 0.015)
        expected_pck = np.array([(distances <= t).mean(axis=0).mean() for t in thresholds])
        expected_auc = np.trapz(expected_pck, thresholds) / np.trapz(np.ones_like(thresholds), thresholds)
        np.testing.assert_allclose(pck, expected_pck)
        self.assertAlmostEqual(auc, expected_auc)

    def test_evaluate_sample_returns_raw_and_aligned_distances(self):
        eval_parallel = load_script()
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        verts = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        result = eval_parallel.evaluate_sample((xyz, verts, xyz.copy(), verts.copy()))

        np.testing.assert_allclose(result["xyz"], [0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["xyz_pa"], [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(result["mesh"], [0.0, 0.0])
        np.testing.assert_allclose(result["mesh_pa"], [0.0, 0.0], atol=1e-7)
        self.assertEqual(result["f_scores"], [1.0, 1.0])
        self.assertEqual(result["f_scores_aligned"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
