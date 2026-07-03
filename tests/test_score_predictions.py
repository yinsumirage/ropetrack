import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "score_predictions.py"
    spec = importlib.util.spec_from_file_location("score_predictions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ParallelEvalTest(unittest.TestCase):
    def test_measure_distances_matches_ho3d_evalutil_semantics(self):
        scorer = load_script()
        distances = np.array(
            [
                [0.0, 0.01],
                [0.02, 0.03],
            ],
            dtype=np.float64,
        )

        mean, auc, pck, thresholds = scorer.measure_distances(distances, 0.0, 0.05, 100)

        self.assertAlmostEqual(mean, 0.015)
        expected_pck = np.array([(distances <= t).mean(axis=0).mean() for t in thresholds])
        expected_auc = np.trapz(expected_pck, thresholds) / np.trapz(np.ones_like(thresholds), thresholds)
        np.testing.assert_allclose(pck, expected_pck)
        self.assertAlmostEqual(auc, expected_auc)

    def test_evaluate_sample_returns_raw_and_aligned_distances(self):
        scorer = load_script()
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

        result = scorer.evaluate_sample((xyz, verts, xyz.copy(), verts.copy()))

        np.testing.assert_allclose(result["xyz"], [0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["xyz_pa"], [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-7)
        np.testing.assert_allclose(result["mesh"], [0.0, 0.0])
        np.testing.assert_allclose(result["mesh_pa"], [0.0, 0.0], atol=1e-7)
        self.assertEqual(result["f_scores"], [1.0, 1.0])
        self.assertEqual(result["f_scores_aligned"], [1.0, 1.0])

    def test_load_inputs_accepts_separate_prediction_and_gt_dirs(self):
        scorer = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            gt_dir = root / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            (gt_dir / "evaluation_xyz.json").write_text(json.dumps([[[0, 0, 0]]]))
            (gt_dir / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]]]))
            (pred_dir / "pred.json").write_text(json.dumps([[[[0, 0, 0]]], [[[0, 0, 0]]]]))

            xyz, verts, pred_xyz, pred_verts = scorer.load_inputs(pred_dir, gt_dir)

        self.assertEqual(xyz, [[[0, 0, 0]]])
        self.assertEqual(verts, [[[0, 0, 0]]])
        self.assertEqual(pred_xyz, [[[0, 0, 0]]])
        self.assertEqual(pred_verts, [[[0, 0, 0]]])


if __name__ == "__main__":
    unittest.main()
