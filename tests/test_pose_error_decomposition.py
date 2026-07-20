import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "analyze_pose_error_decomposition.py"
    spec = importlib.util.spec_from_file_location("analyze_pose_error_decomposition", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PoseErrorDecompositionTest(unittest.TestCase):
    def setUp(self):
        self.script = load_script()
        self.gt = np.random.default_rng(7).normal(size=(8, 3))
        angle = 0.7
        self.rotation = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

    def test_translation_rotation_scale_and_local_oracles(self):
        translation = self.script.decompose_sample(self.gt + [2.0, -1.0, 4.0], self.gt)
        self.assertAlmostEqual(translation["E_T"], 0.0, places=10)

        rotated = self.script.decompose_sample(self.gt @ self.rotation.T, self.gt)
        self.assertAlmostEqual(rotated["E_RT"], 0.0, places=10)

        scaled = self.script.decompose_sample(self.gt * 2.5, self.gt)
        self.assertAlmostEqual(scaled["E_Sim3"], 0.0, places=10)

        cube = np.asarray([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
        deformed = cube.copy()
        deformed[:, 0] *= 1.3
        self.assertGreater(self.script.decompose_sample(deformed, cube)["E_Sim3"], 0.01)

    def test_reflection_is_not_a_legal_rotation(self):
        centered = self.gt - self.gt.mean(axis=0)
        reflected = centered.copy()
        reflected[:, 0] *= -1
        values = self.script.decompose_sample(reflected, centered)
        self.assertGreater(values["E_RT"], 0.01)
        self.assertGreater(values["E_Sim3"], 0.01)
        legacy = self.script.align_w_scale(centered, reflected)
        self.assertLess(np.linalg.norm(legacy - centered, axis=1).mean(), 1e-6)

    def test_missing_joints_and_sample_id_reordering(self):
        predicted = self.gt + [1.0, 2.0, 3.0]
        predicted[1] = np.nan
        valid = np.ones(len(self.gt), dtype=bool)
        valid[2] = False
        values = self.script.decompose_sample(predicted, self.gt, valid)
        self.assertEqual(values["valid_joint_count"], len(self.gt) - 2)
        self.assertAlmostEqual(values["E_T"], 0.0, places=10)

        source_ids = ["b", "a", "c"]
        rows = np.asarray([[2], [1], [3]])
        np.testing.assert_array_equal(
            self.script.align_prediction_rows(["a", "b", "c"], source_ids, rows),
            [[1], [2], [3]],
        )

    def test_too_few_or_degenerate_joints_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "fewer than three common"):
            self.script.decompose_sample(self.gt[:2], self.gt[:2])
        line = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "non-degenerate"):
            self.script.decompose_sample(line, line)


if __name__ == "__main__":
    unittest.main()
