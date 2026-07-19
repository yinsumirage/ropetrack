import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "validate_dexycb_coordinates.py"
    spec = importlib.util.spec_from_file_location("validate_dexycb_coordinates", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ValidateDexYcbCoordinatesTest(unittest.TestCase):
    def test_projection_matches_opencv_intrinsics(self):
        script = load_script()
        joints = np.asarray([[[1.0, 2.0, 4.0], [0.0, 0.0, 2.0]]], dtype=np.float32)
        k = np.asarray([[[100.0, 0.0, 10.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]]])
        np.testing.assert_allclose(script.project(joints, k), [[[35.0, 120.0], [10.0, 20.0]]])

    def test_gate_selection_covers_depth_and_visible_area_extremes(self):
        script = load_script()
        rows = []
        for index in range(5):
            rows.append({
                "sample_id": f"s/q/c/{index}", "subject_id": "s", "camera_serial": "c",
                "root_depth_m": 0.2 + index, "hand_segmentation_pixels": 100 - 10 * index,
            })
        selected = script.select_gate_rows(rows)
        self.assertEqual({row["sample_id"] for row in selected}, {"s/q/c/0", "s/q/c/4"})

    def test_fixed_rotation_diagnostic_detects_identity(self):
        script = load_script()
        points = np.random.default_rng(3).normal(size=(2, 21, 3)).astype(np.float32) / 100.0
        report = script.fixed_rotation_diagnostic(points, points.copy())
        self.assertAlmostEqual(report["best_fixed_rotation_angle_deg"], 0.0, places=5)
        self.assertLess(report["residual_after_fixed_rotation_mm"]["max"], 1e-4)


if __name__ == "__main__":
    unittest.main()
