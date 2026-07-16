import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ropetrack.datasets.hand_pose import BBoxItem, Ho3dSample
from ropetrack.eval.pipeline import (
    BatchHandPrediction,
    focal_length_for_sample,
    format_prediction,
    select_sample_predictions,
)


class EvalPipelineTest(unittest.TestCase):
    def test_select_sample_predictions_zero_fills_missing_and_picks_best_score(self):
        samples = [
            Ho3dSample("A/0000", Path("a.png"), Path("a.pkl")),
            Ho3dSample("B/0001", Path("b.png"), Path("b.pkl")),
        ]
        low = BatchHandPrediction(
            candidate=BBoxItem(0, 0, samples[0], np.asarray([0, 0, 1, 1], dtype=np.float32), True, 0.1, "detector"),
            vertices=np.ones((778, 3), dtype=np.float32),
            keypoints_3d=np.ones((21, 3), dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
        )
        high = BatchHandPrediction(
            candidate=BBoxItem(0, 1, samples[0], np.asarray([0, 0, 2, 2], dtype=np.float32), True, 0.9, "detector"),
            vertices=np.full((778, 3), 2.0, dtype=np.float32),
            keypoints_3d=np.full((21, 3), 2.0, dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
        )

        selected, failures = select_sample_predictions(samples, [low, high])

        self.assertIs(selected[0], high)
        self.assertIsNone(selected[1])
        self.assertEqual(failures, [{"idx": 1, "sample_id": "B/0001", "error": "RuntimeError('no hand detected')"}])

    def test_freihand_model_keypoints_policy_matches_hamer_export_shape(self):
        sample = Ho3dSample("A/0000", Path("a.png"), Path("a.pkl"))
        hand = BatchHandPrediction(
            candidate=BBoxItem(0, 0, sample, np.zeros(4, dtype=np.float32), True, 1.0, "gt_bbox"),
            vertices=np.zeros((778, 3), dtype=np.float32),
            keypoints_3d=np.asarray([[float(i), 0.0, 0.0] for i in range(21)], dtype=np.float32),
            cam_t=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
        )

        xyz, verts = format_prediction("freihand", hand, np.zeros((16, 778), dtype=np.float32), "model_keypoints", "m")

        self.assertEqual(xyz[4].tolist(), [14.0, 0.0, 0.0])
        self.assertEqual(verts.shape, (778, 3))

    def test_ho3d_mano_vertices_policy_uses_ho3d_camera_and_tips(self):
        sample = Ho3dSample("A/0000", Path("a.png"), Path("a.pkl"))
        verts = np.asarray([[float(i), float(i + 1), float(i + 2)] for i in range(778)], dtype=np.float32)
        regressor = np.zeros((16, 778), dtype=np.float32)
        regressor[0, 0] = 1.0
        hand = BatchHandPrediction(
            candidate=BBoxItem(0, 0, sample, np.zeros(4, dtype=np.float32), True, 1.0, "gt_bbox"),
            vertices=verts,
            keypoints_3d=np.zeros((21, 3), dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
        )

        xyz, _verts = format_prediction("ho3d", hand, regressor, "mano_vertices", "m")

        self.assertEqual(xyz[0].tolist(), [0.0, -1.0, -2.0])
        self.assertEqual(xyz[16].tolist(), [744.0, -745.0, -746.0])

    def test_eval_script_does_not_import_old_bench_modules(self):
        path = Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
        spec = importlib.util.spec_from_file_location("rope_eval", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        source = path.read_text()

        self.assertNotIn("bench_ho3d", source)
        self.assertNotIn("bench_freihand", source)

        pipeline_source = (Path(__file__).resolve().parents[1] / "ropetrack" / "eval" / "pipeline.py").read_text()
        self.assertIn("score_predictions.py", pipeline_source)

    def test_non_eval_split_skips_evaluation_protocol_check(self):
        pipeline_source = (Path(__file__).resolve().parents[1] / "ropetrack" / "eval" / "pipeline.py").read_text()

        self.assertIn('if split == "evaluation":', pipeline_source)
        self.assertIn("validate_eval_protocol(", pipeline_source)

    def test_old_bench_entrypoints_are_removed(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"

        self.assertFalse((scripts / "bench_ho3d.py").exists())
        self.assertFalse((scripts / "bench_freihand.py").exists())
        self.assertFalse((scripts / "bench_eval.py").exists())
        self.assertFalse((scripts / "eval_parallel.py").exists())


class FocalLengthProtocolTest(unittest.TestCase):
    def test_manifest_intrinsic_overrides_model_default(self):
        sample = SimpleNamespace(intrinsic=np.asarray([[736.6, 0, 960], [0, 736.6, 540], [0, 0, 1]]))
        cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000), MODEL=SimpleNamespace(IMAGE_SIZE=224))

        self.assertAlmostEqual(focal_length_for_sample(sample, cfg, (1080, 1920, 3)), 736.6, places=3)

    def test_legacy_adapter_keeps_existing_scaled_default(self):
        sample = SimpleNamespace()
        cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=5000), MODEL=SimpleNamespace(IMAGE_SIZE=224))

        self.assertAlmostEqual(
            focal_length_for_sample(sample, cfg, (480, 640, 3)),
            5000 / 224 * 640,
        )


if __name__ == "__main__":
    unittest.main()
