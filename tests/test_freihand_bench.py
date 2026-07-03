import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_freihand.py"
    spec = importlib.util.spec_from_file_location("bench_freihand", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FreiHandBenchTest(unittest.TestCase):
    def test_iter_eval_samples_builds_gt_bbox_from_projected_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation" / "rgb").mkdir(parents=True)
            (root / "evaluation" / "rgb" / "00000000.jpg").write_text("")
            (root / "evaluation_K.json").write_text(json.dumps([
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
            ]))
            (root / "evaluation_verts.json").write_text(json.dumps([
                [[0.1, 0.2, 1.0], [0.4, 0.5, 1.0]]
            ]))

            bench = load_script()
            samples = list(bench.iter_freihand_eval_samples(root, limit=None))

        self.assertEqual([s.sample_id for s in samples], ["00000000"])
        self.assertTrue(samples[0].image_path.as_posix().endswith("evaluation/rgb/00000000.jpg"))
        self.assertEqual(samples[0].bbox_xyxy.tolist(), [10.0, 20.0, 40.0, 50.0])

    def test_to_camera_adds_translation_without_opengl_flip(self):
        bench = load_script()

        pts = bench.to_camera(
            points=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            cam_t=[10.0, 20.0, 30.0],
            units="m",
        )

        self.assertEqual(pts.tolist(), [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])

    def test_freihand_joints_from_vertices_reorders_mano_and_uses_freihand_tips(self):
        bench = load_script()
        verts = [[float(i), float(i + 1), float(i + 2)] for i in range(778)]
        regressor = [[0.0] * 778 for _ in range(16)]
        for joint_id in range(16):
            regressor[joint_id][joint_id] = 1.0

        joints = bench.freihand_joints_from_vertices(verts, regressor)

        self.assertEqual(joints.shape, (21, 3))
        self.assertEqual(joints[0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(joints[1].tolist(), [13.0, 14.0, 15.0])
        self.assertEqual(joints[4].tolist(), [744.0, 745.0, 746.0])
        self.assertEqual(joints[8].tolist(), [320.0, 321.0, 322.0])
        self.assertEqual(joints[12].tolist(), [443.0, 444.0, 445.0])
        self.assertEqual(joints[16].tolist(), [555.0, 556.0, 557.0])
        self.assertEqual(joints[20].tolist(), [672.0, 673.0, 674.0])

    def test_write_eval_gt_subset_limits_gt_files_to_prediction_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            out = Path(tmp) / "out"
            root.mkdir()
            (root / "evaluation_xyz.json").write_text(json.dumps([1, 2, 3]))
            (root / "evaluation_verts.json").write_text(json.dumps([4, 5, 6]))

            bench = load_script()
            bench.write_eval_gt_subset(root, out, 2)

            self.assertEqual(json.loads((out / "evaluation_xyz.json").read_text()), [1, 2])
            self.assertEqual(json.loads((out / "evaluation_verts.json").read_text()), [4, 5])

    def test_run_export_uses_outer_hand_predictor(self):
        bench = load_script()

        source = Path(bench.__file__).read_text()

        self.assertIn("from ropetrack.backends.hand_predictor import HandPredictor", source)
        self.assertNotIn('repo / "third_party" / "anyhand"', source)
        self.assertNotIn("from scripts.rgb_predictor import AnyHandPredictor", source)


if __name__ == "__main__":
    unittest.main()
