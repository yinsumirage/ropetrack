import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_ho3d_v2.py"
    spec = importlib.util.spec_from_file_location("bench_ho3d_v2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Ho3dV2BenchTest(unittest.TestCase):
    def test_iter_samples_prefers_evaluation_txt_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ["evaluation/B/rgb/0001.png", "evaluation/A/rgb/0000.png"]:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("")
            (root / "evaluation.txt").write_text("B/0001\nA/0000\n")

            bench = load_script()
            samples = list(bench.iter_ho3d_samples(root, limit=None))

        self.assertEqual([s.sample_id for s in samples], ["B/0001", "A/0000"])
        self.assertEqual(samples[0].image_path.as_posix().endswith("evaluation/B/rgb/0001.png"), True)

    def test_to_opengl_adds_camera_translation_and_flips_yz(self):
        bench = load_script()

        pts = bench.to_opengl_camera(
            points=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            cam_t=[10.0, 20.0, 30.0],
            units="m",
        )

        self.assertEqual(pts.tolist(), [[11.0, -22.0, -33.0], [14.0, -25.0, -36.0]])

    def test_select_hand_uses_highest_score(self):
        bench = load_script()

        hands = [
            type("Hand", (), {"score": 0.1})(),
            type("Hand", (), {"score": 0.9})(),
            type("Hand", (), {"score": 0.2})(),
        ]

        self.assertIs(bench.select_hand(hands), hands[1])

    def test_hand_bbox_from_meta_is_xyxy(self):
        bench = load_script()

        bbox = bench.hand_bbox_from_meta({"handBoundingBox": [1, 2, 3, 4]})

        self.assertEqual(bbox.tolist(), [[1.0, 2.0, 3.0, 4.0]])

    def test_infers_order_from_gt_root_and_meta_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sample_id, root_xyz in [("B/0001", [1, 2, 3]), ("A/0000", [4, 5, 6])]:
                seq, frame = sample_id.split("/")
                (root / "evaluation" / seq / "rgb").mkdir(parents=True, exist_ok=True)
                (root / "evaluation" / seq / "meta").mkdir(parents=True, exist_ok=True)
                (root / "evaluation" / seq / "rgb" / f"{frame}.png").write_text("")
                with (root / "evaluation" / seq / "meta" / f"{frame}.pkl").open("wb") as f:
                    pickle.dump({"handJoints3D": root_xyz}, f)
            (root / "evaluation_xyz.json").write_text(json.dumps([
                [[1, 2, 3]],
                [[4, 5, 6]],
            ]))

            bench = load_script()
            samples = list(bench.iter_ho3d_samples(root, limit=None))

        self.assertEqual([s.sample_id for s in samples], ["B/0001", "A/0000"])

    def test_pushd_restores_cwd(self):
        bench = load_script()
        before = os.getcwd()

        with tempfile.TemporaryDirectory() as tmp:
            with bench.pushd(Path(tmp)):
                self.assertEqual(os.getcwd(), str(Path(tmp)))

        self.assertEqual(os.getcwd(), before)

    def test_iter_samples_accepts_jpg_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation" / "SM1" / "rgb").mkdir(parents=True, exist_ok=True)
            (root / "evaluation" / "SM1" / "meta").mkdir(parents=True, exist_ok=True)
            (root / "evaluation" / "SM1" / "rgb" / "0000.jpg").write_text("")
            (root / "evaluation.txt").write_text("SM1/0000\n")

            bench = load_script()
            sample = next(iter(bench.iter_ho3d_samples(root, limit=None)))

        self.assertTrue(sample.image_path.as_posix().endswith("SM1/rgb/0000.jpg"))

    def test_ho3d_joints_from_vertices_uses_mano_order_and_tips(self):
        bench = load_script()
        verts = [[float(i), float(i + 1), float(i + 2)] for i in range(778)]
        regressor = [[0.0] * 778 for _ in range(16)]
        for joint_id in range(16):
            regressor[joint_id][joint_id] = 1.0

        joints = bench.ho3d_joints_from_vertices(verts, regressor)

        self.assertEqual(joints.shape, (21, 3))
        self.assertEqual(joints[0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(joints[15].tolist(), [15.0, 16.0, 17.0])
        self.assertEqual(joints[16].tolist(), [744.0, 745.0, 746.0])
        self.assertEqual(joints[17].tolist(), [333.0, 334.0, 335.0])
        self.assertEqual(joints[20].tolist(), [672.0, 673.0, 674.0])

    def test_optional_path_str_keeps_none_unset(self):
        bench = load_script()

        self.assertIsNone(bench.optional_path_str(None))
        self.assertEqual(bench.optional_path_str(Path("model.ckpt")), "model.ckpt")

    def test_parse_args_accepts_hamer_backend_and_checkpoint(self):
        bench = load_script()
        old_argv = sys.argv
        sys.argv = [
            "bench",
            "--out-dir",
            "out",
            "--backend",
            "hamer",
            "--hamer-ckpt",
            "hamer.ckpt",
        ]
        try:
            args = bench.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.backend, "hamer")
        self.assertEqual(args.hamer_ckpt, Path("hamer.ckpt"))

    def test_predictor_kwargs_passes_backend_specific_checkpoints(self):
        bench = load_script()
        args = type("Args", (), {
            "backend": "hamer",
            "device": "cuda",
            "batch_size": 2,
            "wilor_ckpt": Path("wilor.ckpt"),
            "wilor_cfg": Path("wilor.yaml"),
            "hamer_ckpt": Path("hamer.ckpt"),
        })()

        kwargs = bench.predictor_kwargs(args)

        self.assertEqual(kwargs["backend"], "hamer")
        self.assertEqual(kwargs["hamer_ckpt"], "hamer.ckpt")
        self.assertEqual(kwargs["wilor_ckpt"], "wilor.ckpt")
        self.assertEqual(kwargs["wilor_cfg"], "wilor.yaml")

    def test_run_backend_with_bbox_dispatches_to_hamer(self):
        bench = load_script()
        calls = []

        class Predictor:
            def _run_wilor(self, *args):
                calls.append("wilor")
                return []

            def _run_hamer(self, *args):
                calls.append("hamer")
                return []

        bench.run_backend_with_bbox(Predictor(), "hamer", "img", "boxes", "is_right", "scores")

        self.assertEqual(calls, ["hamer"])


if __name__ == "__main__":
    unittest.main()
