import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_ho3d_v2_wilor.py"
    spec = importlib.util.spec_from_file_location("bench_ho3d_v2_wilor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Ho3dWilorBenchTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
