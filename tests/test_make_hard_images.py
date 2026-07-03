import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_hard_images.py"
    spec = importlib.util.spec_from_file_location("make_hard_images", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MakeHardImagesTest(unittest.TestCase):
    def test_mask_effect_changes_only_inside_bbox(self):
        hard = load_script()
        image = Image.new("RGB", (10, 10), (255, 0, 0))

        out = hard.apply_hard_effect(image, [2, 2, 8, 8], effect="mask", severity=0.5, seed=1)

        self.assertEqual(out.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(out.getpixel((5, 5)), (0, 0, 0))

    def test_tip_square_masks_given_fingertip_points(self):
        hard = load_script()
        image = Image.new("RGB", (20, 20), (255, 255, 255))

        out = hard.apply_hard_effect(
            image,
            [0, 0, 20, 20],
            effect="tip_square",
            severity=0.2,
            seed=1,
            points_xy=[(5, 5)],
        )

        self.assertEqual(out.getpixel((5, 5)), (0, 0, 0))
        self.assertEqual(out.getpixel((15, 15)), (255, 255, 255))

    def test_project_fingertips_from_joints_uses_tip_indices(self):
        hard = load_script()
        joints = [[0.0, 0.0, 1.0] for _ in range(21)]
        for joint_id in (4, 8, 12, 16, 20):
            joints[joint_id] = [float(joint_id), float(joint_id + 1), 1.0]
        K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]

        tips = hard.project_fingertips_from_joints(joints, K)

        self.assertEqual(tips[0], (41.0, 52.0))
        self.assertEqual(tips[-1], (201.0, 212.0))

    def test_build_freihand_subset_writes_hard_root_and_manifest(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "freihand"
            out = Path(tmp) / "hard"
            (src / "evaluation" / "rgb").mkdir(parents=True)
            Image.new("RGB", (224, 224), (255, 255, 255)).save(src / "evaluation" / "rgb" / "00000000.jpg")
            Image.new("RGB", (224, 224), (255, 255, 255)).save(src / "evaluation" / "rgb" / "00000001.jpg")
            K = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
            verts = [[[0.2, 0.2, 1.0], [0.8, 0.8, 1.0]], [[0.1, 0.1, 1.0], [0.3, 0.3, 1.0]]]
            xyz = [[[0.2, 0.2, 1.0]], [[0.1, 0.1, 1.0]]]
            (src / "evaluation_K.json").write_text(json.dumps([K, K]))
            (src / "evaluation_verts.json").write_text(json.dumps(verts))
            (src / "evaluation_xyz.json").write_text(json.dumps(xyz))

            hard.build_freihand_hard_root(src, out, effect="mask", severity=0.5, limit=1, seed=3)

            self.assertTrue((out / "evaluation" / "rgb" / "00000000.jpg").exists())
            self.assertEqual(len(json.loads((out / "evaluation_xyz.json").read_text())), 1)
            self.assertEqual(len(json.loads((out / "evaluation_verts.json").read_text())), 1)
            self.assertEqual(len(json.loads((out / "evaluation_K.json").read_text())), 1)
            rows = [(json.loads(line)) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "00000000")
            self.assertEqual(rows[0]["effect"], "mask")

    def test_build_ho3d_subset_writes_evaluation_txt_and_subset_gt(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0000.png")
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0001.png")
            for frame in ("0000", "0001"):
                with (meta / f"{frame}.pkl").open("wb") as f:
                    pickle.dump({"handBoundingBox": [10, 10, 50, 50], "handJoints3D": [[0, 0, 0]]}, f)
            (src / "evaluation.txt").write_text("AP10/0000\nAP10/0001\n")
            (src / "evaluation_xyz.json").write_text(json.dumps([[[0, 0, 0]], [[1, 1, 1]]]))
            (src / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]], [[1, 1, 1]]]))

            hard.build_ho3d_hard_root(src, out, effect="mask", severity=0.5, limit=1, seed=3)

            self.assertEqual((out / "evaluation.txt").read_text(), "AP10/0000\n")
            self.assertTrue((out / "evaluation" / "AP10" / "rgb" / "0000.png").exists())
            self.assertTrue((out / "evaluation" / "AP10" / "meta" / "0000.pkl").exists())
            self.assertEqual(len(json.loads((out / "evaluation_xyz.json").read_text())), 1)
            rows = [(json.loads(line)) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "AP10/0000")

    def test_build_ho3d_subset_can_use_run_meta_sample_order(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            order_path = Path(tmp) / "run_meta.json"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0000.png")
            with (meta / "0000.pkl").open("wb") as f:
                pickle.dump({"handBoundingBox": [10, 10, 50, 50], "handJoints3D": [[0, 0, 0]]}, f)
            (src / "evaluation_xyz.json").write_text(json.dumps([[[0, 0, 0]]]))
            (src / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]]]))
            order_path.write_text(json.dumps({"sample_order": ["AP10/0000"]}))

            hard.build_ho3d_hard_root(src, out, effect="mask", severity=0.5, limit=1, seed=3, sample_order_file=order_path)

            self.assertEqual((out / "evaluation.txt").read_text(), "AP10/0000\n")

    def test_build_ho3d_uses_evaluation_xyz_for_fingertip_points(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0000.png")
            K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]
            with (meta / "0000.pkl").open("wb") as f:
                pickle.dump({"handBoundingBox": [10, 10, 50, 50], "handJoints3D": [0, 0, 1], "camMat": K}, f)
            joints = [[0.0, 0.0, 1.0] for _ in range(21)]
            joints[4] = [4.0, 5.0, 1.0]
            (src / "evaluation_xyz.json").write_text(json.dumps([joints]))
            (src / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]]]))
            (src / "evaluation.txt").write_text("AP10/0000\n")

            hard.build_ho3d_hard_root(src, out, effect="tip_square", severity=0.5, limit=1, seed=3)

            rows = [(json.loads(line)) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["points_xy"][0], [41.0, 52.0])


if __name__ == "__main__":
    unittest.main()
