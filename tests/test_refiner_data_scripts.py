import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script(rel_path: str):
    path = ROOT / rel_path
    name = rel_path.replace("/", "_").replace("\\", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def freihand_joints(tip_x: float = 4.0):
    joints = [[0.0, 0.0, 1.0] for _ in range(21)]
    for joint_id in range(1, 5):
        joints[joint_id] = [float(joint_id), 0.0, 1.0]
    joints[4] = [tip_x, 0.0, 1.0]
    return joints


class RefinerDataScriptsTest(unittest.TestCase):
    def test_make_hard_images_builds_freihand_training_root(self):
        hard = load_script("scripts/make_hard_images.py")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "freihand"
            out = Path(tmp) / "hard"
            (src / "training" / "rgb").mkdir(parents=True)
            Image.new("RGB", (224, 224), (255, 255, 255)).save(src / "training" / "rgb" / "00000000.jpg")
            K = [[20.0, 0.0, 20.0], [0.0, 20.0, 20.0], [0.0, 0.0, 1.0]]
            write_json(src / "training_K.json", [K])
            write_json(src / "training_verts.json", [[[0.1, 0.1, 1.0], [0.8, 0.8, 1.0]]])
            write_json(src / "training_xyz.json", [freihand_joints()])

            hard.build_freihand_hard_root(src, out, "mask", 0.5, limit=1, seed=1, split="training")

            self.assertTrue((out / "training" / "rgb" / "00000000.jpg").exists())
            self.assertTrue((out / "training_K.json").exists())
            self.assertTrue((out / "training_verts.json").exists())
            self.assertTrue((out / "training_xyz.json").exists())
            rows = [json.loads(line) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "00000000")

    def test_make_rope_labels_writes_freihand_training_labels(self):
        maker = load_script("scripts/make_rope_labels.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "freihand"
            out = Path(tmp) / "rope.jsonl"
            (root / "training" / "rgb").mkdir(parents=True)
            write_json(root / "training_xyz.json", [freihand_joints()])
            write_json(root / "training_K.json", [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

            maker.write_rope_labels("freihand", root, out, split="training")

            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "00000000")
            self.assertEqual(rows[0]["dataset"], "freihand")
            self.assertAlmostEqual(rows[0]["rope_norm"][0], 1.0)

    def test_build_freihand_refiner_cache_writes_expected_arrays(self):
        builder = load_script("scripts/rope_refiner/build_freihand_refiner_cache.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "freihand"
            pred_dir = Path(tmp) / "pred"
            rope_labels = Path(tmp) / "rope.jsonl"
            run_meta = Path(tmp) / "run_meta.json"
            output = Path(tmp) / "cache.npz"
            pred_dir.mkdir()
            mano = [float(i) for i in range(61)]
            write_json(root / "training_mano.json", [[mano]])
            gt_row = {
                "sample_id": "00000000",
                "dataset": "freihand",
                "rope_norm": [1.0, None, None, None, None],
                "gt_rope_norm": [9.0, 9.0, 9.0, 9.0, 9.0],
                "rope_chain_m": [4.0, None, None, None, None],
                "rope_valid": [True, False, False, False, False],
                "normalization": {"fist_ratio": 0.5},
            }
            rope_labels.write_text(json.dumps(gt_row) + "\n", encoding="utf-8")
            pred_joints = freihand_joints(tip_x=3.0)
            write_json(pred_dir / "pred.json", [[pred_joints], [[[0.0, 0.0, 0.0]]]])
            write_json(run_meta, {"sample_order": ["00000000"]})

            builder.build_cache(root, rope_labels, pred_dir, run_meta, output, split="training")

            with np.load(output) as data:
                self.assertEqual(data["sample_id"].tolist(), ["00000000"])
                self.assertEqual(data["base_hand_pose"].shape, (1, 45))
                self.assertEqual(data["target_hand_pose"].shape, (1, 45))
                np.testing.assert_array_equal(data["base_hand_pose"], data["target_hand_pose"])
                self.assertAlmostEqual(float(data["target_hand_pose"][0, 0]), 3.0)
                self.assertAlmostEqual(float(data["input_rope_norm"][0, 0]), 1.0)
                self.assertAlmostEqual(float(data["gt_rope_norm"][0, 0]), 1.0)
                self.assertAlmostEqual(float(data["base_rope_norm"][0, 0]), 0.5)
                self.assertEqual(data["rope_valid"].tolist(), [[True, False, False, False, False]])


if __name__ == "__main__":
    unittest.main()
