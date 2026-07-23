import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

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
        hard = load_script("scripts/datasets/make_hard_images.py")
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
        maker = load_script("scripts/datasets/make_rope_labels.py")
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

if __name__ == "__main__":
    unittest.main()
