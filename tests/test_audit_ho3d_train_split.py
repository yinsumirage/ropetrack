import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "audit_ho3d_train_split.py"
    spec = importlib.util.spec_from_file_location("audit_ho3d_train_split", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_meta(with_bbox: bool = True, with_nan: bool = False):
    meta = {
        "handJoints3D": np.random.default_rng(1).normal(size=(21, 3)),
        "handPose": np.zeros(48),
        "handBeta": np.zeros(10),
        "handTrans": np.zeros(3),
        "camMat": np.eye(3),
        "objName": "021_bleach_cleanser",
    }
    if with_bbox:
        meta["handBoundingBox"] = [10.0, 10.0, 200.0, 200.0]
    if with_nan:
        meta["handJoints3D"] = np.full((21, 3), np.nan)
    return meta


def build_root(tmp: Path, num_seq: int = 2, frames_per_seq: int = 3, ext: str = ".jpg", with_bbox: bool = True):
    ids = []
    for seq_idx in range(num_seq):
        seq = f"SEQ{seq_idx}"
        rgb = tmp / "train" / seq / "rgb"
        meta_dir = tmp / "train" / seq / "meta"
        rgb.mkdir(parents=True)
        meta_dir.mkdir(parents=True)
        for frame_idx in range(frames_per_seq):
            frame = f"{frame_idx:04d}"
            (rgb / f"{frame}{ext}").write_bytes(b"fake image bytes")
            with (meta_dir / f"{frame}.pkl").open("wb") as handle:
                pickle.dump(make_meta(with_bbox=with_bbox, with_nan=(seq_idx == 0 and frame_idx == 0)), handle)
            ids.append(f"{seq}/{frame}")
    (tmp / "train.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


class AuditHo3dTrainSplitTest(unittest.TestCase):
    def test_full_audit_report(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = build_root(root)
            out = root / "report.json"
            script.main([
                "--input-root", str(root),
                "--sample-count", "6",
                "--output", str(out),
            ])
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["num_split_ids"], len(ids))
            self.assertEqual(report["num_sequences"], 2)
            self.assertEqual(report["num_metas_audited"], 6)
            self.assertEqual(report["image_extensions"], {".jpg": 6})
            self.assertAlmostEqual(report["frac_with_handBoundingBox"], 1.0)
            self.assertEqual(report["missing_images"], [])
            self.assertEqual(report["shape_failures"], {})
            self.assertEqual(report["nan_annotation_counts"], {"handJoints3D": 1})

    def test_detects_missing_bbox_and_images(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_root(root, with_bbox=False)
            # remove one image to trigger the missing-image path
            victim = next((root / "train" / "SEQ0" / "rgb").iterdir())
            victim.unlink()
            out = root / "report.json"
            script.main(["--input-root", str(root), "--sample-count", "6", "--output", str(out)])
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["frac_with_handBoundingBox"], 0.0)
            self.assertGreaterEqual(len(report["missing_images"]), 1)

    def test_bad_split_format_raises(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.txt").write_text("no_slash_id\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                script.main(["--input-root", str(root), "--output", str(root / "r.json")])


if __name__ == "__main__":
    unittest.main()
