import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "rope_refiner" / "make_qualitative_panels.py"
    spec = importlib.util.spec_from_file_location("make_qualitative_panels", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_setup(tmp: Path, num: int = 6):
    rng = np.random.default_rng(3)
    gt = rng.normal(scale=0.03, size=(num, 21, 3))
    base = gt + rng.normal(scale=0.004, size=gt.shape)
    # sample 2 is where the student helps by far the most
    # (per-joint noise, not a constant offset: PA alignment absorbs translations)
    base[2] = gt[2] + rng.normal(scale=0.012, size=(21, 3))
    student = base.copy()
    student[2] = gt[2] + rng.normal(scale=0.0005, size=(21, 3))
    teacher = student + rng.normal(scale=0.0005, size=gt.shape)

    (tmp / "evaluation_xyz.json").write_text(json.dumps(gt.tolist()), encoding="utf-8")
    verts_stub = [[[0.0, 0.0, 0.0]]] * num
    for name, xyz in (("base_pred.json", base), ("teacher_pred.json", teacher), ("student_pred.json", student)):
        (tmp / name).write_text(json.dumps([xyz.tolist(), verts_stub]), encoding="utf-8")
    return gt, base, student


class QualitativePanelsTest(unittest.TestCase):
    def test_top_improved_selection_and_outputs(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_setup(root)
            out = root / "panels"
            script.main([
                "--dataset", "freihand",
                "--gt-dir", str(root),
                "--base", str(root / "base_pred.json"),
                "--variant", f"teacher={root / 'teacher_pred.json'}",
                "--variant", f"student={root / 'student_pred.json'}",
                "--top-k", "2",
                "--output-dir", str(out),
            ])
            manifest = json.loads((out / "panels_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest), 2)
            # the constructed most-improved sample must rank first
            self.assertEqual(manifest[0]["sample_index"], 2)
            self.assertGreater(manifest[0]["improvement_mm"], manifest[1]["improvement_mm"])
            self.assertIn("teacher", manifest[0]["variant_pa_mm"])
            for row in manifest:
                panel = out / row["panel"]
                self.assertTrue(panel.exists())
                self.assertGreater(panel.stat().st_size, 5000)

    def test_freihand_image_overlay_panel(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt, _, _ = write_setup(root, num=4)
            # push GT to sit in front of a synthetic camera so projection is finite
            gt = gt + np.asarray([0.0, 0.0, 0.5])
            (root / "evaluation_xyz.json").write_text(json.dumps(gt.tolist()), encoding="utf-8")
            K = [[500.0, 0.0, 112.0], [0.0, 500.0, 112.0], [0.0, 0.0, 1.0]]
            (root / "evaluation_K.json").write_text(json.dumps([K] * 4), encoding="utf-8")
            hard_root = root / "hard"
            rgb = hard_root / "evaluation" / "rgb"
            rgb.mkdir(parents=True)
            from PIL import Image

            for idx in range(4):
                Image.new("RGB", (224, 224), (40, 40, 40)).save(rgb / f"{idx:08d}.jpg")

            out = root / "panels"
            script.main([
                "--dataset", "freihand",
                "--gt-dir", str(root),
                "--base", str(root / "base_pred.json"),
                "--variant", f"student={root / 'student_pred.json'}",
                "--hard-root", str(hard_root),
                "--top-k", "1",
                "--output-dir", str(out),
            ])
            manifest = json.loads((out / "panels_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((out / manifest[0]["panel"]).exists())

    def test_rank_variant_validation(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_setup(root)
            with self.assertRaises(ValueError):
                script.main([
                    "--dataset", "freihand",
                    "--gt-dir", str(root),
                    "--base", str(root / "base_pred.json"),
                    "--variant", f"student={root / 'student_pred.json'}",
                    "--rank-variant", "nonexistent",
                    "--output-dir", str(root / "panels"),
                ])

    def test_shape_mismatch_raises(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_setup(root)
            bad = json.loads((root / "student_pred.json").read_text(encoding="utf-8"))
            bad[0] = bad[0][:-1]
            (root / "student_pred.json").write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                script.main([
                    "--dataset", "freihand",
                    "--gt-dir", str(root),
                    "--base", str(root / "base_pred.json"),
                    "--variant", f"student={root / 'student_pred.json'}",
                    "--output-dir", str(root / "panels"),
                ])


if __name__ == "__main__":
    unittest.main()
