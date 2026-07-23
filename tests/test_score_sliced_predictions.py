import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "ropetrack" / "eval" / "slices.py"
    spec = importlib.util.spec_from_file_location("score_sliced_predictions", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_gt(num_samples: int, seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.03, size=(num_samples, 21, 3))


class OccludedFingersTest(unittest.TestCase):
    def setUp(self):
        self.script = load_script()

    def test_tip_effects_occlude_all_fingers(self):
        for effect in ("tip_circle", "tip_square", "tip_blur", "tip_mixed", "finger_end", "blur"):
            row = {"effect": effect, "bbox_xyxy": [0, 0, 100, 100], "severity": 0.8}
            self.assertEqual(self.script.occluded_fingers_for_row(row, (224, 224)), [True] * 5)

    def test_mask_effect_uses_centered_rect(self):
        # bbox 0..100, severity 0.5 -> rect 25..75
        row = {
            "effect": "mask",
            "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
            "severity": 0.5,
            "points_xy": [[50.0, 50.0], [10.0, 10.0], [74.0, 30.0], [90.0, 90.0], [26.0, 73.0]],
        }
        flags = self.script.occluded_fingers_for_row(row, (224, 224))
        self.assertEqual(flags, [True, False, True, False, True])

    def test_mask_boundary_matches_painted_pixels(self):
        # PIL paints rect end-inclusive: pixels 25..75 are black, so
        # continuous coords in [25, 76) count as occluded.
        row = {
            "effect": "mask",
            "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
            "severity": 0.5,
            "points_xy": [[75.5, 30.0], [76.0, 30.0], [24.9, 30.0], [25.0, 30.0], [50.0, 75.9]],
        }
        flags = self.script.occluded_fingers_for_row(row, (224, 224))
        self.assertEqual(flags, [True, False, False, True, True])

    def test_mask_with_missing_points_is_undecidable(self):
        row = {"effect": "mask", "bbox_xyxy": [0, 0, 100, 100], "severity": 0.5, "points_xy": [[50.0, 50.0]]}
        self.assertIsNone(self.script.occluded_fingers_for_row(row, (224, 224)))

    def test_rng_dependent_effects_are_undecidable(self):
        for effect in ("crop", "mixed"):
            row = {"effect": effect, "bbox_xyxy": [0, 0, 100, 100], "severity": 0.5}
            self.assertIsNone(self.script.occluded_fingers_for_row(row, (224, 224)))

    def test_rect_helpers_match_make_hard_images(self):
        # Reference values from scripts/datasets/make_hard_images.py.
        self.assertEqual(self.script.clamp_bbox([-5.0, 10.0, 300.0, 200.0], 224, 224), (0, 10, 224, 200))
        self.assertEqual(self.script.centered_rect(0, 0, 100, 100, 0.5), (25, 25, 75, 75))


class PerJointDistancesTest(unittest.TestCase):
    def setUp(self):
        self.script = load_script()

    def test_zero_for_identical(self):
        gt = make_gt(2)
        distances = self.script.per_joint_pa_distances(gt, gt.copy())
        self.assertLess(float(distances.max()), 1e-8)

    def test_invariant_to_similarity_transform(self):
        gt = make_gt(2)
        pred = 1.7 * gt + np.asarray([0.5, -0.2, 0.1])
        distances = self.script.per_joint_pa_distances(gt, pred)
        self.assertLess(float(distances.max()), 1e-7)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            self.script.per_joint_pa_distances(np.zeros((2, 21, 3)), np.zeros((3, 21, 3)))


class BuildReportTest(unittest.TestCase):
    NUM = 8
    THUMB_TIP = 4  # freihand thumb fingertip joint id

    def setUp(self):
        self.script = load_script()
        self.gt = make_gt(self.NUM)
        rng = np.random.default_rng(31)
        # base: large error on the thumb tip, small noise elsewhere
        self.base = self.gt + rng.normal(scale=1e-4, size=self.gt.shape)
        self.base[:, self.THUMB_TIP] += np.asarray([0.02, 0.0, 0.0])
        # refined: thumb-tip error halved, everything else untouched
        self.refined = self.base.copy()
        self.refined[:, self.THUMB_TIP] -= np.asarray([0.01, 0.0, 0.0])
        # manifest: thumb occluded on every sample (mask covers only the thumb tip point)
        self.manifest = [
            {
                "sample_id": f"{i:08d}",
                "effect": "mask",
                "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
                "severity": 0.5,
                "points_xy": [[50.0, 50.0], [1.0, 1.0], [1.0, 99.0], [99.0, 1.0], [99.0, 99.0]],
            }
            for i in range(self.NUM)
        ]

    def _cache(self, residual_matches_error: bool = True) -> dict:
        rng = np.random.default_rng(37)
        base_rope = rng.uniform(0.2, 0.8, size=(self.NUM, 5)).astype(np.float32)
        input_rope = base_rope.copy()
        # thumb residual present on all samples
        input_rope[:, 0] = base_rope[:, 0] + (0.3 if residual_matches_error else 0.0)
        return {
            "sample_id": np.asarray([f"{i:08d}" for i in range(self.NUM)]),
            "base_rope_norm": base_rope,
            "input_rope_norm": input_rope,
            "rope_valid": np.ones((self.NUM, 5), dtype=bool),
        }

    def test_slices_show_concentrated_gain(self):
        report = self.script.build_report(
            "freihand", self.gt, self.base, self.refined, self.manifest, self._cache(), (224, 224), 4
        )
        slices = report["slices"]
        self.assertEqual(report["num_samples"], self.NUM)
        self.assertEqual(report["occlusion"]["num_with_occlusion_info"], self.NUM)
        # refinement helps: deltas negative where the fix landed
        self.assertLess(slices["occluded_fingertips"]["delta_cm"], 0.0)
        self.assertLess(slices["all_joints"]["delta_cm"], 0.0)
        # concentration: occluded-tip improvement much larger than the all-joint average
        self.assertLess(slices["occluded_fingertips"]["delta_cm"], 4.0 * slices["all_joints"]["delta_cm"])
        # untouched fingers should not change materially
        self.assertLess(abs(slices["clean_fingertips"]["delta_cm"]), abs(slices["occluded_fingertips"]["delta_cm"]) / 5.0)
        # per-finger table: thumb occluded on all samples
        thumb = report["per_finger"]["thumb"]
        self.assertEqual(thumb["occluded"]["n"], self.NUM)
        self.assertEqual(thumb["clean"]["n"], 0)
        self.assertLess(thumb["occluded"]["tip_delta_cm"], 0.0)

    def test_report_without_manifest_and_cache(self):
        report = self.script.build_report(
            "freihand", self.gt, self.base, self.refined, None, None, (224, 224), 4
        )
        self.assertIn("all_joints", report["slices"])
        self.assertIn("fingertips", report["slices"])
        self.assertNotIn("occluded_fingertips", report["slices"])
        self.assertNotIn("residual_buckets", report)

    def test_cache_manifest_id_mismatch_raises(self):
        cache = self._cache()
        cache["sample_id"] = cache["sample_id"][::-1].copy()
        with self.assertRaises(ValueError):
            self.script.build_report(
                "freihand", self.gt, self.base, self.refined, self.manifest, cache, (224, 224), 4
            )

    def test_manifest_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.script.build_report(
                "freihand", self.gt, self.base, self.refined, self.manifest[:-1], None, (224, 224), 4
            )

    def test_residual_buckets_and_correlations_present(self):
        report = self.script.build_report(
            "freihand", self.gt, self.base, self.refined, self.manifest, self._cache(), (224, 224), 4
        )
        buckets = report["residual_buckets"]
        self.assertEqual(len(buckets), 4)
        self.assertEqual(sum(row.get("n", 0) for row in buckets), self.NUM)
        self.assertIn("all_joints", report["correlations"])
        self.assertIn("pearson_residual_vs_improvement", report["correlations"]["all_joints"])


class MainEndToEndTest(unittest.TestCase):
    def test_main_writes_json_and_tsv(self):
        script = load_script()
        num = 6
        gt = make_gt(num, seed=41)
        base = gt + 0.001
        refined = gt.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation_xyz.json").write_text(json.dumps(gt.tolist()), encoding="utf-8")
            (root / "base_pred.json").write_text(json.dumps([base.tolist(), [[[0.0, 0.0, 0.0]]] * num]), encoding="utf-8")
            (root / "pred.json").write_text(json.dumps([refined.tolist(), [[[0.0, 0.0, 0.0]]] * num]), encoding="utf-8")
            out_dir = root / "out"

            script.main([
                str(root), str(out_dir),
                "--dataset", "freihand",
            ])

            report = json.loads((out_dir / "sliced_scores.json").read_text(encoding="utf-8"))
            self.assertEqual(report["num_samples"], num)
            self.assertAlmostEqual(report["slices"]["all_joints"]["refined_cm"], 0.0, places=6)
            tsv = (out_dir / "sliced_scores.tsv").read_text(encoding="utf-8")
            self.assertIn("all_joints", tsv)

    def test_all_occluded_manifest_writes_strict_json(self):
        # tip_* manifests occlude every finger, so clean slices are empty;
        # their NaN means must serialize as null, not the invalid NaN token.
        script = load_script()
        num = 4
        gt = make_gt(num, seed=47)
        # non-rigid per-joint noise: a constant offset would be absorbed by PA
        base = gt + np.random.default_rng(53).normal(scale=0.002, size=gt.shape)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation_xyz.json").write_text(json.dumps(gt.tolist()), encoding="utf-8")
            (root / "base_pred.json").write_text(json.dumps([base.tolist(), [[[0.0, 0.0, 0.0]]] * num]), encoding="utf-8")
            (root / "pred.json").write_text(json.dumps([gt.tolist(), [[[0.0, 0.0, 0.0]]] * num]), encoding="utf-8")
            manifest = root / "hard_manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps({
                        "sample_id": f"{i:08d}",
                        "effect": "tip_square",
                        "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
                        "severity": 0.8,
                    })
                    for i in range(num)
                ) + "\n",
                encoding="utf-8",
            )
            out_dir = root / "out"

            script.main([
                str(root), str(out_dir),
                "--dataset", "freihand",
                "--hard-manifest", str(manifest),
            ])

            text = (out_dir / "sliced_scores.json").read_text(encoding="utf-8")
            self.assertNotIn("NaN", text)
            report = json.loads(text)
            self.assertIsNone(report["slices"]["clean_fingertips"]["base_cm"])
            self.assertLess(report["slices"]["occluded_fingertips"]["delta_cm"], 0.0)


if __name__ == "__main__":
    unittest.main()
