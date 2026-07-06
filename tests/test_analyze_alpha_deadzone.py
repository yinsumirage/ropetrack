import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ropetrack.refine.actions import FINGER_POSE_GROUPS


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "rope_refiner" / "analyze_alpha_deadzone.py"
    spec = importlib.util.spec_from_file_location("analyze_alpha_deadzone", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_inputs(num_samples: int = 32, deadzone: bool = True, seed: int = 43):
    """Toy data where closure tracks base curl iff `deadzone` is True."""
    rng = np.random.default_rng(seed)
    curl_scale = rng.uniform(0.05, 1.0, size=(num_samples, 5))
    base_pose = np.zeros((num_samples, 45), dtype=np.float32)
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        # put all magnitude on the first joint's z component
        base_pose[:, 3 * joints[0] + 2] = curl_scale[:, finger_idx]

    alpha = (0.1 * curl_scale).astype(np.float32) if deadzone else np.full((num_samples, 5), 0.05, dtype=np.float32)
    base_residual = np.full((num_samples, 5), 0.4)
    closure = 0.3 * curl_scale if deadzone else np.full((num_samples, 5), 0.2)
    refined_residual = base_residual - closure
    valid = np.ones((num_samples, 5), dtype=bool)
    return base_pose, alpha, base_residual, refined_residual, valid


class BuildReportTest(unittest.TestCase):
    def setUp(self):
        self.script = load_script()

    def test_deadzone_signature_positive_correlation(self):
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs(deadzone=True)
        report = self.script.build_report(base_pose, alpha, base_res, refined_res, valid, "mult5", 4)
        self.assertGreater(report["pooled"]["pearson_curl_vs_closure"], 0.9)
        self.assertGreater(report["pooled"]["spearman_curl_vs_closure"], 0.9)
        buckets = report["curl_buckets"]
        self.assertEqual(len(buckets), 4)
        # low-curl bucket closes less than high-curl bucket
        self.assertLess(buckets[0]["mean_closure"], buckets[-1]["mean_closure"])

    def test_no_deadzone_gives_flat_closure(self):
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs(deadzone=False)
        report = self.script.build_report(base_pose, alpha, base_res, refined_res, valid, "mult5", 4)
        self.assertTrue(
            np.isnan(report["pooled"]["pearson_curl_vs_closure"])
            or abs(report["pooled"]["pearson_curl_vs_closure"]) < 0.2
        )
        buckets = report["curl_buckets"]
        self.assertAlmostEqual(buckets[0]["mean_closure"], buckets[-1]["mean_closure"], places=6)

    def test_invalid_fingers_are_ignored(self):
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs()
        valid[:, 2] = False
        refined_res[:, 2] = -99.0  # garbage that must not leak into stats
        report = self.script.build_report(base_pose, alpha, base_res, refined_res, valid, "mult5", 4)
        self.assertEqual(report["num_valid_fingers"], int(valid.sum()))
        for row in report["curl_buckets"]:
            if row.get("n"):
                self.assertGreater(row["mean_closure"], -1.0)

    def test_per_finger_keys(self):
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs()
        report = self.script.build_report(base_pose, alpha, base_res, refined_res, valid, "mult5", 4)
        self.assertEqual(
            sorted(report["per_finger"]), ["index", "middle", "pinky", "ring", "thumb"]
        )


class MainEndToEndTest(unittest.TestCase):
    def test_main_writes_outputs(self):
        script = load_script()
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "cache.npz", base_hand_pose=base_pose, sample_id=np.arange(len(base_pose)).astype(str))
            np.save(root / "alpha.npy", alpha)
            np.savez(
                root / "residuals.npz",
                base_rope_residual=base_res,
                refined_rope_residual=refined_res,
                rope_valid=valid,
                sample_id=np.arange(len(base_pose)).astype(str),
            )
            out_dir = root / "out"
            script.main([
                "--cache", str(root / "cache.npz"),
                "--alpha", str(root / "alpha.npy"),
                "--residuals", str(root / "residuals.npz"),
                "--action-space", "mult5",
                "--output-dir", str(out_dir),
            ])
            report = json.loads((out_dir / "alpha_deadzone.json").read_text(encoding="utf-8"))
            self.assertEqual(report["action_space"], "mult5")
            self.assertTrue((out_dir / "alpha_deadzone.tsv").exists())

    def test_length_mismatch_raises(self):
        script = load_script()
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "cache.npz", base_hand_pose=base_pose[:-1], sample_id=np.arange(len(base_pose) - 1).astype(str))
            np.save(root / "alpha.npy", alpha)
            np.savez(
                root / "residuals.npz",
                base_rope_residual=base_res,
                refined_rope_residual=refined_res,
                rope_valid=valid,
                sample_id=np.arange(len(base_pose)).astype(str),
            )
            with self.assertRaises(ValueError):
                script.main([
                    "--cache", str(root / "cache.npz"),
                    "--alpha", str(root / "alpha.npy"),
                    "--residuals", str(root / "residuals.npz"),
                    "--action-space", "mult5",
                    "--output-dir", str(root / "out"),
                ])

    def test_sample_id_order_mismatch_raises(self):
        script = load_script()
        base_pose, alpha, base_res, refined_res, valid = synthetic_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.arange(len(base_pose)).astype(str)
            np.savez(root / "cache.npz", base_hand_pose=base_pose, sample_id=ids)
            np.save(root / "alpha.npy", alpha)
            np.savez(
                root / "residuals.npz",
                base_rope_residual=base_res,
                refined_rope_residual=refined_res,
                rope_valid=valid,
                sample_id=ids[::-1].copy(),
            )
            with self.assertRaises(ValueError):
                script.main([
                    "--cache", str(root / "cache.npz"),
                    "--alpha", str(root / "alpha.npy"),
                    "--residuals", str(root / "residuals.npz"),
                    "--action-space", "mult5",
                    "--output-dir", str(root / "out"),
                ])


if __name__ == "__main__":
    unittest.main()
