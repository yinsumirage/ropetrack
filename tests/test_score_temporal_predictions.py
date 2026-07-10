import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "score_temporal_predictions.py"


def load_script():
    if not SCRIPT_PATH.exists():
        raise AssertionError("temporal scorer module is missing")
    spec = importlib.util.spec_from_file_location("score_temporal_predictions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MotionMetricsTest(unittest.TestCase):
    def test_motion_metrics_reset_at_gap(self):
        script = load_script()
        ids = np.asarray(
            ["A/0000", "A/0001", "A/0002", "A/0004", "A/0005", "B/0000", "B/0001"]
        )
        gt = np.zeros((7, 21, 3), dtype=np.float32)
        gt[:, :, 0] = np.asarray([0, 1, 2, 10, 11, 20, 21])[:, None] * 0.001

        same = script.temporal_motion_metrics(ids, gt, gt, fps=30.0, raw_frame_step=1)

        self.assertAlmostEqual(same["velocity_error_mm_s"], 0.0)
        self.assertAlmostEqual(same["acceleration_error_mm_s2"], 0.0)
        self.assertAlmostEqual(same["prediction_acceleration_mm_s2"], 0.0)
        self.assertEqual(same["num_velocity_edges"], 4)
        self.assertEqual(same["num_acceleration_triplets"], 1)

    def test_motion_metrics_use_root_relative_fps_and_millimeter_units_without_bridging_gap(self):
        script = load_script()
        ids = np.asarray(["A/0000", "A/0001", "A/0002", "A/0004", "A/0005"])
        gt = np.zeros((5, 21, 3), dtype=np.float64)
        pred = np.zeros_like(gt)
        pred[:, 1, 0] = [0.0, 0.001, 0.004, 10.0, 10.001]

        metrics = script.temporal_motion_metrics(ids, gt, pred, fps=30.0, raw_frame_step=1)

        self.assertAlmostEqual(metrics["velocity_error_mm_s"], 0.15 / (3 * 21) * 1000.0)
        self.assertAlmostEqual(metrics["acceleration_error_mm_s2"], 1.8 / 21 * 1000.0)
        self.assertAlmostEqual(metrics["jitter_mm_s2"], 1.8 / 21 * 1000.0)
        self.assertEqual(metrics["num_velocity_edges"], 3)
        self.assertEqual(metrics["num_acceleration_triplets"], 1)

    def test_phase_lag_positive_means_prediction_trails(self):
        script = load_script()
        gt = np.sin(np.linspace(0, 4 * np.pi, 80))
        pred = np.concatenate([np.repeat(gt[0], 2), gt[:-2]])

        self.assertEqual(script.phase_lag(gt, pred, max_lag=15), 2)

    def test_phase_lag_excludes_constant_series(self):
        script = load_script()
        self.assertIsNone(script.phase_lag(np.ones(40), np.arange(40), max_lag=15))

    def test_recovery_requires_three_consecutive_frames(self):
        script = load_script()
        error = np.asarray([1.0] * 30 + [5.0] * 60 + [2.0, 1.4, 1.4, 1.4] + [1.0] * 26)

        recovered = script.recovery_frames(
            error, context=30, masked=60, recovery=30, margin_mm=0.5, stable=3
        )

        self.assertEqual(recovered, 1)

    def test_unresolved_recovery_returns_recovery_length(self):
        script = load_script()
        error = np.asarray([1.0] * 2 + [5.0] * 2 + [2.0] * 3)
        self.assertEqual(
            script.recovery_frames(error, context=2, masked=2, recovery=3, margin_mm=0.5, stable=2),
            3,
        )

    def test_lag_aggregation_requires_long_segments_and_counts_exclusions(self):
        script = load_script()
        frames = 50
        ids = np.asarray([f"A/{frame:04d}" for frame in range(frames)])
        speed = 0.001 + np.linspace(0.0, 0.002, frames - 1) ** 2
        position = np.concatenate([[0.0], np.cumsum(speed)])
        gt = np.zeros((frames, 21, 3), dtype=np.float64)
        gt[:, 1, 0] = position
        delayed = gt.copy()
        delayed[:2, 1, 0] = gt[0, 1, 0]
        delayed[2:, 1, 0] = gt[:-2, 1, 0]

        metrics = script._lag_metrics(ids, gt, delayed, fps=30.0, raw_frame_step=1)
        constant = script._lag_metrics(ids, gt, np.zeros_like(gt), fps=30.0, raw_frame_step=1)

        self.assertEqual(metrics["phase_lag_frames"], 2.0)
        self.assertEqual(metrics["num_lag_segments"], 1)
        self.assertEqual(constant["num_lag_segments"], 0)
        self.assertEqual(constant["num_lag_segments_excluded"], 1)


class ScorerEndToEndTest(unittest.TestCase):
    def _fixture(self, root: Path):
        rng = np.random.default_rng(17)
        episode_length = 120
        ids = [f"{sequence}/{frame:04d}" for sequence in ("A", "B") for frame in range(episode_length)]
        gt_xyz = rng.normal(scale=0.03, size=(len(ids), 21, 3))
        gt_verts = rng.normal(scale=0.04, size=(len(ids), 8, 3))
        frame_xyz = gt_xyz.copy()
        frame_verts = gt_verts.copy()
        alternating = np.where(np.arange(len(ids)) % 2, 0.004, -0.004)
        frame_xyz[:, 1, 0] += alternating
        frame_verts[:, 2, 1] += alternating

        gt_dir = root / "gt"
        gt_dir.mkdir()
        (gt_dir / "evaluation_xyz.json").write_text(json.dumps(gt_xyz.tolist()), encoding="utf-8")
        (gt_dir / "evaluation_verts.json").write_text(json.dumps(gt_verts.tolist()), encoding="utf-8")
        run_meta = root / "run_meta.json"
        run_meta.write_text(json.dumps({"sample_order": ids}), encoding="utf-8")

        method_dirs = {}
        for name, xyz, verts, closure, latency in (
            ("frame", frame_xyz, frame_verts, 0.25, 1.5),
            ("temporal", gt_xyz, gt_verts, 0.75, 2.5),
        ):
            directory = root / name
            directory.mkdir()
            (directory / "pred.json").write_text(json.dumps([xyz.tolist(), verts.tolist()]), encoding="utf-8")
            base_residual = np.full((len(ids), 5), 0.4, dtype=np.float32)
            refined_residual = base_residual * (1.0 - closure)
            np.savez(
                directory / "rope_residuals.npz",
                sample_id=np.asarray(ids),
                base_rope_residual=base_residual,
                refined_rope_residual=refined_residual,
                rope_valid=np.ones_like(base_residual, dtype=bool),
            )
            (directory / "summary.json").write_text(
                json.dumps({"num_samples": len(ids), "timing": {"wall_seconds": 0.1, "per_sample_ms": latency}}),
                encoding="utf-8",
            )
            method_dirs[name] = directory

        manifest = root / "hard_manifest.jsonl"
        rows = []
        for sequence in ("A", "B"):
            for frame in range(episode_length):
                phase = "context" if frame < 30 else "masked" if frame < 90 else "recovery"
                rows.append(
                    {
                        "sample_id": f"{sequence}/{frame:04d}",
                        "episode_id": f"{sequence}:0",
                        "episode_phase": phase,
                        "episode_offset": frame,
                        "segment_id": sequence,
                        "effect": "mask",
                        "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
                        "severity": 0.8,
                        "points_xy": [[50.0, 50.0]] * 5,
                    }
                )
        manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return ids, gt_dir, run_meta, manifest, method_dirs

    def test_main_writes_phase_metrics_paired_deltas_and_sequence_bootstrap(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, gt_dir, run_meta, manifest, methods = self._fixture(root)
            output = root / "scores.json"

            result = script.main(
                [
                    "--method", f"frame={methods['frame']}",
                    "--method", f"temporal={methods['temporal']}",
                    "--reference", "frame",
                    "--gt-dir", str(gt_dir),
                    "--run-meta", str(run_meta),
                    "--hard-manifest", str(manifest),
                    "--fps", "30",
                    "--raw-frame-step", "1",
                    "--output", str(output),
                ]
            )

            self.assertEqual(result, output)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                set(("methods", "reference", "paired_deltas", "bootstrap_ci", "rope_closure", "timing"))
                - set(report),
                set(),
            )
            self.assertEqual(report["reference"], "frame")
            self.assertLess(report["methods"]["temporal"]["pa_mpjpe_mm"], report["methods"]["frame"]["pa_mpjpe_mm"])
            self.assertIn("masked_pa_mpjpe_mm", report["methods"]["temporal"])
            self.assertIn("masked_occluded_tip_pa_mpjpe_mm", report["methods"]["temporal"])
            self.assertIn("recovery_frames", report["methods"]["temporal"])
            self.assertLess(report["paired_deltas"]["temporal"]["pa_mpjpe_mm"], 0.0)
            self.assertLess(report["paired_deltas"]["temporal"]["masked_occluded_tip_pa_mpjpe_mm"], 0.0)
            self.assertEqual(len(report["bootstrap_ci"]["temporal"]["pa_mpjpe_mm"]), 2)
            self.assertEqual(
                len(report["bootstrap_ci"]["temporal"]["masked_occluded_tip_pa_mpjpe_mm"]), 2
            )
            self.assertAlmostEqual(report["rope_closure"]["temporal"], 0.75, places=6)
            self.assertEqual(report["timing"]["frame"]["per_sample_ms"], 1.5)
            self.assertEqual(report["protocol"]["bootstrap_iterations"], 2000)
            self.assertEqual(report["protocol"]["bootstrap_seed"], 20260710)
            self.assertEqual(report["protocol"]["episode_phases"], {"context": 30, "masked": 60, "recovery": 30})
            self.assertNotIn("NaN", output.read_text(encoding="utf-8"))

    def test_method_residual_order_mismatch_fails_loudly(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, gt_dir, run_meta, _, methods = self._fixture(root)
            residual = methods["temporal"] / "rope_residuals.npz"
            values = np.full((len(ids), 5), 0.1, dtype=np.float32)
            np.savez(
                residual,
                sample_id=np.asarray(ids[::-1]),
                base_rope_residual=values,
                refined_rope_residual=values,
                rope_valid=np.ones_like(values, dtype=bool),
            )

            with self.assertRaisesRegex(ValueError, "sample_id order mismatch"):
                script.main(
                    [
                        "--method", f"frame={methods['frame']}",
                        "--method", f"temporal={methods['temporal']}",
                        "--reference", "frame",
                        "--gt-dir", str(gt_dir),
                        "--run-meta", str(run_meta),
                        "--output", str(root / "scores.json"),
                    ]
                )

    def test_base_export_may_omit_refiner_artifacts(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, gt_dir, run_meta, _, methods = self._fixture(root)
            base = root / "base"
            base.mkdir()
            (base / "pred.json").write_text(
                (methods["frame"] / "pred.json").read_text(encoding="utf-8"), encoding="utf-8"
            )

            report = script.build_report(
                {"base": base}, "base", gt_dir, run_meta, None, fps=30.0, raw_frame_step=1
            )

            self.assertIsNone(report["rope_closure"]["base"])
            self.assertIsNone(report["timing"]["base"])

    def test_prediction_row_count_mismatch_fails_loudly(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, gt_dir, run_meta, _, methods = self._fixture(root)
            pred_path = methods["temporal"] / "pred.json"
            xyz, verts = json.loads(pred_path.read_text(encoding="utf-8"))
            pred_path.write_text(json.dumps([xyz[:-1], verts[:-1]]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be \[240, P, 3\]"):
                script.build_report(
                    {"frame": methods["frame"], "temporal": methods["temporal"]},
                    "frame",
                    gt_dir,
                    run_meta,
                    None,
                    fps=30.0,
                    raw_frame_step=1,
                )

    def test_summary_schema_rejects_coercive_numbers(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                json.dumps({"num_samples": 12.9, "timing": {"wall_seconds": 0.1, "per_sample_ms": 1.0}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "num_samples"):
                script._load_timing(summary, 12)

            summary.write_text(
                json.dumps({"num_samples": 12, "timing": {"wall_seconds": True, "per_sample_ms": "1.0"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "timing"):
                script._load_timing(summary, 12)

    def test_manifest_segment_id_cannot_bridge_a_raw_frame_gap(self):
        script = load_script()
        order = [f"A/{frame:04d}" for frame in range(5)] + [f"A/{frame:04d}" for frame in range(10, 15)]
        phases = ("context", "masked", "recovery", "recovery", "recovery")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "hard_manifest.jsonl"
            rows = []
            for index, sample_id in enumerate(order):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "episode_id": f"episode-{index // 5}",
                        "episode_phase": phases[index % 5],
                        "episode_offset": index % 5,
                        "segment_id": "A",
                    }
                )
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "segment_id.*gap"):
                script._load_episode_manifest(manifest, order, raw_frame_step=1)

    def test_manifest_requires_globally_unique_episode_ids(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids, _, _, manifest, _ = self._fixture(root)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            for row in rows[120:]:
                row["episode_id"] = "A:0"
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "episode_id.*globally unique"):
                script._load_episode_manifest(manifest, ids, raw_frame_step=1)

    def test_occluded_tip_mask_ignores_retained_recipe_on_clean_phases(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, manifest, _ = self._fixture(root)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

            mask = script._masked_occluded_tip_mask(rows)

            phases = np.asarray([row["episode_phase"] for row in rows])
            self.assertFalse(mask[phases != "masked"].any())
            self.assertEqual(int(mask.sum()), 2 * 60 * 5)

    def test_recovery_uses_episode_offset_instead_of_storage_order(self):
        script = load_script()
        ids = np.asarray([f"A/{frame:04d}" for frame in range(120)])
        errors = np.asarray([1.0] * 30 + [5.0] * 60 + [2.0, 1.4, 1.4, 1.4] + [1.0] * 26)
        rows = [
            {
                "sample_id": ids[offset],
                "episode_id": "episode-0",
                "episode_phase": "context" if offset < 30 else "masked" if offset < 90 else "recovery",
                "episode_offset": offset,
                "segment_id": "A:0",
            }
            for offset in range(120)
        ]
        permutation = np.concatenate([np.arange(90), np.arange(119, 89, -1)])

        metrics = script._phase_metrics(
            errors[permutation],
            [rows[index] for index in permutation],
            ids[permutation],
            raw_frame_step=1,
        )

        self.assertEqual(metrics["recovery_frames"], 1.0)

    def test_phase_metrics_reject_non_registered_episode_lengths(self):
        script = load_script()
        phases = ("context", "masked", "recovery", "recovery", "recovery")
        ids = np.asarray([f"A/{frame:04d}" for frame in range(len(phases))])
        rows = [
            {
                "sample_id": sample_id,
                "episode_id": "episode-0",
                "episode_phase": phase,
                "episode_offset": offset,
                "segment_id": "A:0",
            }
            for offset, (sample_id, phase) in enumerate(zip(ids, phases, strict=True))
        ]

        with self.assertRaisesRegex(ValueError, "30/60/30"):
            script._phase_metrics(np.ones(len(phases)), rows, ids, raw_frame_step=1)


class SequenceBootstrapTest(unittest.TestCase):
    def test_bootstrap_resamples_whole_sequences_and_preserves_frame_weights(self):
        script = load_script()
        keys = ["A", "B", "C"]
        sequence_metrics = {
            "reference": {"pa_mpjpe_mm": {"A": (0.0, 9), "B": (0.0, 1), "C": (0.0, 1)}},
            "method": {"pa_mpjpe_mm": {"A": (0.0, 9), "B": (10.0, 1), "C": (20.0, 1)}},
        }
        iterations = 200
        seed = 13

        report = script.sequence_bootstrap_ci(keys, sequence_metrics, "reference", iterations=iterations, seed=seed)

        draws = np.random.default_rng(seed).integers(0, len(keys), size=(iterations, len(keys)))
        values = np.asarray([0.0, 10.0, 20.0])
        weights = np.asarray([9.0, 1.0, 1.0])
        expected = np.percentile(
            (values[draws] * weights[draws]).sum(axis=1) / weights[draws].sum(axis=1),
            [2.5, 97.5],
        )
        np.testing.assert_allclose(report["method"]["pa_mpjpe_mm"], expected)


if __name__ == "__main__":
    unittest.main()
