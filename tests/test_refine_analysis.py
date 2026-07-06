import unittest

import numpy as np

from ropetrack.refine.analysis import (
    bucket_indices,
    json_sanitize,
    pearson,
    quantile_bucket_edges,
    rope_abs_residual,
    spearman,
    summarize_rope_residuals,
)


class JsonSanitizeTest(unittest.TestCase):
    def test_replaces_non_finite_floats(self):
        import json

        obj = {
            "a": float("nan"),
            "b": [1.0, float("inf"), {"c": float("-inf")}],
            "d": "NaN-looking string",
            "e": 2,
        }
        clean = json_sanitize(obj)
        self.assertIsNone(clean["a"])
        self.assertIsNone(clean["b"][1])
        self.assertIsNone(clean["b"][2]["c"])
        self.assertEqual(clean["d"], "NaN-looking string")
        self.assertEqual(clean["e"], 2)
        json.loads(json.dumps(clean, allow_nan=False))  # strict JSON round-trip


class RopeResidualTest(unittest.TestCase):
    def test_masked_residual(self):
        pred = np.asarray([[0.5, 0.2, 0.0, 0.0, 0.0]])
        target = np.asarray([[0.3, 0.6, 0.0, 0.0, 0.0]])
        valid = np.asarray([[True, True, False, False, False]])
        residual = rope_abs_residual(pred, target, valid)
        self.assertAlmostEqual(residual[0, 0], 0.2, places=6)
        self.assertAlmostEqual(residual[0, 1], 0.4, places=6)
        self.assertTrue(np.isnan(residual[0, 2:]).all())

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            rope_abs_residual(np.zeros((1, 5)), np.zeros((2, 5)), np.ones((1, 5), dtype=bool))

    def test_summary_closure(self):
        valid = np.ones((2, 5), dtype=bool)
        base = np.full((2, 5), 0.4)
        refined = np.full((2, 5), 0.1)
        summary = summarize_rope_residuals(base, refined, valid)
        self.assertAlmostEqual(summary["base"]["mean_abs"], 0.4, places=6)
        self.assertAlmostEqual(summary["refined"]["mean_abs"], 0.1, places=6)
        self.assertAlmostEqual(summary["closure_frac"], 0.75, places=6)
        self.assertAlmostEqual(summary["frac_fingers_improved"], 1.0, places=6)
        self.assertEqual(summary["num_valid_fingers"], 10)
        self.assertIn("thumb", summary["per_finger"])

    def test_summary_ignores_invalid(self):
        valid = np.zeros((2, 5), dtype=bool)
        valid[0, 0] = True
        base = np.full((2, 5), 99.0)
        base[0, 0] = 0.5
        refined = np.full((2, 5), 99.0)
        refined[0, 0] = 0.25
        summary = summarize_rope_residuals(base, refined, valid)
        self.assertAlmostEqual(summary["base"]["mean_abs"], 0.5, places=6)
        self.assertAlmostEqual(summary["closure_frac"], 0.5, places=6)
        self.assertEqual(summary["num_valid_fingers"], 1)


class CorrelationTest(unittest.TestCase):
    def test_pearson_perfect(self):
        x = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson(x, 2.0 * x + 1.0), 1.0, places=6)
        self.assertAlmostEqual(pearson(x, -x), -1.0, places=6)

    def test_pearson_ignores_nan(self):
        x = np.asarray([1.0, 2.0, np.nan, 4.0])
        y = np.asarray([2.0, 4.0, 100.0, 8.0])
        self.assertAlmostEqual(pearson(x, y), 1.0, places=6)

    def test_pearson_degenerate(self):
        self.assertTrue(np.isnan(pearson(np.asarray([1.0]), np.asarray([2.0]))))
        self.assertTrue(np.isnan(pearson(np.ones(4), np.arange(4.0))))

    def test_spearman_monotonic(self):
        x = np.asarray([1.0, 2.0, 3.0, 4.0])
        y = np.exp(x)  # nonlinear but monotonic
        self.assertAlmostEqual(spearman(x, y), 1.0, places=6)

    def test_spearman_with_ties(self):
        x = np.asarray([1.0, 1.0, 2.0, 3.0])
        y = np.asarray([1.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(spearman(x, y), 1.0, places=6)


class BucketTest(unittest.TestCase):
    def test_quantile_buckets(self):
        values = np.arange(100.0)
        edges = quantile_bucket_edges(values, 4)
        self.assertEqual(len(edges), 3)
        idx = bucket_indices(values, edges)
        counts = [int((idx == b).sum()) for b in range(4)]
        for count in counts:
            self.assertGreaterEqual(count, 24)

    def test_nan_bucket_is_minus_one(self):
        edges = quantile_bucket_edges(np.arange(10.0), 2)
        idx = bucket_indices(np.asarray([np.nan, 1.0]), edges)
        self.assertEqual(idx[0], -1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            quantile_bucket_edges(np.asarray([np.nan]), 2)


if __name__ == "__main__":
    unittest.main()
