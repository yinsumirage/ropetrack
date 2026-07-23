import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from ropetrack.eval.scoring import align_w_scale


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "datasets" / "audit_egodex_quality.py"
    spec = importlib.util.spec_from_file_location("audit_egodex_quality", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EgoDexQualityAuditTest(unittest.TestCase):
    def test_pa_errors_match_existing_scorer(self):
        audit = load_script()
        rng = np.random.default_rng(7)
        gt = rng.normal(size=(3, 21, 3))
        pred = rng.normal(size=(3, 21, 3))
        expected = [np.linalg.norm(gt[i] - align_w_scale(gt[i], pred[i]), axis=1).mean() * 1000 for i in range(3)]

        np.testing.assert_allclose(audit.pa_errors_mm(gt, pred, batch_size=2), expected, atol=1e-8)

    def test_select_diverse_keeps_one_row_per_episode(self):
        audit = load_script()
        rows = [{"episode_id": episode} for episode in ("a", "a", "b", "c")]

        self.assertEqual(audit.select_diverse(np.asarray([0, 1, 2, 3]), rows, 3), [0, 2, 3])

    def test_confidence_bins_separate_missing_native_values(self):
        audit = load_script()
        errors = {"base": np.asarray([10.0, 20.0]), "student": np.asarray([9.0, 24.0])}
        result = audit.score_by_confidence(
            errors, np.asarray([0.1, 1.0]), np.asarray([True, False])
        )

        self.assertEqual(result["bins"][0]["count"], 1)
        self.assertEqual(result["bins"][-1]["tip_confidence"], "missing")
        self.assertEqual(result["bins"][-1]["student_delta_vs_base_mm"], 4.0)

    def test_confidence_bin_order_filters_and_prefers_bin_center(self):
        audit = load_script()
        values = np.asarray([0.1, 0.3, 0.49, 0.7])
        mask = (values >= 0.25) & (values < 0.5)

        np.testing.assert_array_equal(audit.confidence_bin_order(values, mask, 0.375), [1, 2])


if __name__ == "__main__":
    unittest.main()
