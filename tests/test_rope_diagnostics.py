import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_analyzer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rope_diagnostics" / "analyze_rope_errors.py"
    spec = importlib.util.spec_from_file_location("analyze_rope_errors", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RopeDiagnosticsTest(unittest.TestCase):
    def test_parse_run_name_handles_clean_and_hard_runs(self):
        analyzer = load_analyzer()

        self.assertEqual(
            analyzer.parse_run_name("clean_ho3d_v2_wilor"),
            {"split": "clean", "dataset": "ho3d_v2", "hard_kind": "clean", "backend": "wilor"},
        )
        self.assertEqual(
            analyzer.parse_run_name("hard_freihand_finger_end80_hamer"),
            {"split": "hard", "dataset": "freihand", "hard_kind": "finger_end80", "backend": "hamer"},
        )

    def test_summarize_run_reports_mae_bias_bins_and_worst_rows(self):
        analyzer = load_analyzer()
        rows = [
            {
                "sample_id": "a",
                "finger_order": ["thumb", "index"],
                "gt_rope_norm": [1.0, 0.2],
                "pred_rope_norm": [0.8, 0.3],
                "rope_norm_abs_error": [0.2, 0.1],
                "rope_norm_mae": 0.15,
            },
            {
                "sample_id": "b",
                "finger_order": ["thumb", "index"],
                "gt_rope_norm": [0.6, None],
                "pred_rope_norm": [0.2, None],
                "rope_norm_abs_error": [0.4, None],
                "rope_norm_mae": 0.4,
            },
        ]

        summary, per_finger, bins, worst = analyzer.summarize_run("hard_freihand_mask70_wilor", rows, top_k=1)

        self.assertAlmostEqual(summary["rope_norm_mae"], (0.2 + 0.1 + 0.4) / 3)
        self.assertAlmostEqual(summary["rope_norm_bias"], (-0.2 + 0.1 - 0.4) / 3)
        self.assertEqual(per_finger[0]["finger"], "thumb")
        self.assertAlmostEqual(per_finger[0]["mae"], 0.3)
        self.assertEqual(bins[0]["gt_bin"], "closed")
        self.assertEqual(worst[0]["sample_id"], "b")

    def test_write_analysis_outputs_delta_against_matching_clean_run(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / "scores"
            clean = scores / "clean_freihand_wilor"
            hard = scores / "hard_freihand_mask70_wilor"
            clean.mkdir(parents=True)
            hard.mkdir(parents=True)
            clean_row = {
                "sample_id": "a",
                "finger_order": ["thumb"],
                "gt_rope_norm": [1.0],
                "pred_rope_norm": [0.9],
                "rope_norm_abs_error": [0.1],
                "rope_norm_mae": 0.1,
            }
            hard_row = {
                "sample_id": "a",
                "finger_order": ["thumb"],
                "gt_rope_norm": [1.0],
                "pred_rope_norm": [0.5],
                "rope_norm_abs_error": [0.5],
                "rope_norm_mae": 0.5,
            }
            (clean / "rope_errors.jsonl").write_text(json.dumps(clean_row) + "\n")
            (hard / "rope_errors.jsonl").write_text(json.dumps(hard_row) + "\n")

            result = analyzer.write_analysis(scores, root / "analysis", top_k=3, make_plots=False)

            self.assertEqual(result["num_runs"], 2)
            delta = (root / "analysis" / "hard_clean_delta.tsv").read_text()
            self.assertIn("hard_freihand_mask70_wilor", delta)
            self.assertIn("0.4", delta)

    def test_write_analysis_can_make_plots(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores = root / "scores"
            run = scores / "clean_freihand_wilor"
            run.mkdir(parents=True)
            row = {
                "sample_id": "a",
                "finger_order": ["thumb"],
                "gt_rope_norm": [1.0],
                "pred_rope_norm": [0.8],
                "rope_norm_abs_error": [0.2],
                "rope_norm_mae": 0.2,
            }
            (run / "rope_errors.jsonl").write_text(json.dumps(row) + "\n")

            analyzer.write_analysis(scores, root / "analysis", top_k=1, make_plots=True)

            self.assertTrue((root / "analysis" / "figures" / "scatter_clean_freihand_wilor.png").exists())


if __name__ == "__main__":
    unittest.main()
