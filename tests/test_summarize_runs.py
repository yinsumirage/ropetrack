import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "rope_refiner" / "summarize_runs.py"
    spec = importlib.util.spec_from_file_location("summarize_runs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_cell(root: Path, name: str, pa: float = 0.85, closure: float = 0.48, with_sliced: bool = True, with_scores: bool = True) -> Path:
    cell = root / name
    cell.mkdir(parents=True)
    summary = {
        "num_samples": 3960,
        "mode": "optimize",
        "objective": "rope",
        "action_space": "flex15",
        "rope_sensor": {"noise_std": 0.05, "dropout": 0.0, "seed": 0, "frac_valid_after": 1.0},
        "rope_residual": {"closure_frac": closure},
        "optimization": {"steps": 400, "lr": 32.0, "alpha_l2": 0.001, "max_alpha": 0.5, "gate_residual_threshold": 0.1},
        "alpha": {"mean_abs": 0.05, "max_abs": 0.4},
        "gating": {"threshold": 0.1, "frac_fingers_gated": 0.4},
    }
    (cell / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if with_sliced:
        sliced_dir = cell / "sliced"
        sliced_dir.mkdir()
        sliced = {
            "slices": {
                "all_joints": {"base_cm": 1.0068, "refined_cm": pa, "delta_cm": pa - 1.0068, "num_joint_obs": 83160},
                "occluded_fingertips": {"base_cm": 1.5, "refined_cm": 1.0, "delta_cm": -0.5, "num_joint_obs": 5000},
                "clean_fingertips": {"base_cm": 0.8, "refined_cm": 0.78, "delta_cm": -0.02, "num_joint_obs": 5000},
            }
        }
        (sliced_dir / "sliced_scores.json").write_text(json.dumps(sliced), encoding="utf-8")
    if with_scores:
        scores_dir = cell / "scores"
        scores_dir.mkdir()
        scores = {"xyz_procrustes_al_mean3d": pa, "mesh_al_mean3d": pa + 0.01, "f_al_score_5": 0.62}
        (scores_dir / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    return cell


class SummarizeRunsTest(unittest.TestCase):
    def test_aggregates_cells_recursively(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            make_cell(root, "mask70/rope_flex15_gate010", pa=0.8389)
            make_cell(root, "mask70/rope_mult5_gate010", pa=0.8520)
            out = Path(tmp) / "out"

            script.main([str(root), "--output-dir", str(out)])

            rows = json.loads((out / "runs_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            by_cell = {row["cell"]: row for row in rows}
            winner = by_cell["mask70/rope_flex15_gate010"]
            self.assertAlmostEqual(winner["pa_cm"], 0.8389, places=6)
            self.assertAlmostEqual(winner["closure"], 0.48, places=6)
            self.assertAlmostEqual(winner["all_joints_delta_cm"], 0.8389 - 1.0068, places=6)
            self.assertAlmostEqual(winner["occluded_tip_delta_cm"], -0.5, places=6)
            self.assertAlmostEqual(winner["gate_threshold"], 0.1, places=6)
            self.assertAlmostEqual(winner["noise_std"], 0.05, places=6)

            tsv = (out / "runs_summary.tsv").read_text(encoding="utf-8")
            self.assertEqual(len(tsv.strip().splitlines()), 3)  # header + 2 rows
            md = (out / "runs_summary.md").read_text(encoding="utf-8")
            self.assertIn("rope_flex15_gate010", md)

    def test_missing_optional_files_yield_dashes(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            make_cell(root, "bare_cell", with_sliced=False, with_scores=False)
            out = Path(tmp) / "out"
            script.main([str(root), "--output-dir", str(out)])
            rows = json.loads((out / "runs_summary.json").read_text(encoding="utf-8"))
            self.assertIsNone(rows[0]["pa_cm"])
            self.assertIsNone(rows[0]["all_joints_delta_cm"])
            tsv = (out / "runs_summary.tsv").read_text(encoding="utf-8")
            self.assertIn("-", tsv)

    def test_sort_by_metric(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            make_cell(root, "b_cell", pa=0.90)
            make_cell(root, "a_cell", pa=0.80)
            out = Path(tmp) / "out"
            script.main([str(root), "--output-dir", str(out), "--sort-by", "pa_cm"])
            rows = json.loads((out / "runs_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["cell"], "a_cell")

    def test_no_cells_raises(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                script.main([str(Path(tmp)), "--output-dir", str(Path(tmp) / "out")])

    def test_missing_root_raises(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                script.main([str(Path(tmp) / "nope"), "--output-dir", str(Path(tmp) / "out")])


if __name__ == "__main__":
    unittest.main()
