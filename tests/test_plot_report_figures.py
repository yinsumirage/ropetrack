import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "rope_refiner" / "plot_report_figures.py"
    spec = importlib.util.spec_from_file_location("plot_report_figures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summary_rows():
    return [
        {"cell": "sweep/lr2_s120", "closure": 0.038, "all_joints_delta_cm": -0.0162, "noise_std": None, "dropout": None},
        {"cell": "sweep/lr8_s400", "closure": 0.278, "all_joints_delta_cm": -0.1002, "noise_std": None, "dropout": None},
        {"cell": "sweep/lr32_s400", "closure": 0.418, "all_joints_delta_cm": -0.1398, "noise_std": None, "dropout": None},
        {"cell": "noise/n005_d0", "closure": 0.42, "all_joints_delta_cm": -0.1384, "noise_std": 0.05, "dropout": 0.0},
        {"cell": "noise/n010_d0", "closure": 0.49, "all_joints_delta_cm": -0.0778, "noise_std": 0.10, "dropout": 0.0},
        {"cell": "noise/n005_d02", "closure": 0.42, "all_joints_delta_cm": -0.1008, "noise_std": 0.05, "dropout": 0.2},
        {"cell": "bias_std_0p05", "mode": "optimize", "bias_std": "0.050000", "retention_vs_clean": 0.82},
        {"cell": "bias_std_0p05", "mode": "student", "bias_std": "0.050000", "retention_vs_clean": 0.94},
        {"cell": "bias_std_0p10", "mode": "optimize", "bias_std": "0.100000", "retention_vs_clean": 0.46},
        {"cell": "bias_std_0p10", "mode": "student", "bias_std": "0.100000", "retention_vs_clean": 0.75},
        {"cell": "rope_mult5_gate010", "objective": "rope", "dim": 5, "gain_mm": 1.55},
        {"cell": "rope_flex15_gate010", "objective": "rope", "dim": 15, "gain_mm": 1.68},
        {"cell": "rope_pose45_gate010", "objective": "rope", "dim": 45, "gain_mm": 1.77},
        {"cell": "oracle_tip_mult5", "objective": "oracle_tip", "dim": 5, "gain_mm": 1.83},
        {"cell": "oracle_tip_flex15", "objective": "oracle_tip", "dim": 15, "gain_mm": 1.97},
        {"cell": "oracle_tip_pose45", "objective": "oracle_tip", "dim": 45, "gain_mm": 3.15},
        {"cell": "oracle_chain_mult5", "objective": "oracle_chain", "dim": 5, "gain_mm": 1.59},
        {"cell": "oracle_chain_flex15", "objective": "oracle_chain", "dim": 15, "gain_mm": 1.83},
        {"cell": "oracle_chain_pose45", "objective": "oracle_chain", "dim": 45, "gain_mm": 3.39},
    ]


class PlotReportFiguresTest(unittest.TestCase):
    def _write_summary(self, tmp: Path) -> Path:
        path = tmp / "runs_summary.json"
        path.write_text(json.dumps(summary_rows()), encoding="utf-8")
        return path

    def test_dose_response_figure(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            out = Path(tmp) / "figs" / "dose.png"
            script.main([
                "--summary", str(summary), "--figure", "dose_response",
                "--cell-filter", "sweep/", "--output", str(out),
            ])
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_noise_figure(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            out = Path(tmp) / "noise.png"
            script.main([
                "--summary", str(summary), "--figure", "noise",
                "--cell-filter", "noise/", "--output", str(out),
            ])
            self.assertTrue(out.exists())

    def test_empty_filter_matches_all(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            out = Path(tmp) / "all.png"
            script.main(["--summary", str(summary), "--figure", "dose_response", "--output", str(out)])
            self.assertTrue(out.exists())

    def test_no_match_raises(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            with self.assertRaises(ValueError):
                script.main([
                    "--summary", str(summary), "--figure", "noise",
                    "--cell-filter", "does_not_exist", "--output", str(Path(tmp) / "x.png"),
                ])

    def test_noise_figure_requires_noise_rows(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            with self.assertRaises(ValueError):
                script.main([
                    "--summary", str(summary), "--figure", "noise",
                    "--cell-filter", "sweep/", "--output", str(Path(tmp) / "x.png"),
                ])

    def test_e1_calibration_figure(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            out = Path(tmp) / "e1.png"
            script.main([
                "--summary", str(summary), "--figure", "e1_calibration",
                "--cell-filter", "bias_std_", "--output", str(out),
            ])
            self.assertTrue(out.exists())

    def test_e2_scissors_figure(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._write_summary(Path(tmp))
            out = Path(tmp) / "e2.png"
            script.main([
                "--summary", str(summary), "--figure", "e2_scissors",
                "--output", str(out),
            ])
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
