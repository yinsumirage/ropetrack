import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


def load_script(name: str):
    rel_paths = {
        "score_rope_predictions": Path("rope_diagnostics") / "score_rope_predictions.py",
    }
    path = Path(__file__).resolve().parents[1] / "scripts" / rel_paths.get(name, Path(f"{name}.py"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RopeTest(unittest.TestCase):
    def test_freihand_rope_norm_uses_wrist_tip_and_chain_length(self):
        from ropetrack.rope import build_rope_row

        joints = [[0.0, 0.0, 0.0] for _ in range(21)]
        joints[1] = [1.0, 0.0, 0.0]
        joints[2] = [2.0, 0.0, 0.0]
        joints[3] = [3.0, 0.0, 0.0]
        joints[4] = [4.0, 0.0, 0.0]

        row = build_rope_row("freihand", "00000000", joints, fist_ratio=0.5)

        self.assertEqual(row["finger_order"], ["thumb", "index", "middle", "ring", "pinky"])
        self.assertAlmostEqual(row["rope_dist_m"][0], 4.0)
        self.assertAlmostEqual(row["rope_chain_m"][0], 4.0)
        self.assertAlmostEqual(row["rope_norm"][0], 1.0)

    def test_ho3d_rope_uses_dataset_specific_chain_order(self):
        from ropetrack.rope import build_rope_row

        joints = [[0.0, 0.0, 0.0] for _ in range(21)]
        joints[1] = [1.0, 0.0, 0.0]
        joints[2] = [2.0, 0.0, 0.0]
        joints[3] = [3.0, 0.0, 0.0]
        joints[17] = [4.0, 0.0, 0.0]

        row = build_rope_row("ho3d", "AP10/0000", joints, fist_ratio=0.5)

        self.assertAlmostEqual(row["rope_dist_m"][1], 4.0)
        self.assertAlmostEqual(row["rope_chain_m"][1], 4.0)
        self.assertAlmostEqual(row["rope_norm"][1], 1.0)

    def test_invalid_finger_is_marked_invalid(self):
        from ropetrack.rope import build_rope_row

        joints = [[0.0, 0.0, 0.0] for _ in range(21)]
        joints[4] = [float("nan"), 0.0, 0.0]

        row = build_rope_row("freihand", "00000000", joints)

        self.assertFalse(row["rope_valid"][0])
        self.assertIsNone(row["rope_dist_m"][0])
        self.assertIsNone(row["rope_norm"][0])

    def test_make_rope_labels_writes_rows_for_freihand_eval_root(self):
        maker = load_script("make_rope_labels")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "freihand"
            out = Path(tmp) / "rope.jsonl"
            (root / "evaluation" / "rgb").mkdir(parents=True)
            joints = [[0.0, 0.0, 0.0] for _ in range(21)]
            joints[1] = [1.0, 0.0, 0.0]
            joints[2] = [2.0, 0.0, 0.0]
            joints[3] = [3.0, 0.0, 0.0]
            joints[4] = [4.0, 0.0, 0.0]
            (root / "evaluation_xyz.json").write_text(json.dumps([joints, joints]))

            maker.write_rope_labels("freihand", root, out, limit=1)

            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "00000000")
            self.assertAlmostEqual(rows[0]["rope_norm"][0], 1.0)

    def test_make_rope_labels_can_write_visualization_png(self):
        maker = load_script("make_rope_labels")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "freihand"
            out = Path(tmp) / "rope.jsonl"
            viz = Path(tmp) / "viz"
            (root / "evaluation" / "rgb").mkdir(parents=True)
            Image.new("RGB", (32, 32), (255, 255, 255)).save(root / "evaluation" / "rgb" / "00000000.jpg")
            joints = [[0.0, 0.0, 1.0] for _ in range(21)]
            joints[1] = [0.1, 0.0, 1.0]
            joints[2] = [0.2, 0.0, 1.0]
            joints[3] = [0.3, 0.0, 1.0]
            joints[4] = [0.4, 0.0, 1.0]
            K = [[30.0, 0.0, 2.0], [0.0, 30.0, 2.0], [0.0, 0.0, 1.0]]
            (root / "evaluation_xyz.json").write_text(json.dumps([joints]))
            (root / "evaluation_K.json").write_text(json.dumps([K]))

            maker.write_rope_labels("freihand", root, out, limit=1, viz_dir=viz, viz_count=1)

            self.assertTrue((viz / "00000000.png").exists())

    def test_make_rope_labels_maps_ho3d_run_meta_order_to_gt_indices(self):
        maker = load_script("make_rope_labels")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ho3d"
            out = Path(tmp) / "rope.jsonl"
            order = Path(tmp) / "run_meta.json"
            rgb = root / "evaluation" / "AP10" / "rgb"
            meta = root / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            (rgb / "0000.png").write_bytes(b"")
            (rgb / "0001.png").write_bytes(b"")
            for frame in ("0000", "0001"):
                with (meta / f"{frame}.pkl").open("wb") as f:
                    pickle.dump({}, f)
            invalid = [[0.0, 0.0, 0.0] for _ in range(21)]
            valid = [[0.0, 0.0, 0.0] for _ in range(21)]
            valid[13] = [1.0, 0.0, 0.0]
            valid[14] = [2.0, 0.0, 0.0]
            valid[15] = [3.0, 0.0, 0.0]
            valid[16] = [4.0, 0.0, 0.0]
            (root / "evaluation.txt").write_text("AP10/0000\nAP10/0001\n")
            (root / "evaluation_xyz.json").write_text(json.dumps([invalid, valid]))
            order.write_text(json.dumps({"sample_order": ["AP10/0001"]}))

            maker.write_rope_labels("ho3d", root, out, sample_order_file=order)

            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "AP10/0001")
            self.assertAlmostEqual(rows[0]["rope_norm"][0], 1.0)

    def test_score_rope_predictions_reports_mean_absolute_norm_error(self):
        scorer = load_script("score_rope_predictions")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            out_dir = root / "scores"
            pred_dir.mkdir()
            gt_joints = [[0.0, 0.0, 0.0] for _ in range(21)]
            pred_joints = [[0.0, 0.0, 0.0] for _ in range(21)]
            for joint_id, x in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)):
                gt_joints[joint_id] = [x, 0.0, 0.0]
            for joint_id, x in ((1, 1.0), (2, 2.0), (3, 3.0), (4, 3.0)):
                pred_joints[joint_id] = [x, 0.0, 0.0]
            gt_row = {
                "sample_id": "00000000",
                "dataset": "freihand",
                "rope_norm": [1.0, None, None, None, None],
                "rope_dist_m": [4.0, None, None, None, None],
                "rope_chain_m": [4.0, None, None, None, None],
                "rope_valid": [True, False, False, False, False],
                "normalization": {"mode": "chain_length_fist_ratio", "fist_ratio": 0.5},
            }
            (root / "rope.jsonl").write_text(json.dumps(gt_row) + "\n")
            (pred_dir / "pred.json").write_text(json.dumps([[pred_joints], [[[0.0, 0.0, 0.0]]]]))

            scores = scorer.score_rope_predictions(pred_dir, root / "rope.jsonl", out_dir, dataset="freihand")

            self.assertAlmostEqual(scores["rope_norm_mae"], 0.5)
            self.assertEqual(scores["num_samples"], 1)
            self.assertTrue((out_dir / "rope_errors.jsonl").exists())

    def test_score_rope_predictions_counts_collapsed_prediction_as_error(self):
        scorer = load_script("score_rope_predictions")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            out_dir = root / "scores"
            pred_dir.mkdir()
            collapsed = [[0.0, 0.0, 0.0] for _ in range(21)]
            gt_row = {
                "sample_id": "00000000",
                "dataset": "freihand",
                "rope_norm": [1.0, None, None, None, None],
                "rope_dist_m": [4.0, None, None, None, None],
                "rope_chain_m": [4.0, None, None, None, None],
                "rope_valid": [True, False, False, False, False],
                "normalization": {"mode": "chain_length_fist_ratio", "fist_ratio": 0.5},
            }
            (root / "rope.jsonl").write_text(json.dumps(gt_row) + "\n")
            (pred_dir / "pred.json").write_text(json.dumps([[collapsed], [[[0.0, 0.0, 0.0]]]]))

            scores = scorer.score_rope_predictions(pred_dir, root / "rope.jsonl", out_dir, dataset="freihand")

            self.assertEqual(scores["num_valid_fingers"], 1)
            self.assertAlmostEqual(scores["rope_norm_mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
