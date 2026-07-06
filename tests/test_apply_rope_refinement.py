import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_apply_script():
    path = ROOT / "scripts" / "rope_refiner" / "apply_rope_refinement.py"
    spec = importlib.util.spec_from_file_location("apply_rope_refinement", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def freihand_joints(tip_x: float = 4.0):
    joints = [[0.0, 0.0, 1.0] for _ in range(21)]
    for joint_id in range(1, 5):
        joints[joint_id] = [float(joint_id), 0.0, 1.0]
    joints[4] = [tip_x, 0.0, 1.0]
    return joints


def ho3d_joints(tip_x: float = 4.0):
    joints = [[0.0, 0.0, 1.0] for _ in range(21)]
    for joint_id in (13, 14, 15):
        joints[joint_id] = [1.0, 0.0, 1.0]
    joints[16] = [tip_x, 0.0, 1.0]
    return joints


class ApplyRopeRefinementTest(unittest.TestCase):
    def test_apply_finger_curl_alpha_only_changes_selected_finger_groups(self):
        script = load_apply_script()
        base = np.arange(45, dtype=np.float32).reshape(1, 45)
        alpha = np.asarray([[0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        refined = script.apply_finger_curl_alpha(base, alpha)

        changed = np.flatnonzero(np.abs(refined[0] - base[0]) > 1e-6)
        expected = np.asarray([36, 37, 38, 39, 40, 41, 42, 43, 44])
        np.testing.assert_array_equal(changed, expected)

    def test_aa_to_rotmat_shapes_global_and_hand_pose(self):
        script = load_apply_script()

        global_rot = script.aa_to_rotmat(np.zeros((2, 3), dtype=np.float32))
        hand_rot = script.aa_to_rotmat(np.zeros((2, 15, 3), dtype=np.float32))

        self.assertEqual(global_rot.shape, (2, 3, 3))
        self.assertEqual(hand_rot.shape, (2, 15, 3, 3))
        np.testing.assert_allclose(global_rot[0], np.eye(3), atol=1e-6)

    def test_build_inference_cache_uses_run_meta_order_and_has_no_target_pose(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            pred_dir.mkdir()
            rope_labels = root / "rope.jsonl"
            run_meta = root / "run_meta.json"
            mano_cache = root / "mano_cache.npz"
            out = root / "cache.npz"

            rows = []
            for sid in ("00000000", "00000001"):
                rows.append({
                    "sample_id": sid,
                    "dataset": "freihand",
                    "rope_norm": [1.0, None, None, None, None],
                    "rope_chain_m": [4.0, None, None, None, None],
                    "rope_valid": [True, False, False, False, False],
                    "normalization": {"fist_ratio": 0.5},
                })
            rope_labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (pred_dir / "pred.json").write_text(json.dumps([
                [freihand_joints(3.0), freihand_joints(2.0)],
                [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
            ]), encoding="utf-8")
            run_meta.write_text(json.dumps({"sample_order": ["00000001", "00000000"]}), encoding="utf-8")
            np.savez(
                mano_cache,
                sample_id=np.asarray(["00000000", "00000001"]),
                base_hand_pose=np.stack([
                    np.full(45, 7.0, dtype=np.float32),
                    np.full(45, 9.0, dtype=np.float32),
                ]),
            )

            script.build_inference_cache("freihand", rope_labels, pred_dir, run_meta, mano_cache, out)

            with np.load(out) as data:
                self.assertEqual(data["sample_id"].tolist(), ["00000001", "00000000"])
                self.assertNotIn("target_hand_pose", data.files)
                np.testing.assert_array_equal(data["base_hand_pose"][0], np.full(45, 9.0, dtype=np.float32))
                self.assertEqual(data["base_rope_norm"].shape, (2, 5))
                self.assertEqual(data["input_rope_norm"].shape, (2, 5))

    def test_build_inference_cache_uses_ho3d_rope_chain(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            pred_dir.mkdir()
            rope_labels = root / "rope.jsonl"
            run_meta = root / "run_meta.json"
            mano_cache = root / "mano_cache.npz"
            out = root / "cache.npz"

            row = {
                "sample_id": "SM1/0000",
                "dataset": "ho3d",
                "rope_norm": [1.0, None, None, None, None],
                "rope_chain_m": [4.0, None, None, None, None],
                "rope_valid": [True, False, False, False, False],
                "normalization": {"fist_ratio": 0.5},
            }
            rope_labels.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (pred_dir / "pred.json").write_text(json.dumps([
                [ho3d_joints(4.0)],
                [[[0.0, 0.0, 0.0]]],
            ]), encoding="utf-8")
            run_meta.write_text(json.dumps({"sample_order": ["SM1/0000"]}), encoding="utf-8")
            np.savez(
                mano_cache,
                sample_id=np.asarray(["SM1/0000"]),
                base_hand_pose=np.zeros((1, 45), dtype=np.float32),
            )

            script.build_inference_cache("ho3d", rope_labels, pred_dir, run_meta, mano_cache, out)

            with np.load(out) as data:
                self.assertAlmostEqual(float(data["base_rope_norm"][0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
