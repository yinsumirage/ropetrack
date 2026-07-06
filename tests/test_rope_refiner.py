import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from ropetrack.refine.cache import make_refiner_features, validate_cache
from ropetrack.refine.rope_refiner import RopePoseRefiner


ROOT = Path(__file__).resolve().parents[1]


def _tiny_cache(path: Path, n: int = 6) -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, 45)).astype(np.float32)
    base_rope = rng.normal(size=(n, 5)).astype(np.float32)
    input_rope = base_rope + 0.5
    gt_rope = base_rope + 2.0
    valid = np.ones((n, 5), dtype=np.float32)
    target = base + 0.25
    np.savez(
        path,
        sample_id=np.array([f"s{i}" for i in range(n)]),
        base_hand_pose=base,
        base_rope_norm=base_rope,
        input_rope_norm=input_rope.astype(np.float32),
        gt_rope_norm=gt_rope.astype(np.float32),
        rope_valid=valid,
        target_hand_pose=target.astype(np.float32),
    )


class RopeRefinerTest(unittest.TestCase):
    def test_rope_refiner_zero_init_identity(self):
        model = RopePoseRefiner(hidden_dim=16)
        base = torch.randn(3, 45)
        refined, _ = model(
            base,
            torch.randn(3, 5),
            torch.randn(3, 5),
            torch.ones(3, 5),
        )
        self.assertTrue(torch.allclose(refined, base))

    def test_refiner_features_default_to_input_rope_norm(self):
        cache = {
            "sample_id": np.array(["a"]),
            "base_hand_pose": np.zeros((1, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((1, 5), dtype=np.float32),
            "input_rope_norm": np.ones((1, 5), dtype=np.float32),
            "gt_rope_norm": np.full((1, 5), 9.0, dtype=np.float32),
            "rope_valid": np.ones((1, 5), dtype=np.float32),
        }
        arrays = make_refiner_features(cache)
        np.testing.assert_array_equal(arrays["input_rope_norm"], cache["input_rope_norm"])

    def test_cache_validation_catches_mismatched_lengths(self):
        cache = {
            "sample_id": np.array(["a", "b"]),
            "base_hand_pose": np.zeros((2, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((2, 5), dtype=np.float32),
            "input_rope_norm": np.zeros((1, 5), dtype=np.float32),
            "rope_valid": np.ones((2, 5), dtype=np.float32),
            "target_hand_pose": np.zeros((2, 45), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "first dimension"):
            validate_cache(cache)

    def test_cache_validation_allows_non_sample_metadata(self):
        cache = {
            "sample_id": np.array(["a", "b"]),
            "base_hand_pose": np.zeros((2, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((2, 5), dtype=np.float32),
            "input_rope_norm": np.zeros((2, 5), dtype=np.float32),
            "rope_valid": np.ones((2, 5), dtype=np.float32),
            "target_hand_pose": np.zeros((2, 45), dtype=np.float32),
            "finger_order": np.array(["thumb", "index", "middle", "ring", "pinky"]),
        }

        validate_cache(cache)

    def test_training_cache_validation_requires_target_hand_pose(self):
        cache = {
            "sample_id": np.array(["a"]),
            "base_hand_pose": np.zeros((1, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((1, 5), dtype=np.float32),
            "input_rope_norm": np.zeros((1, 5), dtype=np.float32),
            "rope_valid": np.ones((1, 5), dtype=np.float32),
            "gt_joints": np.zeros((1, 21, 3), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "target_hand_pose"):
            validate_cache(cache)

    def test_cache_validation_rejects_bad_target_shape(self):
        cache = {
            "sample_id": np.array(["a", "b"]),
            "base_hand_pose": np.zeros((2, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((2, 5), dtype=np.float32),
            "input_rope_norm": np.zeros((2, 5), dtype=np.float32),
            "rope_valid": np.ones((2, 5), dtype=np.float32),
            "target_hand_pose": np.zeros((2, 1), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "target_hand_pose shape"):
            validate_cache(cache)

    def test_refiner_features_zero_invalid_rope_nans(self):
        cache = {
            "sample_id": np.array(["a"]),
            "base_hand_pose": np.zeros((1, 45), dtype=np.float32),
            "base_rope_norm": np.array([[np.nan, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
            "input_rope_norm": np.array([[np.nan, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
            "rope_valid": np.array([[0.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        }
        arrays = make_refiner_features(cache)
        self.assertEqual(arrays["base_rope_norm"][0, 0], 0.0)
        self.assertEqual(arrays["input_rope_norm"][0, 0], 0.0)
        self.assertTrue(np.isfinite(arrays["base_rope_norm"]).all())
        self.assertTrue(np.isfinite(arrays["input_rope_norm"]).all())

    def test_refiner_features_reject_valid_rope_nans(self):
        cache = {
            "sample_id": np.array(["a"]),
            "base_hand_pose": np.zeros((1, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((1, 5), dtype=np.float32),
            "input_rope_norm": np.array([[np.nan, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            "rope_valid": np.ones((1, 5), dtype=np.float32),
        }
        with self.assertRaisesRegex(ValueError, "input_rope_norm has non-finite valid rope values"):
            make_refiner_features(cache)

    def test_train_cached_refiner_overfits_tiny_cache_and_writes_checkpoint(self):
        with self.subTest("subprocess"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                cache_path = tmp_path / "train.npz"
                ckpt_path = tmp_path / "refiner.pt"
                _tiny_cache(cache_path)

                subprocess.check_call(
                    [
                        sys.executable,
                        "scripts/rope_refiner/train_cached_refiner.py",
                        str(cache_path),
                        str(ckpt_path),
                        "--device",
                        "cpu",
                "--hidden-dim",
                "16",
                "--steps",
                "500",
                "--lr",
                "0.1",
                    ],
                    cwd=ROOT,
                )

                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                self.assertEqual(ckpt["config"]["rope_key"], "input_rope_norm")
                model = RopePoseRefiner(hidden_dim=ckpt["config"]["hidden_dim"])
                model.load_state_dict(ckpt["model_state"])
                with np.load(cache_path) as data:
                    with torch.no_grad():
                        refined, _ = model(
                            torch.from_numpy(data["base_hand_pose"]),
                            torch.from_numpy(data["base_rope_norm"]),
                            torch.from_numpy(data["input_rope_norm"]),
                            torch.from_numpy(data["rope_valid"]),
                        )
                    err = torch.mean(torch.abs(refined - torch.from_numpy(data["target_hand_pose"])))
                self.assertLess(err.item(), 0.05)

    def test_eval_cached_refiner_writes_outputs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "eval.npz"
            ckpt_path = tmp_path / "refiner.pt"
            out_dir = tmp_path / "eval_out"
            _tiny_cache(cache_path, n=3)
            with np.load(cache_path) as data:
                arrays = {key: data[key] for key in data.files}
            np.savez(cache_path, **arrays)
            model = RopePoseRefiner(hidden_dim=8)
            torch.save(
                {"model_state": model.state_dict(), "config": {"hidden_dim": 8, "rope_key": "gt_rope_norm"}},
                ckpt_path,
            )

            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/rope_refiner/eval_cached_refiner.py",
                    str(cache_path),
                    str(ckpt_path),
                    str(out_dir),
                    "--device",
                    "cpu",
                ],
                cwd=ROOT,
            )

            refined = np.load(out_dir / "refined_hand_pose.npy")
            sample_id = np.load(out_dir / "sample_id.npy")
            summary = json.loads((out_dir / "summary.json").read_text())
            self.assertEqual(refined.shape, (3, 45))
            np.testing.assert_array_equal(sample_id, np.array(["s0", "s1", "s2"]))
            self.assertEqual(summary["num_samples"], 3)
            self.assertEqual(summary["mean_abs_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
