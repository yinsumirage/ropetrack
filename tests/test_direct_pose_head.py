import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "rope_refiner" / "direct_pose_head.py"
    spec = importlib.util.spec_from_file_location("direct_pose_head", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DirectPoseHeadTest(unittest.TestCase):
    def test_zero_init_is_identity_and_tokens_are_strict(self):
        script = load_script()
        batch = 3
        pose = torch.randn(batch, 45)
        rope = torch.rand(batch, 5)
        valid = torch.ones(batch, 5)
        plain = script.DirectPoseHead()
        torch.testing.assert_close(plain(pose, rope, rope, valid), pose)
        with self.assertRaises(ValueError):
            plain(pose, rope, rope, valid, torch.randn(batch, 12, 8))
        token = script.DirectPoseHead(token_dim=8)
        torch.testing.assert_close(token(pose, rope, rope, valid, torch.randn(batch, 12, 8)), pose)

    def test_pa_alignment_removes_similarity_transform(self):
        script = load_script()
        gt = torch.randn(4, 21, 3)
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pred = 1.7 * (gt @ rotation.T) + torch.tensor([2.0, -3.0, 0.5])
        aligned = script.pa_align(pred, gt)
        torch.testing.assert_close(aligned, gt, atol=2e-5, rtol=2e-5)

    def test_episode_split_never_splits_parent_episode(self):
        script = load_script()
        ids = np.asarray([f"s01/a/right/{i:05d}" for i in range(4)] + [f"s02/b/left/{i:05d}" for i in range(4)])
        train, val = script.episode_split(ids, 0.5, 3)
        train_parents = {sid.rsplit("/", 1)[0] for sid in ids[train]}
        val_parents = {sid.rsplit("/", 1)[0] for sid in ids[val]}
        self.assertFalse(train_parents & val_parents)

    def test_shuffle_stays_inside_declared_groups(self):
        script = load_script()
        arrays = {
            "input_rope_norm": np.arange(30, dtype=np.float32).reshape(6, 5),
            "base_rope_norm": np.zeros((6, 5), dtype=np.float32),
            "rope_valid": np.ones((6, 5), dtype=bool),
        }
        original = arrays["input_rope_norm"].copy()
        script.apply_rope_mode(arrays, "shuffle", 1, (np.arange(3), np.arange(3, 6)))
        self.assertEqual(set(arrays["input_rope_norm"][:3, 0]), set(original[:3, 0]))
        self.assertEqual(set(arrays["input_rope_norm"][3:, 0]), set(original[3:, 0]))

    def test_append_bundle_is_strict_and_preserves_rows(self):
        script = load_script()
        arrays = {
            "sample_id": np.asarray(["arctic/a/0"]),
            "value": np.asarray([[1.0, 2.0]], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.npz"
            np.savez(good, sample_id=np.asarray(["hot3d/b/0"]), value=np.asarray([[3.0, 4.0]], dtype=np.float32))
            merged = script.append_bundles(arrays, [good])
            self.assertEqual(merged["sample_id"].tolist(), ["arctic/a/0", "hot3d/b/0"])
            np.testing.assert_array_equal(merged["value"], [[1.0, 2.0], [3.0, 4.0]])
            overlap = Path(tmp) / "overlap.npz"
            np.savez(overlap, sample_id=np.asarray(["arctic/a/0"]), value=np.asarray([[5.0, 6.0]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "overlap"):
                script.append_bundles({key: value[:1].copy() for key, value in merged.items()}, [overlap])


if __name__ == "__main__":
    unittest.main()
