import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ropetrack.refine.direct_pose_audit import (
    ExactFallbackDirectPoseHead,
    _matched_rows,
    _parameter_group,
    _scenario_batch,
    select_update_probe_batches,
)


class DirectPoseAuditTest(unittest.TestCase):
    def test_exact_fallback_all_valid_single_invalid_and_all_invalid(self):
        torch.manual_seed(4)
        raw = ExactFallbackDirectPoseHead(token_dim=8)
        for parameter in raw.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
        legacy = type(raw).__mro__[1](token_dim=8)
        legacy.load_state_dict(raw.state_dict())
        pose = torch.randn(3, 45)
        rope = torch.rand(3, 5)
        measured = torch.rand(3, 5)
        tokens = torch.randn(3, 12, 8)
        valid = torch.ones(3, 5)
        legacy_output = legacy(pose, rope, measured, valid, tokens)
        torch.testing.assert_close(raw(pose, rope, measured, valid, tokens), legacy_output, rtol=0, atol=0)
        single = valid.clone()
        single[:, 2] = 0
        single_output = raw(pose, rope, measured, single, tokens)
        torch.testing.assert_close(single_output[:, raw.pose_dims[2]], pose[:, raw.pose_dims[2]], rtol=0, atol=0)
        keep = [0, 1, 3, 4]
        torch.testing.assert_close(single_output[:, raw.pose_dims[keep]], legacy_output[:, raw.pose_dims[keep]], rtol=0, atol=0)
        torch.testing.assert_close(raw(pose, rope, measured, torch.zeros_like(valid), tokens), pose, rtol=0, atol=0)

    def test_batch_selection_is_deterministic_and_episode_disjoint(self):
        ids = np.asarray([f"s{episode}/frame{frame}" for episode in range(8) for frame in range(6)])
        episodes = np.asarray([f"s{episode}" for episode in range(8) for _ in range(6)])
        first = select_update_probe_batches(ids, episodes, batch_size=3, num_batches=4, seed=9)
        second = select_update_probe_batches(ids, episodes, batch_size=3, num_batches=4, seed=9)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertFalse(set(ids[first[0]].reshape(-1)) & set(ids[first[1]].reshape(-1)))
        self.assertFalse(set(episodes[first[0]].reshape(-1)) & set(episodes[first[1]].reshape(-1)))

    def test_attribution_helpers_are_deterministic_and_unique(self):
        arrays = {
            "base_hand_pose": np.zeros((4, 45), dtype=np.float32),
            "base_rope_norm": np.zeros((4, 5), dtype=np.float32),
            "input_rope_norm": np.arange(20, dtype=np.float32).reshape(4, 5),
            "rope_valid": np.ones((4, 5), dtype=np.float32),
        }
        first = _scenario_batch(arrays, np.arange(4), "cpu", "shuffle", 7)
        second = _scenario_batch(arrays, np.arange(4), "cpu", "shuffle", 7)
        self.assertTrue(torch.equal(first["input_rope_norm"], second["input_rope_norm"]))
        self.assertFalse(torch.equal(first["input_rope_norm"], torch.from_numpy(arrays["input_rope_norm"])))

        selected = {"arrays": arrays, "update": np.arange(4).reshape(2, 2)}
        left, right, distance = _matched_rows(selected, selected, 3)
        self.assertEqual(len(set(left.tolist())), 4)
        self.assertEqual(len(set(right.tolist())), 4)
        self.assertTrue(np.allclose(distance, 0.0))
        self.assertEqual(_parameter_group("query.0.weight"), "condition_query")
        self.assertEqual(_parameter_group("attention.in_proj_weight"), "rgb_attention")
        self.assertEqual(_parameter_group("output.2.bias"), "residual_output")


if __name__ == "__main__":
    unittest.main()
