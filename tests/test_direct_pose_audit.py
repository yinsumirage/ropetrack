import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ropetrack.refine.direct_pose_audit import (
    ExactFallbackDirectPoseHead,
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


if __name__ == "__main__":
    unittest.main()
