import numpy as np
import torch

from scripts.rope_refiner.direct_global_orient_head import (
    OrientationHead,
    root_metrics,
    shuffle_rope_within_participant,
)


def test_orientation_heads_start_as_identity_delta():
    batch = {
        "global_orient": torch.randn(2, 3),
        "base_hand_pose": torch.randn(2, 45),
        "base_rope_norm": torch.rand(2, 5),
        "input_rope_norm": torch.rand(2, 5),
        "rope_valid": torch.ones(2, 5),
        "tokens": torch.randn(2, 12, 8),
    }
    for mode in ("rgb", "rope", "rgb_rope"):
        torch.testing.assert_close(OrientationHead(mode, 8)(batch), torch.zeros(2, 3))


def test_root_metrics_are_translation_invariant():
    truth = torch.randn(3, 21, 3)
    shifted = truth + torch.tensor([2.0, -3.0, 4.0])
    loss, mm = root_metrics(shifted, truth)
    assert float(loss) < 1e-6
    assert float(mm) < 1e-3


def test_rope_shuffle_is_deterministic_and_stays_within_participant():
    rope = np.arange(30, dtype=np.float32).reshape(6, 5)
    arrays = {
        "sample_id": np.asarray(["P1_a", "P1_b", "P1_c", "P2_a", "P2_b", "P2_c"]),
        "input_rope_norm": rope,
        "rope_valid": rope + 100,
        "base_rope_norm": rope + 200,
    }
    first = shuffle_rope_within_participant(arrays, 42)
    second = shuffle_rope_within_participant(arrays, 42)
    np.testing.assert_array_equal(first["input_rope_norm"], second["input_rope_norm"])
    np.testing.assert_array_equal(first["base_rope_norm"], arrays["base_rope_norm"])
    assert first["base_rope_norm"] is arrays["base_rope_norm"]
    for rows in (slice(0, 3), slice(3, 6)):
        assert sorted(first["input_rope_norm"][rows, 0]) == sorted(rope[rows, 0])
        np.testing.assert_array_equal(first["rope_valid"][rows] - first["input_rope_norm"][rows], 100)
