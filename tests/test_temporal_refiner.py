import hashlib
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from ropetrack.refine.alpha_student import features_from_cache
from ropetrack.refine.temporal import (
    SequenceSplit,
    build_causal_windows,
    deterministic_sequence_split,
    sequence_frame,
    temporal_features,
)


def toy_cache(sample_ids, input_rope):
    num_rows = len(sample_ids)
    return {
        "sample_id": np.asarray(sample_ids),
        "base_hand_pose": np.zeros((num_rows, 45), dtype=np.float32),
        "base_rope_norm": np.zeros((num_rows, 5), dtype=np.float32),
        "input_rope_norm": np.asarray(input_rope, dtype=np.float64),
        "rope_valid": np.ones((num_rows, 5), dtype=bool),
    }


class TemporalProtocolTest(unittest.TestCase):
    def test_sequence_frame_normalizes_prefixes_and_backslashes(self):
        self.assertEqual(sequence_frame(r"teacher\AP10\0012"), ("AP10", 12))
        self.assertEqual(sequence_frame("teacher/AP11/0004"), ("AP11", 4))

    def test_sequence_frame_rejects_invalid_ids(self):
        for sample_id in ("0001", "A/", "/0001", "A//0001", "A/frame", "A/-1"):
            with self.subTest(sample_id=sample_id), self.assertRaises(ValueError):
                sequence_frame(sample_id)

    def test_sequence_split_dataclass_is_frozen(self):
        split = SequenceSplit(
            train_idx=np.asarray([0]),
            val_idx=np.asarray([1]),
            train_sequences=("A",),
            val_sequences=("B",),
        )
        with self.assertRaises(FrozenInstanceError):
            split.train_sequences = ("C",)

    def test_sequence_split_is_disjoint_and_stable(self):
        ids = np.array(
            [f"{seq}/{frame:04d}" for seq in ("A", "B", "C", "D", "E") for frame in range(2)]
        )
        a = deterministic_sequence_split(ids, val_fraction=0.2, seed=20260710)
        b = deterministic_sequence_split(ids, val_fraction=0.2, seed=20260710)
        self.assertEqual(a.train_sequences, b.train_sequences)
        self.assertTrue(set(a.train_sequences).isdisjoint(a.val_sequences))

        ordered = sorted(
            ("A", "B", "C", "D", "E"),
            key=lambda seq: (hashlib.sha256(f"20260710:{seq}".encode()).hexdigest(), seq),
        )
        self.assertEqual(a.val_sequences, tuple(ordered[:1]))
        self.assertEqual(a.train_sequences, tuple(ordered[1:]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([a.train_idx, a.val_idx])), np.arange(len(ids))
        )

    def test_sequence_split_rejects_invalid_fraction_and_empty_train(self):
        ids = np.asarray(["A/0000", "B/0000"])
        for fraction in (-0.1, 0.0, 1.0, 1.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                deterministic_sequence_split(ids, fraction, seed=0)
        with self.assertRaises(ValueError):
            deterministic_sequence_split(["A/0000"], 0.2, seed=0)
        with self.assertRaises(ValueError):
            deterministic_sequence_split([], 0.2, seed=0)

    def test_causal_windows_reset_on_sequence_and_gap(self):
        ids = np.array(["A/0000", "A/0004", "A/0012", "B/0000"])
        x = np.arange(8, dtype=np.float32).reshape(4, 2)
        windows, valid, lengths = build_causal_windows(
            ids, x, history_length=3, raw_frame_step=4, history_step=4
        )
        np.testing.assert_array_equal(valid[1], [True, True, False])
        np.testing.assert_array_equal(valid[2], [True, False, False])
        np.testing.assert_array_equal(valid[3], [True, False, False])
        np.testing.assert_array_equal(windows[1, :2], x[:2])
        np.testing.assert_array_equal(lengths, [1, 2, 1, 1])

    def test_future_changes_do_not_change_current_window(self):
        ids = np.array(["A/0000", "A/0004", "A/0008"])
        x = np.arange(6, dtype=np.float32).reshape(3, 2)
        before = build_causal_windows(ids, x, 3, 4, 4)[0][1].copy()
        x[2] = 999
        after = build_causal_windows(ids, x, 3, 4, 4)[0][1]
        np.testing.assert_array_equal(before, after)

    def test_dense_rows_can_use_sparse_history_without_becoming_gaps(self):
        ids = np.asarray([f"A/{i:04d}" for i in range(6)])
        x = np.arange(6, dtype=np.float32)[:, None]
        windows, valid, _ = build_causal_windows(
            ids, x, 3, raw_frame_step=1, history_step=2
        )
        np.testing.assert_array_equal(windows[4, :3, 0], [0, 2, 4])
        np.testing.assert_array_equal(valid[4], [True, True, True])

    def test_causal_windows_restore_unsorted_rows_and_output_types(self):
        ids = np.asarray(["A/0004", "B/0000", "A/0000", "A/0008"])
        x = np.asarray([[14], [100], [10], [18]], dtype=np.float32)
        windows, valid, lengths = build_causal_windows(ids, x, 3, 4, 4)

        self.assertEqual(windows.shape, (4, 3, 1))
        self.assertEqual(windows.dtype, np.float32)
        self.assertEqual(valid.dtype, np.bool_)
        self.assertEqual(lengths.dtype, np.int64)
        np.testing.assert_array_equal(windows[0, :, 0], [10, 14, 0])
        np.testing.assert_array_equal(windows[1, :, 0], [100, 0, 0])
        np.testing.assert_array_equal(windows[2, :, 0], [10, 0, 0])
        np.testing.assert_array_equal(windows[3, :, 0], [10, 14, 18])
        np.testing.assert_array_equal(valid[0], [True, True, False])
        np.testing.assert_array_equal(lengths, [2, 1, 1, 3])

    def test_causal_windows_reject_invalid_inputs(self):
        ids = np.asarray(["A/0000", "A/0004"])
        x = np.zeros((2, 1), dtype=np.float32)
        for history_length, raw_step, history_step in (
            (0, 4, 4),
            (3, 0, 4),
            (3, 4, 0),
            (3, 4, 6),
        ):
            with self.subTest(
                history_length=history_length,
                raw_step=raw_step,
                history_step=history_step,
            ), self.assertRaises(ValueError):
                build_causal_windows(ids, x, history_length, raw_step, history_step)
        with self.assertRaises(ValueError):
            build_causal_windows(ids, x[:1], 3, 4, 4)
        with self.assertRaises(ValueError):
            build_causal_windows(["A/0000", "A/0000"], x, 3, 4, 4)

    def test_temporal_features_use_rope_difference_without_crossing_gaps(self):
        ids = ["A/0004", "A/0000", "A/0012", "B/0000", "B/0004"]
        rope = np.asarray([[2] * 5, [1] * 5, [9] * 5, [3] * 5, [5] * 5])
        cache = toy_cache(ids, rope)

        features = temporal_features(cache, raw_frame_step=4)

        self.assertEqual(features.shape, (5, 70))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[:, :65], features_from_cache(cache))
        expected_delta = np.asarray([[1] * 5, [0] * 5, [0] * 5, [0] * 5, [2] * 5])
        np.testing.assert_array_equal(features[:, 65:], expected_delta)

    def test_temporal_features_reject_invalid_protocol_inputs(self):
        cache = toy_cache(["A/0000", "A/0004"], [[0] * 5, [1] * 5])
        with self.assertRaises(ValueError):
            temporal_features(cache, raw_frame_step=0)

        mismatch = dict(cache)
        mismatch["sample_id"] = mismatch["sample_id"][:1]
        with self.assertRaises(ValueError):
            temporal_features(mismatch, raw_frame_step=4)

        duplicate = dict(cache)
        duplicate["sample_id"] = np.asarray(["A/0000", "A/0000"])
        with self.assertRaises(ValueError):
            temporal_features(duplicate, raw_frame_step=4)


if __name__ == "__main__":
    unittest.main()
