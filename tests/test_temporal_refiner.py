import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from ropetrack.refine import alpha_student, temporal
from ropetrack.refine.alpha_student import RopeAlphaStudent, features_from_cache
from ropetrack.refine.temporal import (
    SequenceSplit,
    build_causal_windows,
    deterministic_sequence_split,
    episode_schedule,
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


def load_temporal_trainer():
    path = Path(__file__).resolve().parents[1] / "scripts" / "rope_refiner" / "train_temporal_student.py"
    spec = importlib.util.spec_from_file_location("train_temporal_student", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    def test_episode_schedule_resets_at_gap_and_sequence(self):
        ids = (
            [f"A/{frame:04d}" for frame in range(120)]
            + ["A/0200", "A/0201"]
            + [f"B/{frame:04d}" for frame in range(120)]
        )

        schedule = episode_schedule(
            ids, context=30, masked=60, recovery=30, raw_frame_step=1
        )

        self.assertEqual(schedule[0].phase, "context")
        self.assertEqual(schedule[30].phase, "masked")
        self.assertEqual(schedule[90].phase, "recovery")
        self.assertEqual(schedule[120].phase, "tail")
        self.assertEqual(schedule[122].phase, "context")
        self.assertEqual(schedule[0].segment_id, "A:0")
        self.assertEqual(schedule[120].segment_id, "A:1")
        self.assertEqual(schedule[122].segment_id, "B:0")
        self.assertEqual(schedule[0].episode_id, "A:0:0")
        self.assertIsNone(schedule[120].episode_id)
        self.assertEqual(schedule[122].episode_id, "B:0:0")
        self.assertEqual(schedule[90].episode_offset, 90)
        self.assertEqual(schedule[120].episode_offset, 0)

    def test_episode_schedule_sorts_for_protocol_but_restores_input_order(self):
        schedule = episode_schedule(
            ["A/0001", "A/0000", "A/0002"],
            context=1,
            masked=1,
            recovery=1,
            raw_frame_step=1,
        )

        self.assertEqual([row.phase for row in schedule], ["masked", "context", "recovery"])
        self.assertEqual([row.episode_offset for row in schedule], [1, 0, 2])

    def test_episode_schedule_allows_no_recovery(self):
        schedule = episode_schedule(
            [f"A/{frame:04d}" for frame in range(6)],
            context=2,
            masked=4,
            recovery=0,
            raw_frame_step=1,
        )

        self.assertEqual([row.phase for row in schedule], ["context"] * 2 + ["masked"] * 4)

    def test_episode_schedule_rejects_invalid_lengths_steps_and_duplicates(self):
        ids = ["A/0000", "A/0001", "A/0002"]
        for context, masked, recovery, raw_frame_step in (
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, -1, 1),
            (1, 1, 1, 0),
        ):
            with self.subTest(
                context=context,
                masked=masked,
                recovery=recovery,
                raw_frame_step=raw_frame_step,
            ), self.assertRaises(ValueError):
                episode_schedule(ids, context, masked, recovery, raw_frame_step)
        with self.assertRaises(ValueError):
            episode_schedule(["A/0000", "A/0000"], 1, 1, 1, 1)

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

    def test_causal_ema_resets_at_gap_and_sequence_and_restores_row_order(self):
        ids = np.asarray(["A/0001", "A/0000", "A/0003", "B/0000"])
        values = np.asarray([[1.0], [0.0], [10.0], [20.0]], dtype=np.float32)

        filtered = temporal.causal_ema(ids, values, decay=0.5, raw_frame_step=1)

        self.assertEqual(filtered.dtype, np.float32)
        np.testing.assert_allclose(filtered[:, 0], [0.5, 0.0, 10.0, 20.0])

    def test_causal_ema_rejects_invalid_decay_and_rows(self):
        ids = np.asarray(["A/0000", "A/0001"])
        values = np.zeros((2, 1), dtype=np.float32)
        for decay in (-0.1, 1.0, float("nan"), float("inf")):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                temporal.causal_ema(ids, values, decay=decay, raw_frame_step=1)
        with self.assertRaises(ValueError):
            temporal.causal_ema(ids, values[:1], decay=0.5, raw_frame_step=1)

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

    def test_dense_history_and_feature_delta_use_separate_cadences(self):
        ids = [f"A/{frame:04d}" for frame in range(9)]
        rope = np.repeat(np.arange(9, dtype=np.float32)[:, None], 5, axis=1)
        cache = toy_cache(ids, rope)

        features = temporal_features(cache, raw_frame_step=1, delta_step=4)
        windows, valid, lengths = build_causal_windows(
            ids, features, history_length=3, raw_frame_step=1, history_step=4
        )

        np.testing.assert_array_equal(windows[8, :, 50], [0.0, 4.0, 8.0])
        np.testing.assert_array_equal(valid[8], [True, True, True])
        self.assertEqual(int(lengths[8]), 3)
        np.testing.assert_array_equal(features[8, 65:], [4.0] * 5)

    def test_missing_raw_frame_resets_sparse_history_and_feature_delta(self):
        ids = [f"A/{frame:04d}" for frame in (0, 1, 2, 3, 4, 6, 7, 8)]
        rope = np.repeat(
            np.asarray([0, 1, 2, 3, 4, 6, 7, 8], dtype=np.float32)[:, None],
            5,
            axis=1,
        )
        cache = toy_cache(ids, rope)

        features = temporal_features(cache, raw_frame_step=1, delta_step=4)
        windows, valid, lengths = build_causal_windows(
            ids, features, history_length=3, raw_frame_step=1, history_step=4
        )

        np.testing.assert_array_equal(valid[-1], [True, False, False])
        self.assertEqual(int(lengths[-1]), 1)
        np.testing.assert_array_equal(windows[-1, 0, 50], 8.0)
        np.testing.assert_array_equal(features[-1, 65:], 0.0)

    def test_temporal_feature_delta_default_preserves_v1_values(self):
        cache = toy_cache(
            ["A/0000", "A/0004", "A/0008"],
            np.arange(15, dtype=np.float32).reshape(3, 5),
        )

        implicit = temporal_features(cache, raw_frame_step=4)
        explicit = temporal_features(cache, raw_frame_step=4, delta_step=4)

        np.testing.assert_array_equal(implicit, explicit)

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

        for delta_step in (0, 6):
            with self.subTest(delta_step=delta_step), self.assertRaises(ValueError):
                temporal_features(cache, raw_frame_step=4, delta_step=delta_step)


class TemporalAugmentationTest(unittest.TestCase):
    def test_temporal_stats_ignore_validation_extremes(self):
        features = np.zeros((6, 70), dtype=np.float32)
        features[:4, 0] = [0, 1, 2, 3]
        features[:, 60:65] = 1.0
        features[4:, 0] = 9999

        mean, std = temporal.temporal_feature_stats(features, np.asarray([0, 1, 2, 3]))

        self.assertEqual(mean.dtype, np.float32)
        self.assertEqual(std.dtype, np.float32)
        self.assertEqual(mean.shape, (70,))
        self.assertEqual(std.shape, (70,))
        self.assertAlmostEqual(float(mean[0]), 1.5)
        self.assertLess(float(std[0]), 2.0)
        self.assertAlmostEqual(float(std[1]), 1e-4)
        np.testing.assert_array_equal(std[60:65], np.ones(5, dtype=np.float32))
        self.assertTrue((std >= 1e-4).all())

        dropped = features[:4].copy()
        dropped[0, [60, 62, 64]] = 0.0
        normalized_valid = ((dropped - mean) / std)[:, 60:65]
        np.testing.assert_array_equal(
            normalized_valid,
            dropped[:, 60:65] - np.float32(1.0),
        )
        self.assertTrue(np.isfinite(normalized_valid).all())
        self.assertLessEqual(float(np.abs(normalized_valid).max()), 1.0)

    def test_temporal_stats_reject_invalid_rows_and_indices(self):
        features = np.zeros((4, 70), dtype=np.float32)
        for bad_features, rows in (
            (np.zeros((4, 69), dtype=np.float32), np.asarray([0, 1])),
            (features, np.asarray([], dtype=np.int64)),
            (features, np.asarray([[0, 1]])),
            (features, np.asarray([0.0, 1.0])),
            (features, np.asarray([0, 4])),
        ):
            with self.subTest(shape=bad_features.shape, rows=rows), self.assertRaises(ValueError):
                temporal.temporal_feature_stats(bad_features, rows)

    def test_zero_sensor_clears_all_sensor_channels_without_mutating_input(self):
        cache = toy_cache(
            ["A/0000", "A/0001", "B/0000"],
            np.full((3, 5), 0.6, dtype=np.float32),
        )
        before = copy.deepcopy(cache)

        out = temporal.prepare_temporal_cache(
            cache,
            sensor_mode="zero",
            seed=3,
            raw_frame_step=1,
            aug_noise_std=10.0,
            aug_dropout=1.0,
            aug_bias_std=10.0,
            aug_scale_range=1.0,
        )

        self.assertFalse(out["rope_valid"].any())
        np.testing.assert_array_equal(out["input_rope_norm"], 0.0)
        np.testing.assert_array_equal(cache["input_rope_norm"], before["input_rope_norm"])
        np.testing.assert_array_equal(cache["rope_valid"], before["rope_valid"])
        self.assertIsNot(out["input_rope_norm"], cache["input_rope_norm"])

    def test_segment_bias_and_scale_are_shared_but_reset_at_gaps(self):
        ids = ["A/0000", "A/0001", "A/0003", "A/0004", "B/0000", "B/0001"]
        cache = toy_cache(ids, np.full((6, 5), 0.4, dtype=np.float32))

        out = temporal.prepare_temporal_cache(
            cache,
            sensor_mode="normal",
            seed=7,
            raw_frame_step=1,
            aug_noise_std=0.0,
            aug_dropout=0.0,
            aug_bias_std=0.02,
            aug_scale_range=0.1,
        )

        np.testing.assert_array_equal(out["input_rope_norm"][0], out["input_rope_norm"][1])
        np.testing.assert_array_equal(out["input_rope_norm"][2], out["input_rope_norm"][3])
        np.testing.assert_array_equal(out["input_rope_norm"][4], out["input_rope_norm"][5])
        self.assertFalse(np.array_equal(out["input_rope_norm"][0], out["input_rope_norm"][2]))
        self.assertFalse(np.array_equal(out["input_rope_norm"][2], out["input_rope_norm"][4]))

    def test_normal_sensor_is_deterministic_and_preserves_original_invalid(self):
        cache = toy_cache(
            ["A/0000", "A/0001", "B/0000"],
            np.full((3, 5), 0.5, dtype=np.float32),
        )
        cache["rope_valid"][1, 2] = False
        before = copy.deepcopy(cache)
        kwargs = dict(
            sensor_mode="normal",
            seed=11,
            raw_frame_step=1,
            aug_noise_std=0.03,
            aug_dropout=0.25,
            aug_bias_std=0.01,
            aug_scale_range=0.05,
        )

        first = temporal.prepare_temporal_cache(cache, **kwargs)
        second = temporal.prepare_temporal_cache(cache, **kwargs)

        np.testing.assert_array_equal(first["input_rope_norm"], second["input_rope_norm"])
        np.testing.assert_array_equal(first["rope_valid"], second["rope_valid"])
        self.assertFalse(first["rope_valid"][1, 2])
        self.assertEqual(float(first["input_rope_norm"][1, 2]), 0.0)
        np.testing.assert_array_equal(cache["input_rope_norm"], before["input_rope_norm"])
        np.testing.assert_array_equal(cache["rope_valid"], before["rope_valid"])

    def test_prepare_temporal_cache_rejects_invalid_controls(self):
        cache = toy_cache(["A/0000", "B/0000"], np.full((2, 5), 0.5))
        defaults = dict(
            sensor_mode="normal",
            seed=0,
            raw_frame_step=1,
            aug_noise_std=0.0,
            aug_dropout=0.0,
            aug_bias_std=0.0,
            aug_scale_range=0.0,
        )
        for key, value in (
            ("sensor_mode", "bad"),
            ("raw_frame_step", 0),
            ("aug_noise_std", -0.1),
            ("aug_dropout", -0.1),
            ("aug_dropout", 1.1),
            ("aug_bias_std", -0.1),
            ("aug_scale_range", -0.1),
            ("aug_scale_range", 1.1),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                temporal.prepare_temporal_cache(cache, **dict(defaults, **{key: value}))

    def test_shuffle_history_preserves_current_padding_and_input(self):
        windows = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
        valid = np.asarray([[True, True, True, True], [True, True, False, False]])
        before = windows.copy()

        first = temporal.shuffle_history(windows, valid, seed=3)
        second = temporal.shuffle_history(windows, valid, seed=3)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(windows, before)
        for row, length in enumerate(valid.sum(1)):
            np.testing.assert_array_equal(first[row, length - 1], windows[row, length - 1])
            np.testing.assert_array_equal(first[row, length:], windows[row, length:])
            self.assertEqual(
                {tuple(value) for value in first[row, : length - 1]},
                {tuple(value) for value in windows[row, : length - 1]},
            )

    def test_shuffle_history_rejects_non_left_aligned_mask(self):
        windows = np.zeros((1, 3, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "left-aligned"):
            temporal.shuffle_history(windows, np.asarray([[True, False, True]]), seed=0)


def temporal_training_fixture(root: Path):
    teacher_dir = root / "teacher"
    teacher_dir.mkdir()
    ids = np.asarray([f"{sequence}/{frame:04d}" for sequence in ("A", "B") for frame in range(4)])
    input_rope = np.repeat(np.linspace(0.1, 0.8, len(ids), dtype=np.float32)[:, None], 5, axis=1)
    cache = toy_cache(ids, input_rope)
    cache["base_rope_norm"][:] = 0.2
    split_seed = 17
    split = deterministic_sequence_split(ids, val_fraction=0.5, seed=split_seed)

    framewise = RopeAlphaStudent(out_dim=5, hidden_dim=8, max_alpha=0.5, in_dim=65)
    with torch.no_grad():
        for parameter in framewise.parameters():
            parameter.zero_()
        framewise.net[0].weight[0, 50] = 1.0
        framewise.net[2].weight[0, 0] = 1.0
        framewise.net[4].weight[0, 0] = 1.0
        teacher_alpha = framewise(torch.from_numpy(features_from_cache(cache))).numpy()

    np.savez(teacher_dir / "refiner_eval_cache.npz", **cache)
    np.save(teacher_dir / "alpha.npy", teacher_alpha)
    (teacher_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_samples": len(ids),
                "action_space": "mult5",
                "optimization": {"gate_residual_threshold": 0.1},
            }
        ),
        encoding="utf-8",
    )
    framewise_config = {
        "in_dim": 65,
        "out_dim": 5,
        "hidden_dim": 8,
        "max_alpha": 0.5,
        "image_feature_dim": 0,
        "action_space": "mult5",
        "gate_threshold": 0.1,
        "feature_mean": [0.0] * 65,
        "feature_std": [1.0] * 65,
        "split_by": "sequence",
        "split_seed": split_seed,
        "train_sequences": list(split.train_sequences),
        "val_sequences": list(split.val_sequences),
        "num_train": len(split.train_idx),
        "num_val": len(split.val_idx),
        "seed": 5,
        "sources": [{"dir": str(teacher_dir), "num_samples": len(ids)}],
    }
    checkpoint = root / "framewise.pt"
    torch.save({"model_state": framewise.state_dict(), "config": framewise_config}, checkpoint)
    return teacher_dir, checkpoint, teacher_alpha, framewise_config


def past_residual_training_fixture(root: Path):
    teacher_dir, framewise_checkpoint, teacher_alpha, framewise_config = (
        temporal_training_fixture(root)
    )
    framewise_payload = torch.load(
        framewise_checkpoint, map_location="cpu", weights_only=True
    )
    base = temporal.TemporalRopeAlphaStudent(70, 5, hidden_dim=8, max_alpha=0.5)
    config = {
        "model_type": "causal_gru",
        "in_dim": 70,
        "out_dim": 5,
        "hidden_dim": 8,
        "max_alpha": 0.5,
        "action_space": "mult5",
        "gate_threshold": 0.1,
        "history_length": 1,
        "raw_frame_step": 1,
        "history_step": 1,
        "temporal_feature_mean": [0.0] * 70,
        "temporal_feature_std": [1.0] * 70,
        "split_by": "sequence",
        "split_seed": 17,
        "val_frac": 0.5,
        "train_sequences": framewise_config["train_sequences"],
        "val_sequences": framewise_config["val_sequences"],
        "num_train": framewise_config["num_train"],
        "num_val": framewise_config["num_val"],
        "teacher_source": {"dir": str(teacher_dir), "num_samples": 8},
        "framewise_sources": framewise_config["sources"],
    }
    checkpoint = root / "k1.pt"
    temporal.save_temporal_checkpoint(
        checkpoint, base, config, framewise_payload
    )
    return teacher_dir, checkpoint, teacher_alpha, config


class TemporalTrainingTest(unittest.TestCase):
    def test_past_residual_loss_penalizes_only_logit_residual(self):
        trainer = load_temporal_trainer()
        prediction = torch.tensor([[0.4, -0.3]])
        target = prediction.clone()
        residual = torch.tensor([[2.0, -4.0]])

        loss = trainer._past_residual_loss(prediction, target, residual)

        self.assertAlmostEqual(float(loss), 1e-4 * 10.0)

    def test_v2_one_epoch_no_improvement_keeps_zero_residual_and_k1(self):
        trainer = load_temporal_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, k1_checkpoint, _, _ = past_residual_training_fixture(root)
            out_dir = root / "out"

            summary = trainer.train_temporal_student(
                teacher_dir,
                None,
                "mult5",
                out_dir,
                base_temporal_checkpoint=k1_checkpoint,
                history_length=2,
                raw_frame_step=1,
                inference_raw_frame_step=1,
                history_step=1,
                feature_delta_step=1,
                hidden_dim=8,
                lr=1e-3,
                batch_size=4,
                max_epochs=1,
                patience=1,
                val_frac=0.5,
                split_seed=17,
                seed=3,
                aug_noise_std=0.0,
                aug_dropout=0.0,
                device="cpu",
            )

            path = out_dir / "temporal_student.pt"
            raw = torch.load(path, map_location="cpu", weights_only=True)
            model, config, base, _, framewise, _ = (
                temporal.load_past_residual_checkpoint(path, "cpu")
            )

        self.assertEqual(raw["schema_version"], 2)
        self.assertEqual(config["model_type"], "causal_past_residual_gru")
        self.assertEqual(config["train_raw_frame_step"], 1)
        self.assertEqual(config["inference_raw_frame_step"], 1)
        self.assertEqual(config["feature_delta_step"], 1)
        self.assertEqual(summary["best_epoch"], -1)
        self.assertEqual(summary["base_temporal_zero_baseline_val_l1"], 0.0)
        self.assertEqual(summary["best_temporal_val_l1"], 0.0)
        torch.testing.assert_close(model.head.weight, torch.zeros_like(model.head.weight))
        torch.testing.assert_close(model.head.bias, torch.zeros_like(model.head.bias))
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in framewise.parameters()))
        windows = torch.randn(3, 2, 70)
        lengths = torch.tensor([2, 1, 2])
        k1 = torch.randn(3, 5).clamp(-0.5, 0.5)
        self.assertTrue(torch.equal(model(windows, lengths, k1), k1))

    def test_v2_optimizer_excludes_nested_k1_and_framewise_parameters(self):
        trainer = load_temporal_trainer()
        captured = {}
        original_load = trainer.load_temporal_checkpoint
        original_adam = trainer.torch.optim.Adam

        def capture_load(*args, **kwargs):
            loaded = original_load(*args, **kwargs)
            captured["nested_parameters"] = [
                parameter
                for module in (loaded[0], loaded[2])
                for parameter in module.parameters()
            ]
            captured["nested"] = {
                id(parameter) for parameter in captured["nested_parameters"]
            }
            return loaded

        def capture_adam(parameters, *args, **kwargs):
            parameters = list(parameters)
            captured["optimizer"] = {id(parameter) for parameter in parameters}
            return original_adam(parameters, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, k1_checkpoint, _, _ = past_residual_training_fixture(root)
            with mock.patch.object(
                trainer, "load_temporal_checkpoint", side_effect=capture_load
            ), mock.patch.object(trainer.torch.optim, "Adam", side_effect=capture_adam):
                trainer.train_temporal_student(
                    teacher_dir,
                    None,
                    "mult5",
                    root / "out",
                    base_temporal_checkpoint=k1_checkpoint,
                    history_length=2,
                    raw_frame_step=1,
                    inference_raw_frame_step=1,
                    history_step=1,
                    feature_delta_step=1,
                    hidden_dim=8,
                    batch_size=4,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    aug_noise_std=0.0,
                    aug_dropout=0.0,
                    device="cpu",
                )

        self.assertTrue(captured["optimizer"])
        self.assertTrue(captured["nested"])
        self.assertTrue(captured["optimizer"].isdisjoint(captured["nested"]))
        self.assertFalse(
            any(parameter.requires_grad for parameter in captured["nested_parameters"])
        )

    def test_subset_cache_copies_required_rows_in_order(self):
        trainer = load_temporal_trainer()
        cache = toy_cache(
            ["A/0000", "A/0001", "B/0000"],
            np.arange(15, dtype=np.float32).reshape(3, 5),
        )
        cache["unused"] = np.arange(3)

        subset = trainer._subset_cache(cache, np.asarray([2, 0]))

        self.assertEqual(
            set(subset),
            {
                "sample_id",
                "base_hand_pose",
                "base_rope_norm",
                "input_rope_norm",
                "rope_valid",
            },
        )
        for key in subset:
            np.testing.assert_array_equal(subset[key], np.asarray(cache[key])[[2, 0]])
        subset["input_rope_norm"][0, 0] = -1.0
        self.assertNotEqual(float(cache["input_rope_norm"][2, 0]), -1.0)

    def test_two_sequence_one_epoch_cpu_smoke_is_self_contained_and_clean_val(self):
        trainer = load_temporal_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, framewise_config = temporal_training_fixture(root)
            out_dir = root / "out"

            summary = trainer.train_temporal_student(
                teacher_dir,
                checkpoint,
                "mult5",
                out_dir,
                history_length=2,
                raw_frame_step=1,
                history_step=1,
                hidden_dim=8,
                lr=1e-3,
                batch_size=4,
                max_epochs=1,
                patience=1,
                val_frac=0.5,
                split_seed=17,
                seed=3,
                sensor_mode="normal",
                shuffle_rope=False,
                shuffle_history=False,
                aug_noise_std=10.0,
                aug_dropout=1.0,
                aug_bias_std=10.0,
                aug_scale_range=1.0,
                device="cpu",
            )

            checkpoint_path = out_dir / "temporal_student.pt"
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue((out_dir / "train_log.json").exists())
            config = summary["config"]
            self.assertEqual(config["model_type"], "causal_gru")
            self.assertTrue(config["framewise_frozen"])
            self.assertEqual(config["split_by"], "sequence")
            self.assertTrue(set(config["train_sequences"]).isdisjoint(config["val_sequences"]))
            self.assertEqual(len(config["temporal_feature_mean"]), 70)
            self.assertEqual(len(config["temporal_feature_std"]), 70)
            self.assertEqual(config["val_frac"], 0.5)
            self.assertEqual(config["lr"], 1e-3)
            self.assertEqual(config["batch_size"], 4)
            self.assertEqual(config["max_epochs"], 1)
            self.assertEqual(config["patience"], 1)
            self.assertEqual(
                config["clean_validation"],
                {
                    "augmentation": False,
                    "sensor_mode": "normal",
                    "shuffle_rope": False,
                    "shuffle_history": False,
                },
            )
            self.assertEqual(summary["framewise_zero_baseline_val_l1"], 0.0)
            self.assertEqual(summary["best_temporal_val_l1"], 0.0)

            raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            for key, value in torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"].items():
                torch.testing.assert_close(raw["framewise"]["model_state"][key], value)
            model, loaded_config, framewise, loaded_framewise = temporal.load_temporal_checkpoint(
                checkpoint_path, "cpu"
            )

        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_framewise, framewise_config)
        self.assertTrue(all(not parameter.requires_grad for parameter in framewise.parameters()))
        torch.testing.assert_close(model.head.weight, torch.zeros_like(model.head.weight))
        torch.testing.assert_close(model.head.bias, torch.zeros_like(model.head.bias))
        windows = torch.randn(2, 2, 70)
        lengths = torch.tensor([2, 1])
        base_alpha = torch.tensor([[0.1] * 5, [-0.2] * 5])
        torch.testing.assert_close(model(windows, lengths, base_alpha), base_alpha)

    def test_shuffle_rope_uses_partition_permutation_without_crossing_splits(self):
        trainer = load_temporal_trainer()
        row_values = np.arange(8, dtype=np.float32)
        cache = toy_cache(
            [f"{sequence}/{frame:04d}" for sequence in "ABCD" for frame in range(2)],
            np.repeat(row_values[:, None], 5, axis=1),
        )
        cache["base_rope_norm"] = np.repeat((row_values + 10)[:, None], 5, axis=1)
        cache["rope_valid"] = np.asarray(
            [[bool((row >> finger) & 1) for finger in range(5)] for row in range(8)]
        )
        before = copy.deepcopy(cache)
        train_rows = np.asarray([0, 1, 2, 3])
        val_rows = np.asarray([4, 5, 6, 7])

        trainer._shuffle_rope_within(cache, (train_rows, val_rows), seed=0)

        expected_permutation = np.asarray([2, 0, 1, 3, 7, 6, 5, 4])
        for key in ("base_rope_norm", "input_rope_norm", "rope_valid"):
            np.testing.assert_array_equal(cache[key], before[key][expected_permutation])
            for rows in (train_rows, val_rows):
                np.testing.assert_array_equal(
                    np.sort(cache[key][rows], axis=0),
                    np.sort(before[key][rows], axis=0),
                )
        reassigned_row = int(cache["input_rope_norm"][0, 0])
        self.assertEqual(reassigned_row, 2)
        self.assertNotEqual(
            sequence_frame(cache["sample_id"][0])[0],
            sequence_frame(before["sample_id"][reassigned_row])[0],
        )

    def test_training_pipeline_never_mixes_train_and_validation_caches(self):
        trainer = load_temporal_trainer()
        calls = {
            "prepare": [],
            "temporal_features": [],
            "windows": [],
            "framewise": [],
            "shuffle": [],
        }
        original_prepare = trainer.prepare_temporal_cache
        original_temporal_features = trainer.temporal_features
        original_windows = trainer._normalized_windows
        original_framewise = trainer._framewise_alpha
        original_shuffle = trainer._shuffle_rope_within

        def ids(cache):
            return tuple(str(sample_id) for sample_id in cache["sample_id"])

        def prepare_wrapper(cache, *args, **kwargs):
            calls["prepare"].append(ids(cache))
            return original_prepare(cache, *args, **kwargs)

        def temporal_features_wrapper(cache, *args, **kwargs):
            calls["temporal_features"].append(ids(cache))
            return original_temporal_features(cache, *args, **kwargs)

        def windows_wrapper(cache, *args, **kwargs):
            calls["windows"].append(ids(cache))
            return original_windows(cache, *args, **kwargs)

        def framewise_wrapper(model, config, cache, device):
            calls["framewise"].append(ids(cache))
            return original_framewise(model, config, cache, device)

        def shuffle_wrapper(cache, partitions, seed):
            calls["shuffle"].append(
                (
                    ids(cache),
                    tuple(tuple(np.asarray(rows).tolist()) for rows in partitions),
                )
            )
            return original_shuffle(cache, partitions, seed)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, framewise_config = temporal_training_fixture(root)
            all_ids = tuple(
                f"{sequence}/{frame:04d}"
                for sequence in ("A", "B")
                for frame in range(4)
            )
            train_sequences = set(framewise_config["train_sequences"])
            train_ids = tuple(
                sample_id
                for sample_id in all_ids
                if sequence_frame(sample_id)[0] in train_sequences
            )
            val_ids = tuple(sample_id for sample_id in all_ids if sample_id not in train_ids)

            with mock.patch.object(
                trainer, "prepare_temporal_cache", prepare_wrapper
            ), mock.patch.object(
                trainer, "temporal_features", temporal_features_wrapper
            ), mock.patch.object(
                trainer, "_normalized_windows", windows_wrapper
            ), mock.patch.object(
                trainer, "_framewise_alpha", framewise_wrapper
            ), mock.patch.object(
                trainer, "_shuffle_rope_within", shuffle_wrapper
            ):
                trainer.train_temporal_student(
                    teacher_dir,
                    checkpoint,
                    "mult5",
                    root / "out",
                    history_length=2,
                    hidden_dim=8,
                    batch_size=4,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    shuffle_rope=True,
                    shuffle_history=True,
                    device="cpu",
                )

        self.assertEqual(calls["prepare"], [val_ids, train_ids])
        self.assertEqual(calls["temporal_features"], [train_ids, val_ids, train_ids])
        self.assertEqual(calls["windows"], [val_ids, train_ids])
        self.assertEqual(calls["framewise"], [val_ids, train_ids])
        self.assertEqual([entry[0] for entry in calls["shuffle"]], [val_ids, train_ids])
        for sample_ids, partitions in calls["shuffle"]:
            self.assertEqual(partitions, (tuple(range(len(sample_ids))),))
        for entries in calls.values():
            for entry in entries:
                sample_ids = entry[0] if isinstance(entry[0], tuple) else entry
                self.assertNotEqual(sample_ids, all_ids)

    def test_zero_sensor_is_retained_in_clean_validation(self):
        trainer = load_temporal_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, _ = temporal_training_fixture(root)

            summary = trainer.train_temporal_student(
                teacher_dir,
                checkpoint,
                "mult5",
                root / "out",
                history_length=2,
                hidden_dim=8,
                batch_size=4,
                max_epochs=1,
                patience=1,
                val_frac=0.5,
                split_seed=17,
                sensor_mode="zero",
                aug_noise_std=10.0,
                aug_dropout=1.0,
                aug_bias_std=10.0,
                aug_scale_range=1.0,
                device="cpu",
            )

        self.assertGreater(summary["framewise_zero_baseline_val_l1"], 0.0)
        self.assertEqual(summary["config"]["clean_validation"]["sensor_mode"], "zero")

    def test_trainer_rejects_framewise_source_from_another_teacher(self):
        trainer = load_temporal_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, _ = temporal_training_fixture(root)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["config"]["sources"] = [
                {"dir": str(root / "another_teacher"), "num_samples": 8}
            ]
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(ValueError, "sources"):
                trainer.train_temporal_student(
                    teacher_dir,
                    checkpoint,
                    "mult5",
                    root / "out",
                    history_length=2,
                    hidden_dim=8,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    device="cpu",
                )

    def test_non_finite_training_loss_fails_loudly(self):
        trainer = load_temporal_trainer()
        original_forward = trainer.TemporalRopeAlphaStudent.forward

        def nan_during_training(model, windows, lengths, base_alpha):
            if model.training:
                return base_alpha + model.head.weight.sum() * torch.full_like(
                    base_alpha, float("nan")
                )
            return original_forward(model, windows, lengths, base_alpha)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, _ = temporal_training_fixture(root)
            with mock.patch.object(
                trainer.TemporalRopeAlphaStudent, "forward", nan_during_training
            ), self.assertRaisesRegex(FloatingPointError, "train"):
                trainer.train_temporal_student(
                    teacher_dir,
                    checkpoint,
                    "mult5",
                    root / "out",
                    history_length=2,
                    hidden_dim=8,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    device="cpu",
                )

    def test_non_finite_validation_loss_fails_loudly(self):
        trainer = load_temporal_trainer()
        original_forward = trainer.TemporalRopeAlphaStudent.forward

        def nan_during_validation(model, windows, lengths, base_alpha):
            if not model.training:
                return torch.full_like(base_alpha, float("nan"))
            return original_forward(model, windows, lengths, base_alpha)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, _ = temporal_training_fixture(root)
            with mock.patch.object(
                trainer.TemporalRopeAlphaStudent, "forward", nan_during_validation
            ), self.assertRaisesRegex(FloatingPointError, "validation"):
                trainer.train_temporal_student(
                    teacher_dir,
                    checkpoint,
                    "mult5",
                    root / "out",
                    history_length=2,
                    hidden_dim=8,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    device="cpu",
                )

    def test_trainer_rejects_incompatible_framewise_provenance(self):
        trainer = load_temporal_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, _ = temporal_training_fixture(root)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["config"]["split_by"] = "frame"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "split_by"):
                trainer.train_temporal_student(
                    teacher_dir,
                    checkpoint,
                    "mult5",
                    root / "out",
                    history_length=2,
                    raw_frame_step=1,
                    history_step=1,
                    hidden_dim=8,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    device="cpu",
                )

    def test_trainer_rejects_stale_framewise_validity_scale(self):
        trainer = load_temporal_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir, checkpoint, _, _ = temporal_training_fixture(root)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["config"]["feature_std"][60:65] = [1e-4] * 5
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(ValueError, "validity"):
                trainer.train_temporal_student(
                    teacher_dir,
                    checkpoint,
                    "mult5",
                    root / "out",
                    history_length=2,
                    hidden_dim=8,
                    max_epochs=1,
                    patience=1,
                    val_frac=0.5,
                    split_seed=17,
                    device="cpu",
                )

    def test_temporal_training_cli_has_the_plan_controls(self):
        trainer = load_temporal_trainer()
        args = trainer.parse_args(
            [
                "--teacher-dir", "teacher",
                "--framewise-checkpoint", "framewise.pt",
                "--action-space", "mult5",
                "--out-dir", "out",
                "--history-length", "2",
                "--raw-frame-step", "1",
                "--history-step", "1",
                "--hidden-dim", "8",
                "--lr", "0.001",
                "--batch-size", "4",
                "--max-epochs", "1",
                "--patience", "1",
                "--val-frac", "0.5",
                "--split-seed", "17",
                "--seed", "3",
                "--sensor-mode", "zero",
                "--shuffle-rope",
                "--shuffle-history",
                "--aug-noise-std", "0.1",
                "--aug-dropout", "0.2",
                "--aug-bias-std", "0.3",
                "--aug-scale-range", "0.4",
                "--device", "cpu",
            ]
        )
        self.assertEqual(args.teacher_dir, Path("teacher"))
        self.assertEqual(args.framewise_checkpoint, Path("framewise.pt"))
        self.assertEqual(args.sensor_mode, "zero")
        self.assertTrue(args.shuffle_rope)
        self.assertTrue(args.shuffle_history)

    def test_v2_training_cli_uses_mutually_exclusive_base_checkpoint(self):
        trainer = load_temporal_trainer()
        common = [
            "--teacher-dir",
            "teacher",
            "--action-space",
            "mult5",
            "--out-dir",
            "out",
        ]
        args = trainer.parse_args(
            common
            + [
                "--base-temporal-checkpoint",
                "k1.pt",
                "--raw-frame-step",
                "4",
                "--inference-raw-frame-step",
                "1",
                "--history-step",
                "4",
                "--feature-delta-step",
                "4",
            ]
        )

        self.assertIsNone(args.framewise_checkpoint)
        self.assertEqual(args.base_temporal_checkpoint, Path("k1.pt"))
        self.assertEqual(args.inference_raw_frame_step, 1)
        self.assertEqual(args.feature_delta_step, 4)
        with self.assertRaises(SystemExit):
            trainer.parse_args(
                common
                + [
                    "--framewise-checkpoint",
                    "frame.pt",
                    "--base-temporal-checkpoint",
                    "k1.pt",
                ]
            )


def framewise_payload():
    model = RopeAlphaStudent(out_dim=2, hidden_dim=5, max_alpha=0.5, in_dim=65)
    with torch.no_grad():
        model.net[-1].bias.copy_(torch.tensor([0.1, -0.2]))
    config = {
        "in_dim": 65,
        "out_dim": 2,
        "hidden_dim": 5,
        "max_alpha": 0.5,
        "tag": "framewise",
    }
    return model, {
        "model_state": model.state_dict(),
        "config": config,
        "ignored": "not nested in temporal checkpoints",
    }


def temporal_checkpoint_config(in_dim=4, out_dim=2, hidden_dim=6, max_alpha=0.5):
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden_dim": hidden_dim,
        "max_alpha": max_alpha,
        "temporal_feature_mean": [0.0] * in_dim,
        "temporal_feature_std": [1.0] * in_dim,
    }


def k1_payload():
    framewise = RopeAlphaStudent(out_dim=5, hidden_dim=7, max_alpha=0.5, in_dim=65)
    framewise_config = {
        "in_dim": 65,
        "out_dim": 5,
        "hidden_dim": 7,
        "max_alpha": 0.5,
        "image_feature_dim": 0,
        "action_space": "mult5",
        "gate_threshold": 0.1,
        "feature_mean": [0.0] * 65,
        "feature_std": [1.0] * 65,
        "split_by": "sequence",
        "split_seed": 17,
        "train_sequences": ["A"],
        "val_sequences": ["B"],
        "num_train": 4,
        "num_val": 4,
        "sources": [{"dir": "teacher", "num_samples": 8}],
    }
    base = temporal.TemporalRopeAlphaStudent(70, 5, hidden_dim=6, max_alpha=0.5)
    config = {
        "model_type": "causal_gru",
        "in_dim": 70,
        "out_dim": 5,
        "hidden_dim": 6,
        "max_alpha": 0.5,
        "action_space": "mult5",
        "gate_threshold": 0.1,
        "history_length": 1,
        "raw_frame_step": 4,
        "history_step": 4,
        "temporal_feature_mean": [0.0] * 70,
        "temporal_feature_std": [1.0] * 70,
        "split_by": "sequence",
        "split_seed": 17,
        "val_frac": 0.5,
        "train_sequences": ["A"],
        "val_sequences": ["B"],
        "num_train": 4,
        "num_val": 4,
        "teacher_source": {"dir": "teacher", "num_samples": 8},
        "framewise_sources": [{"dir": "teacher", "num_samples": 8}],
    }
    return base, {
        "model_state": base.state_dict(),
        "config": config,
        "framewise": {
            "model_state": framewise.state_dict(),
            "config": framewise_config,
        },
    }


def past_residual_config():
    return {
        "schema_version": 2,
        "model_type": "causal_past_residual_gru",
        "in_dim": 70,
        "out_dim": 5,
        "hidden_dim": 8,
        "max_alpha": 0.5,
        "action_space": "mult5",
        "gate_threshold": 0.1,
        "history_length": 4,
        "train_raw_frame_step": 4,
        "inference_raw_frame_step": 1,
        "history_step": 4,
        "feature_delta_step": 4,
        "temporal_feature_mean": [0.0] * 70,
        "temporal_feature_std": [1.0] * 70,
        "split_by": "sequence",
        "split_seed": 17,
        "val_frac": 0.5,
        "train_sequences": ["A"],
        "val_sequences": ["B"],
        "num_train": 4,
        "num_val": 4,
        "teacher_source": {"dir": "teacher", "num_samples": 8},
    }


class TemporalModelTest(unittest.TestCase):
    def test_past_residual_zero_head_is_bitwise_k1_identity(self):
        model = temporal.PastResidualRopeAlphaStudent(
            in_dim=70, out_dim=5, hidden_dim=8, max_alpha=0.5
        )
        windows = torch.randn(4, 4, 70)
        lengths = torch.tensor([4, 3, 2, 1])
        base = torch.tensor(
            [[0.1] * 5, [-0.2] * 5, [0.49] * 5, [-0.5] * 5],
            dtype=torch.float32,
        )

        actual = model(windows, lengths, base)

        self.assertTrue(torch.equal(actual, base))

    def test_past_residual_no_past_is_bitwise_k1_after_training(self):
        torch.manual_seed(41)
        model = temporal.PastResidualRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_()
        windows = torch.randn(3, 4, 4)
        lengths = torch.ones(3, dtype=torch.long)
        base = torch.tensor([[0.1, -0.2], [0.5, -0.5], [-0.0, 0.0]])

        actual = model(windows, lengths, base)

        self.assertTrue(torch.equal(actual, base))

    def test_past_residual_excludes_current_slot_but_uses_past(self):
        torch.manual_seed(43)
        model = temporal.PastResidualRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_()
        windows = torch.randn(2, 4, 4)
        lengths = torch.tensor([4, 2])
        expected = model.logit_residual(windows, lengths)

        changed_current = windows.clone()
        changed_current[0, 3] += 1_000.0
        changed_current[1, 1] -= 1_000.0
        actual = model.logit_residual(changed_current, lengths)

        self.assertTrue(torch.equal(actual, expected))

        changed_past = windows.clone()
        changed_past[0, 0] += 1_000.0
        past_actual = model.logit_residual(changed_past, lengths)
        self.assertFalse(torch.equal(past_actual[0], expected[0]))

    def test_past_residual_output_is_bounded(self):
        model = temporal.PastResidualRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            model.head.bias.fill_(100.0)
        output = model(
            torch.zeros(2, 2, 4),
            torch.full((2,), 2, dtype=torch.long),
            torch.tensor([[0.49, -0.49], [0.0, 0.0]]),
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertLessEqual(float(output.abs().max()), 0.5)

    def test_past_residual_schema_v2_roundtrip_is_self_contained(self):
        torch.manual_seed(47)
        model = temporal.PastResidualRopeAlphaStudent(70, 5, hidden_dim=8, max_alpha=0.5)
        with torch.no_grad():
            model.head.weight.normal_()
        config = past_residual_config()
        base_model, base_payload = k1_payload()

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "k1.pt"
            torch.save(base_payload, source)
            embedded = torch.load(source, map_location="cpu", weights_only=True)
            path = Path(tmp) / "v2.pt"
            temporal.save_past_residual_checkpoint(path, model, config, embedded)
            source.unlink()

            raw = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(raw), {"schema_version", "model_state", "config", "base_temporal"}
            )
            self.assertEqual(raw["schema_version"], 2)
            self.assertEqual(raw["base_temporal"]["config"], base_payload["config"])
            loaded = temporal.load_past_residual_checkpoint(path, "cpu")

        (
            loaded_model,
            loaded_config,
            loaded_base,
            loaded_base_config,
            loaded_framewise,
            loaded_framewise_config,
        ) = loaded
        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_base_config, base_payload["config"])
        self.assertEqual(loaded_framewise_config, base_payload["framewise"]["config"])
        self.assertFalse(loaded_model.training)
        self.assertTrue(all(not p.requires_grad for p in loaded_base.parameters()))
        self.assertTrue(all(not p.requires_grad for p in loaded_framewise.parameters()))
        windows = torch.randn(3, 4, 70)
        lengths = torch.tensor([4, 2, 1])
        base_alpha = torch.randn(3, 5).clamp(-0.5, 0.5)
        torch.testing.assert_close(
            loaded_model(windows, lengths, base_alpha),
            model(windows, lengths, base_alpha),
        )
        for key, value in base_model.state_dict().items():
            torch.testing.assert_close(loaded_base.state_dict()[key], value)

    def test_past_residual_schema_rejects_provenance_and_cadence_mismatches(self):
        model = temporal.PastResidualRopeAlphaStudent(70, 5, hidden_dim=8, max_alpha=0.5)
        _, base_payload = k1_payload()
        config = past_residual_config()
        cases = (
            (dict(config, action_space="pose45"), "action_space"),
            (dict(config, gate_threshold=0.2), "gate_threshold"),
            (dict(config, split_seed=18), "split_seed"),
            (
                dict(config, teacher_source={"dir": "other", "num_samples": 8}),
                "teacher_source",
            ),
            (dict(config, out_dim=15), "out_dim"),
            (dict(config, train_raw_frame_step=2), "train_raw_frame_step"),
            (dict(config, inference_raw_frame_step=3), "inference_raw_frame_step"),
            (dict(config, history_step=8), "history_step"),
            (dict(config, feature_delta_step=8), "feature_delta_step"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            for index, (bad_config, message) in enumerate(cases):
                path = Path(tmp) / f"bad-{index}.pt"
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    temporal.save_past_residual_checkpoint(
                        path, model, bad_config, base_payload
                    )
                self.assertFalse(path.exists())

    def test_past_residual_save_rejects_broken_nested_k1_state(self):
        model = temporal.PastResidualRopeAlphaStudent(70, 5, hidden_dim=8, max_alpha=0.5)
        _, base_payload = k1_payload()
        base_payload = copy.deepcopy(base_payload)
        base_payload["model_state"].pop("head.bias")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken-v2.pt"
            with self.assertRaises(RuntimeError):
                temporal.save_past_residual_checkpoint(
                    path, model, past_residual_config(), base_payload
                )
            self.assertFalse(path.exists())

    def test_past_residual_schema_rejects_nested_framewise_provenance(self):
        model = temporal.PastResidualRopeAlphaStudent(70, 5, hidden_dim=8, max_alpha=0.5)
        config = past_residual_config()
        _, source = k1_payload()
        cases = []

        bad_split = copy.deepcopy(source)
        bad_split["framewise"]["config"]["split_seed"] = 18
        cases.append((bad_split, "split_seed"))

        bad_teacher = copy.deepcopy(source)
        bad_teacher["framewise"]["config"]["sources"] = [
            {"dir": "other", "num_samples": 8}
        ]
        cases.append((bad_teacher, "sources"))

        bad_image = copy.deepcopy(source)
        image_framewise = RopeAlphaStudent(
            out_dim=5, hidden_dim=7, max_alpha=0.5, in_dim=68
        )
        bad_image["framewise"]["model_state"] = image_framewise.state_dict()
        bad_image["framewise"]["config"]["in_dim"] = 68
        bad_image["framewise"]["config"]["image_feature_dim"] = 3
        cases.append((bad_image, "image_feature_dim"))

        with tempfile.TemporaryDirectory() as tmp:
            for index, (bad_source, message) in enumerate(cases):
                path = Path(tmp) / f"nested-{index}.pt"
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    temporal.save_past_residual_checkpoint(
                        path, model, config, bad_source
                    )
                self.assertFalse(path.exists())

    def test_past_residual_schema_requires_70d_temporal_features(self):
        _, source = k1_payload()
        source = copy.deepcopy(source)
        base = temporal.TemporalRopeAlphaStudent(69, 5, hidden_dim=6, max_alpha=0.5)
        source["model_state"] = base.state_dict()
        source["config"]["in_dim"] = 69
        source["config"]["temporal_feature_mean"] = [0.0] * 69
        source["config"]["temporal_feature_std"] = [1.0] * 69
        model = temporal.PastResidualRopeAlphaStudent(69, 5, hidden_dim=8, max_alpha=0.5)
        config = dict(
            past_residual_config(),
            in_dim=69,
            temporal_feature_mean=[0.0] * 69,
            temporal_feature_std=[1.0] * 69,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong-dim.pt"
            with self.assertRaisesRegex(ValueError, "in_dim"):
                temporal.save_past_residual_checkpoint(path, model, config, source)
            self.assertFalse(path.exists())

    def test_temporal_model_has_single_layer_architecture_and_dimension_attrs(self):
        model = temporal.TemporalRopeAlphaStudent(70, 15, hidden_dim=16, max_alpha=0.5)

        self.assertEqual((model.in_dim, model.out_dim, model.hidden_dim), (70, 15, 16))
        self.assertEqual(model.max_alpha, 0.5)
        self.assertEqual([type(layer) for layer in model.encoder], [nn.Linear, nn.ReLU, nn.LayerNorm])
        self.assertTrue(model.gru.batch_first)
        self.assertEqual(model.gru.num_layers, 1)
        self.assertEqual(model.gru.input_size, 16)
        self.assertEqual(model.gru.hidden_size, 16)
        torch.testing.assert_close(model.head.weight, torch.zeros_like(model.head.weight))
        torch.testing.assert_close(model.head.bias, torch.zeros_like(model.head.bias))

    def test_temporal_zero_head_equals_framewise_alpha(self):
        model = temporal.TemporalRopeAlphaStudent(
            in_dim=70, out_dim=15, hidden_dim=16, max_alpha=0.5
        )
        windows = torch.randn(3, 4, 70)
        lengths = torch.tensor([4, 2, 1])
        base = torch.tensor([[0.1] * 15, [-0.2] * 15, [0.0] * 15])

        torch.testing.assert_close(model(windows, lengths, base), base)

    def test_temporal_output_is_bounded_and_finite(self):
        model = temporal.TemporalRopeAlphaStudent(70, 15, 8, 0.5)
        with torch.no_grad():
            model.head.bias.fill_(100)
        out = model(
            torch.zeros(2, 1, 70),
            torch.ones(2, dtype=torch.long),
            torch.tensor([[0.5] * 15, [-0.5] * 15]),
        )

        self.assertEqual(out.shape, (2, 15))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(float(out.abs().max()), 0.5)

    def test_bounded_transform_promotes_low_precision_base_alpha(self):
        model = temporal.TemporalRopeAlphaStudent(4, 1, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            model.head.bias.fill_(-100)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                out = model(
                    torch.zeros(1, 1, 4),
                    torch.ones(1, dtype=torch.long),
                    torch.tensor([[0.5]], dtype=dtype),
                )
                self.assertEqual(out.dtype, torch.float32)
                self.assertTrue(torch.isfinite(out).all())
                self.assertLess(float(out.item()), 0.0)

    def test_packed_lengths_make_padded_slots_irrelevant(self):
        torch.manual_seed(3)
        model = temporal.TemporalRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            model.head.weight.normal_()
        windows = torch.randn(2, 4, 4)
        lengths = torch.tensor([2, 4])
        base = torch.zeros(2, 2)
        expected = model(windows, lengths, base)

        changed_padding = windows.clone()
        changed_padding[0, 2:] = 1_000_000
        actual = model(changed_padding, lengths, base)

        torch.testing.assert_close(actual[0], expected[0])

    def test_temporal_model_rejects_invalid_dimensions_and_lengths(self):
        for max_alpha in (0.0, -0.5):
            with self.subTest(max_alpha=max_alpha), self.assertRaises(ValueError):
                temporal.TemporalRopeAlphaStudent(4, 2, 6, max_alpha)

        model = temporal.TemporalRopeAlphaStudent(4, 2, 6, 0.5)
        windows = torch.zeros(2, 3, 4)
        base = torch.zeros(2, 2)
        for lengths in (torch.tensor([0, 1]), torch.tensor([4, 1])):
            with self.subTest(lengths=lengths.tolist()), self.assertRaises(ValueError):
                model(windows, lengths, base)
        with self.assertRaisesRegex(ValueError, "integer"):
            model(windows, torch.tensor([2.5, 1.0]), base)
        with self.assertRaisesRegex(ValueError, "floating"):
            model(windows, torch.tensor([3, 1]), base.to(torch.int64))
        with self.assertRaises(ValueError):
            model(windows[:, :, :3], torch.tensor([3, 1]), base)
        with self.assertRaises(ValueError):
            model(windows, torch.tensor([3, 1]), base[:, :1])

    def test_student_from_payload_reconstructs_eval_model(self):
        original, payload = framewise_payload()
        loaded, config = alpha_student.student_from_payload(payload, "cpu")

        self.assertEqual(config["tag"], "framewise")
        self.assertFalse(loaded.training)
        features = torch.randn(3, 65)
        torch.testing.assert_close(loaded(features), original(features))

    def test_temporal_checkpoint_roundtrip_is_self_contained(self):
        torch.manual_seed(5)
        model = temporal.TemporalRopeAlphaStudent(70, 2, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            model.head.weight.normal_()
            model.head.bias.copy_(torch.tensor([0.2, -0.1]))
        config = dict(temporal_checkpoint_config(70, 2, 6), tag="temporal")
        _, payload = framewise_payload()
        framewise_state = {key: value.clone() for key, value in payload["model_state"].items()}
        framewise_config = copy.deepcopy(payload["config"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "temporal.pt"
            temporal.save_temporal_checkpoint(path, model, config, payload)

            self.assertTrue(path.exists())
            self.assertEqual(payload["config"], framewise_config)
            for key, value in framewise_state.items():
                torch.testing.assert_close(payload["model_state"][key], value)

            payload["config"]["tag"] = "mutated-after-save"
            with torch.no_grad():
                next(iter(payload["model_state"].values())).add_(100)

            raw = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(raw), {"model_state", "config", "framewise"})
            self.assertEqual(set(raw["framewise"]), {"model_state", "config"})
            self.assertEqual(raw["framewise"]["config"], framewise_config)
            for key, value in framewise_state.items():
                torch.testing.assert_close(raw["framewise"]["model_state"][key], value)

            loaded, loaded_config, framewise, loaded_framewise_config = (
                temporal.load_temporal_checkpoint(path, "cpu")
            )

        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_framewise_config, framewise_config)
        self.assertFalse(loaded.training)
        self.assertFalse(framewise.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in framewise.parameters()))
        windows = torch.randn(3, 4, 70)
        lengths = torch.tensor([4, 2, 1])
        base = torch.zeros(3, 2)
        torch.testing.assert_close(loaded(windows, lengths, base), model(windows, lengths, base))
        for key, value in framewise_state.items():
            torch.testing.assert_close(framewise.state_dict()[key], value)

    def test_temporal_alpha_uses_nested_framewise_and_authoritative_window_config(self):
        torch.manual_seed(13)
        cache = toy_cache(
            ["A/0001", "A/0000", "A/0003", "B/0000"],
            np.linspace(0.1, 0.8, 20, dtype=np.float32).reshape(4, 5),
        )
        cache["base_rope_norm"][:] = 0.3
        framewise = RopeAlphaStudent(out_dim=5, hidden_dim=7, max_alpha=0.5, in_dim=65)
        temporal_model = temporal.TemporalRopeAlphaStudent(70, 5, hidden_dim=6, max_alpha=0.5)
        with torch.no_grad():
            for parameter in framewise.parameters():
                parameter.uniform_(-0.1, 0.1)
            for parameter in temporal_model.parameters():
                parameter.uniform_(-0.1, 0.1)

        legacy_framewise_std = np.linspace(0.5, 1.5, 65)
        legacy_framewise_std[60:65] = 1e-4
        framewise_config = {
            "in_dim": 65,
            "out_dim": 5,
            "hidden_dim": 7,
            "max_alpha": 0.5,
            "image_feature_dim": 0,
            "action_space": "mult5",
            "gate_threshold": 0.1,
            "feature_mean": np.linspace(-0.2, 0.2, 65).tolist(),
            "feature_std": legacy_framewise_std.tolist(),
        }
        legacy_temporal_std = np.linspace(0.5, 1.5, 70)
        legacy_temporal_std[60:65] = 1e-4
        config = {
            "model_type": "causal_gru",
            "in_dim": 70,
            "out_dim": 5,
            "hidden_dim": 6,
            "max_alpha": 0.5,
            "action_space": "mult5",
            "gate_threshold": 0.1,
            "history_length": 3,
            "raw_frame_step": 1,
            "history_step": 1,
            "temporal_feature_mean": np.linspace(-0.3, 0.3, 70).tolist(),
            "temporal_feature_std": legacy_temporal_std.tolist(),
        }
        payload = {"model_state": framewise.state_dict(), "config": framewise_config}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "temporal.pt"
            temporal.save_temporal_checkpoint(checkpoint, temporal_model, config, payload)
            actual, actual_config = temporal.temporal_alpha(cache, checkpoint, "cpu")
            legacy, legacy_config = temporal._temporal_alpha_v1(
                cache, checkpoint, "cpu"
            )

        frame_features = alpha_student.normalize_features(
            features_from_cache(cache),
            np.asarray(framewise_config["feature_mean"], dtype=np.float32),
            np.asarray(framewise_config["feature_std"], dtype=np.float32),
        )
        with torch.no_grad():
            base_alpha = framewise(torch.from_numpy(frame_features))
        features70 = temporal_features(cache, raw_frame_step=1)
        features70 = (
            features70 - np.asarray(config["temporal_feature_mean"], dtype=np.float32)
        ) / np.asarray(config["temporal_feature_std"], dtype=np.float32)
        windows, _, lengths = build_causal_windows(
            cache["sample_id"], features70, 3, 1, 1
        )
        with torch.no_grad():
            expected = temporal_model(
                torch.from_numpy(windows), torch.from_numpy(lengths), base_alpha
            ).numpy()

        self.assertEqual(actual_config, config)
        self.assertEqual(legacy_config, config)
        np.testing.assert_array_equal(actual, legacy)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_v2_temporal_alpha_dispatch_and_disable_history_return_same_cadence_k1(self):
        cache = toy_cache(
            [f"A/{frame:04d}" for frame in range(9)],
            np.repeat(np.arange(9, dtype=np.float32)[:, None], 5, axis=1),
        )
        cache["base_rope_norm"][:] = 0.25
        _, base_payload = k1_payload()
        base_payload["framewise"]["model_state"]["net.4.bias"].fill_(0.2)
        residual = temporal.PastResidualRopeAlphaStudent(
            70, 5, hidden_dim=8, max_alpha=0.5
        )
        with torch.no_grad():
            residual.head.bias.fill_(0.4)
        config = past_residual_config()

        framewise, framewise_config = alpha_student.student_from_payload(
            base_payload["framewise"], "cpu"
        )
        base_model = temporal.TemporalRopeAlphaStudent(70, 5, 6, 0.5)
        base_model.load_state_dict(base_payload["model_state"])
        frame_features = alpha_student.normalize_features(
            features_from_cache(cache),
            np.asarray(framewise_config["feature_mean"], dtype=np.float32),
            np.asarray(framewise_config["feature_std"], dtype=np.float32),
        )
        features70 = temporal_features(cache, raw_frame_step=1, delta_step=4)
        base_windows, _, base_lengths = build_causal_windows(
            cache["sample_id"], features70, 1, 1, 4
        )
        with torch.no_grad():
            frame_alpha = framewise(torch.from_numpy(frame_features))
            expected_k1 = base_model(
                torch.from_numpy(base_windows),
                torch.from_numpy(base_lengths),
                frame_alpha,
            ).numpy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2.pt"
            temporal.save_past_residual_checkpoint(
                path, residual, config, base_payload
            )
            disabled, disabled_config = temporal.temporal_alpha(
                cache, path, "cpu", disable_history=True
            )
            enabled, enabled_config = temporal.temporal_alpha(cache, path, "cpu")

        np.testing.assert_array_equal(disabled, expected_k1)
        self.assertEqual(disabled_config, config)
        self.assertEqual(enabled_config, config)
        np.testing.assert_array_equal(enabled[0], expected_k1[0])
        self.assertFalse(np.array_equal(enabled[-1], expected_k1[-1]))

    def test_temporal_checkpoint_save_requires_every_schema_key(self):
        model = temporal.TemporalRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        config = temporal_checkpoint_config()
        _, payload = framewise_payload()
        required = (
            "in_dim",
            "out_dim",
            "hidden_dim",
            "max_alpha",
            "temporal_feature_mean",
            "temporal_feature_std",
        )

        with tempfile.TemporaryDirectory() as tmp:
            for key in required:
                bad_config = {name: value for name, value in config.items() if name != key}
                path = Path(tmp) / f"missing-{key}.pt"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    temporal.save_temporal_checkpoint(path, model, bad_config, payload)
                self.assertFalse(path.exists())

    def test_temporal_checkpoint_save_rejects_invalid_schema_values(self):
        model = temporal.TemporalRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        config = temporal_checkpoint_config()
        _, payload = framewise_payload()
        cases = (
            (dict(config, in_dim=4.0), "in_dim"),
            (dict(config, out_dim=0), "out_dim"),
            (dict(config, hidden_dim=-1), "hidden_dim"),
            (dict(config, max_alpha=float("nan")), "max_alpha"),
            (temporal_checkpoint_config(in_dim=5), "in_dim"),
            (temporal_checkpoint_config(out_dim=3), "out_dim"),
            (temporal_checkpoint_config(hidden_dim=7), "hidden_dim"),
            (temporal_checkpoint_config(max_alpha=0.4), "max_alpha"),
            (dict(config, temporal_feature_mean=[0.0] * 3), "temporal_feature_mean"),
            (dict(config, temporal_feature_mean=[0.0, 0.0, float("nan"), 0.0]), "temporal_feature_mean"),
            (dict(config, temporal_feature_std=[1.0, 1.0, 0.0, 1.0]), "temporal_feature_std"),
            (dict(config, temporal_feature_std=[1.0, 1.0, float("inf"), 1.0]), "temporal_feature_std"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            for index, (config, message) in enumerate(cases):
                path = Path(tmp) / f"invalid-{index}.pt"
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    temporal.save_temporal_checkpoint(path, model, config, payload)
                self.assertFalse(path.exists())

    def test_temporal_checkpoint_rejects_incompatible_framewise_on_save(self):
        model = temporal.TemporalRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        config = temporal_checkpoint_config()
        _, payload = framewise_payload()

        with tempfile.TemporaryDirectory() as tmp:
            for key, value in (("out_dim", 3), ("max_alpha", 0.4)):
                bad_payload = dict(payload, config=dict(payload["config"], **{key: value}))
                path = Path(tmp) / f"bad-framewise-{key}.pt"
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    temporal.save_temporal_checkpoint(path, model, config, bad_payload)
                self.assertFalse(path.exists())

            legacy_payload = dict(payload, config=dict(payload["config"]))
            legacy_payload["config"].pop("max_alpha")
            path = Path(tmp) / "legacy-framewise.pt"
            temporal.save_temporal_checkpoint(path, model, config, legacy_payload)
            _, _, framewise, framewise_config = temporal.load_temporal_checkpoint(path, "cpu")
            self.assertNotIn("max_alpha", framewise_config)
            self.assertEqual(framewise.max_alpha, 0.5)

    def test_temporal_checkpoint_revalidates_full_schema_on_load(self):
        model = temporal.TemporalRopeAlphaStudent(4, 2, hidden_dim=6, max_alpha=0.5)
        config = temporal_checkpoint_config()
        _, framewise = framewise_payload()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "temporal.pt"
            temporal.save_temporal_checkpoint(path, model, config, framewise)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            cases = (
                (dict(payload, config={key: value for key, value in config.items() if key != "in_dim"}), "in_dim"),
                (dict(payload, config=dict(config, in_dim=4.0)), "in_dim"),
                (dict(payload, config=dict(config, out_dim=3)), "out_dim"),
                (dict(payload, config=dict(config, max_alpha=float("nan"))), "max_alpha"),
                (dict(payload, config=dict(config, temporal_feature_mean=[0.0] * 3)), "temporal_feature_mean"),
                (dict(payload, config=dict(config, temporal_feature_mean=[0.0, 0.0, float("nan"), 0.0])), "temporal_feature_mean"),
                (dict(payload, config=dict(config, temporal_feature_std=[1.0, 0.0, 1.0, 1.0])), "temporal_feature_std"),
                (
                    dict(
                        payload,
                        framewise=dict(
                            payload["framewise"],
                            config=dict(payload["framewise"]["config"], out_dim=3),
                        ),
                    ),
                    "out_dim",
                ),
                (
                    dict(
                        payload,
                        framewise=dict(
                            payload["framewise"],
                            config=dict(payload["framewise"]["config"], max_alpha=0.4),
                        ),
                    ),
                    "max_alpha",
                ),
            )
            for bad_payload, message in cases:
                torch.save(bad_payload, path)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    temporal.load_temporal_checkpoint(path, "cpu")


if __name__ == "__main__":
    unittest.main()
