import copy
import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ropetrack.refine import alpha_student, temporal
from ropetrack.refine.alpha_student import RopeAlphaStudent, features_from_cache
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


class TemporalModelTest(unittest.TestCase):
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
