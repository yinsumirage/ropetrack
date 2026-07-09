import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from ropetrack.refine.actions import apply_action_np
from ropetrack.refine.alpha_student import (
    RopeAlphaStudent,
    STUDENT_FEATURE_DIM,
    build_student_features,
    feature_stats,
    features_from_cache,
    load_student,
    normalize_features,
    save_student_checkpoint,
    student_alpha,
)

from test_apply_rope_refinement import FakeMano, toy_cache, write_toy_mano_cache


def load_trainer():
    path = ROOT / "scripts" / "rope_refiner" / "train_alpha_student.py"
    spec = importlib.util.spec_from_file_location("train_alpha_student", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_teacher(num: int = 256, seed: int = 5):
    """Teacher alpha is a simple deterministic function of the rope residual."""
    rng = np.random.default_rng(seed)
    pose = rng.normal(scale=0.3, size=(num, 45)).astype(np.float32)
    base_rope = rng.uniform(0.2, 0.8, size=(num, 5)).astype(np.float32)
    residual = rng.uniform(-0.4, 0.4, size=(num, 5)).astype(np.float32)
    input_rope = np.clip(base_rope + residual, 0.0, 1.0).astype(np.float32)
    valid = np.ones((num, 5), dtype=bool)
    cache = {
        "sample_id": np.asarray([f"{i:08d}" for i in range(num)]),
        "base_hand_pose": pose,
        "base_rope_norm": base_rope,
        "input_rope_norm": input_rope,
        "rope_valid": valid,
    }
    teacher_alpha = np.clip((input_rope - base_rope) * 1.0, -0.4, 0.4).astype(np.float32)
    return cache, teacher_alpha


class ModelAndFeatureTest(unittest.TestCase):
    def test_zero_init_predicts_no_correction(self):
        model = RopeAlphaStudent(out_dim=15)
        out = model(torch.randn(3, STUDENT_FEATURE_DIM))
        self.assertTrue(torch.allclose(out, torch.zeros(3, 15)))

    def test_alpha_is_bounded(self):
        model = RopeAlphaStudent(out_dim=5, max_alpha=0.5)
        for layer in model.net:
            if hasattr(layer, "weight"):
                torch.nn.init.normal_(layer.weight, std=5.0)
        out = model(torch.randn(64, STUDENT_FEATURE_DIM))
        self.assertLessEqual(float(out.abs().max()), 0.5)

    def test_feature_layout_and_masking(self):
        pose = np.zeros((2, 45), dtype=np.float32)
        base = np.full((2, 5), 0.6, dtype=np.float32)
        inp = np.full((2, 5), 0.9, dtype=np.float32)
        valid = np.ones((2, 5), dtype=bool)
        valid[0, 2] = False
        features = build_student_features(pose, base, inp, valid)
        self.assertEqual(features.shape, (2, STUDENT_FEATURE_DIM))
        self.assertAlmostEqual(float(features[1, 45 + 10 + 2]), 0.3, places=6)  # residual col
        self.assertEqual(float(features[0, 45 + 2]), 0.0)       # base rope masked
        self.assertEqual(float(features[0, 45 + 5 + 2]), 0.0)   # input rope masked
        self.assertEqual(float(features[0, 45 + 10 + 2]), 0.0)  # residual masked
        self.assertEqual(float(features[0, 45 + 15 + 2]), 0.0)  # valid flag

    def test_normalization_stats(self):
        rng = np.random.default_rng(3)
        features = rng.normal(loc=2.0, scale=3.0, size=(100, STUDENT_FEATURE_DIM)).astype(np.float32)
        mean, std = feature_stats(features)
        normed = normalize_features(features, mean, std)
        self.assertLess(float(np.abs(normed.mean(axis=0)).max()), 1e-4)
        self.assertLess(float(np.abs(normed.std(axis=0) - 1.0).max()), 1e-3)

    def test_checkpoint_roundtrip(self):
        cache, _ = synthetic_teacher(num=8)
        model = RopeAlphaStudent(out_dim=5, hidden_dim=32)
        features = features_from_cache(cache)
        mean, std = feature_stats(features)
        config = {
            "out_dim": 5, "hidden_dim": 32, "max_alpha": 0.5, "action_space": "mult5",
            "gate_threshold": 0.1, "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.pt"
            save_student_checkpoint(path, model, config)
            loaded, loaded_config = load_student(path, "cpu")
            self.assertEqual(loaded_config["action_space"], "mult5")
            alpha, config_out = student_alpha(cache, path, "cpu")
            self.assertEqual(alpha.shape, (8, 5))
            with torch.no_grad():
                expected = loaded(torch.from_numpy(normalize_features(features, mean, std))).numpy()
            np.testing.assert_allclose(alpha, expected, atol=1e-6)
            self.assertEqual(config_out["gate_threshold"], 0.1)


class TrainStudentTest(unittest.TestCase):
    def _train(self, trainer, cache, teacher_alpha, tmp, **overrides):
        params = dict(
            gate_threshold=0.1,
            hidden_dim=64,
            lr=3e-3,
            batch_size=128,
            max_epochs=60,
            patience=15,
            val_frac=0.2,
            seed=0,
            aug_noise_std=0.0,
            aug_dropout=0.0,
            device="cpu",
        )
        params.update(overrides)
        return trainer.train_student(cache, teacher_alpha, "mult5", Path(tmp) / "out", **params)

    def test_learns_residual_mapping_and_beats_zero_baseline(self):
        trainer = load_trainer()
        cache, teacher_alpha = synthetic_teacher()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._train(trainer, cache, teacher_alpha, tmp)
            self.assertTrue(summary["beats_zero_baseline"])
            self.assertLess(summary["best_val_loss"], 0.3 * summary["zero_baseline_val_l1"])
            out_dir = Path(tmp) / "out"
            self.assertTrue((out_dir / "student.pt").exists())
            log = json.loads((out_dir / "train_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log["summary"]["config"]["action_space"], "mult5")

    def test_learns_with_noise_augmentation(self):
        trainer = load_trainer()
        cache, teacher_alpha = synthetic_teacher()
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._train(
                trainer,
                cache,
                teacher_alpha,
                tmp,
                aug_noise_std=0.05,
                aug_dropout=0.1,
                aug_bias_std=0.02,
                aug_bias_fixed=0.01,
                aug_scale_range=0.05,
            )
            self.assertTrue(summary["beats_zero_baseline"])
            config = summary["config"]
            self.assertAlmostEqual(config["aug_bias_std"], 0.02)
            self.assertAlmostEqual(config["aug_bias_fixed"], 0.01)
            self.assertAlmostEqual(config["aug_scale_range"], 0.05)

    def test_bias_scale_cli_parse(self):
        trainer = load_trainer()
        args = trainer.parse_args([
            "--teacher-dir", "teacher",
            "--action-space", "mult5",
            "--out-dir", "out",
            "--aug-bias-std", "0.05",
            "--aug-bias-fixed", "-0.05",
            "--aug-scale-range", "0.1",
        ])
        self.assertAlmostEqual(args.aug_bias_std, 0.05)
        self.assertAlmostEqual(args.aug_bias_fixed, -0.05)
        self.assertAlmostEqual(args.aug_scale_range, 0.1)

    def test_shuffled_rope_control_destroys_learning(self):
        trainer = load_trainer()
        cache_a, teacher_alpha = synthetic_teacher()
        cache_b, _ = synthetic_teacher()
        with tempfile.TemporaryDirectory() as tmp:
            normal = self._train(trainer, cache_a, teacher_alpha, tmp)
        with tempfile.TemporaryDirectory() as tmp:
            shuffled = self._train(trainer, cache_b, teacher_alpha, tmp, shuffle_rope=True)
        self.assertLess(normal["best_val_loss"], 0.3 * normal["zero_baseline_val_l1"])
        self.assertGreater(shuffled["best_val_loss"], 0.6 * shuffled["zero_baseline_val_l1"])
        self.assertGreater(shuffled["best_val_loss"], 2.0 * normal["best_val_loss"])

    def test_alpha_dim_mismatch_raises(self):
        trainer = load_trainer()
        cache, teacher_alpha = synthetic_teacher()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                trainer.train_student(cache, np.zeros((len(teacher_alpha), 15), dtype=np.float32), "mult5", Path(tmp), gate_threshold=0.1, device="cpu")

    def test_rope_loss_requires_mano_cache(self):
        trainer = load_trainer()
        cache, teacher_alpha = synthetic_teacher()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._train(trainer, cache, teacher_alpha, tmp, rope_loss_weight=1.0)

    def test_rope_consistency_loss_smoke(self):
        trainer = load_trainer()
        num = 32
        cache = toy_cache(num, 0.8, 1.1)
        rng = np.random.default_rng(9)
        teacher_alpha = rng.uniform(-0.2, 0.2, size=(num, 5)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            mano_cache = Path(tmp) / "mano_cache.npz"
            write_toy_mano_cache(mano_cache, num)
            summary = trainer.train_student(
                cache,
                teacher_alpha,
                "mult5",
                Path(tmp) / "out",
                gate_threshold=0.1,
                hidden_dim=32,
                batch_size=16,
                max_epochs=3,
                patience=3,
                val_frac=0.25,
                seed=0,
                aug_noise_std=0.0,
                aug_dropout=0.0,
                rope_loss_weight=1.0,
                mano_cache=mano_cache,
                device="cpu",
                mano_module=FakeMano(),
            )
            self.assertEqual(summary["epochs_run"], 3)
            self.assertIn("best_val_loss", summary)


class MultiTeacherTest(unittest.TestCase):
    def _write_teacher_dir(self, root: Path, name: str, num: int, seed: int) -> Path:
        cache, teacher_alpha = synthetic_teacher(num=num, seed=seed)
        teacher_dir = root / name
        teacher_dir.mkdir(parents=True)
        np.savez(teacher_dir / "refiner_eval_cache.npz", **cache)
        np.save(teacher_dir / "alpha.npy", teacher_alpha)
        return teacher_dir

    def test_merges_multiple_teachers(self):
        trainer = load_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir_a = self._write_teacher_dir(root, "freihand_teacher", 64, seed=5)
            dir_b = self._write_teacher_dir(root, "ho3d_teacher", 32, seed=6)
            cache, alpha, sources = trainer.load_teacher_dirs([dir_a, dir_b])
            self.assertEqual(len(cache["sample_id"]), 96)
            self.assertEqual(alpha.shape, (96, 5))
            self.assertEqual([s["num_samples"] for s in sources], [64, 32])
            ids = [str(sid) for sid in cache["sample_id"]]
            self.assertTrue(ids[0].startswith("freihand_teacher/"))
            self.assertTrue(ids[64].startswith("ho3d_teacher/"))
            # merged set still trains
            summary = trainer.train_student(
                cache, alpha, "mult5", root / "out",
                gate_threshold=0.1, hidden_dim=64, lr=3e-3, batch_size=64,
                max_epochs=40, patience=10, val_frac=0.2, seed=0,
                aug_noise_std=0.0, aug_dropout=0.0, device="cpu", sources=sources,
            )
            self.assertTrue(summary["beats_zero_baseline"])
            self.assertEqual([s["num_samples"] for s in summary["config"]["sources"]], [64, 32])

    def test_single_dir_behavior_unchanged(self):
        trainer = load_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            teacher_dir = self._write_teacher_dir(Path(tmp), "only_teacher", 16, seed=7)
            cache, alpha, sources = trainer.load_teacher_dirs([teacher_dir])
            self.assertEqual(str(cache["sample_id"][0]), "00000000")  # no prefix for single source
            self.assertEqual(len(sources), 1)

    def test_alpha_dim_mismatch_across_dirs_raises(self):
        trainer = load_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir_a = self._write_teacher_dir(root, "a", 16, seed=5)
            cache_b, _ = synthetic_teacher(num=16, seed=6)
            dir_b = root / "b"
            dir_b.mkdir()
            np.savez(dir_b / "refiner_eval_cache.npz", **cache_b)
            np.save(dir_b / "alpha.npy", np.zeros((16, 15), dtype=np.float32))
            with self.assertRaises(ValueError):
                trainer.load_teacher_dirs([dir_a, dir_b])

    def test_rope_loss_with_multiple_dirs_raises_in_main(self):
        trainer = load_trainer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dir_a = self._write_teacher_dir(root, "a", 16, seed=5)
            dir_b = self._write_teacher_dir(root, "b", 16, seed=6)
            with self.assertRaises(ValueError):
                trainer.main([
                    "--teacher-dir", str(dir_a), str(dir_b),
                    "--action-space", "mult5",
                    "--out-dir", str(root / "out"),
                    "--rope-loss-weight", "0.5",
                    "--device", "cpu",
                ])


class ImageFeatureStudentTest(unittest.TestCase):
    """P3 v0 head: image features concatenated to the 65-d rope/pose features."""

    @staticmethod
    def image_dependent_teacher(num: int = 512, feat_dim: int = 16, seed: int = 21):
        """Teacher alpha depends ONLY on the image feature — unlearnable without it."""
        rng = np.random.default_rng(seed)
        cache, _ = synthetic_teacher(num=num, seed=seed)
        image_features = rng.normal(size=(num, feat_dim)).astype(np.float32)
        teacher_alpha = np.clip(0.3 * image_features[:, :5], -0.4, 0.4).astype(np.float32)
        return cache, teacher_alpha, image_features

    def _train(self, trainer, cache, teacher_alpha, tmp, image_features=None):
        return trainer.train_student(
            cache, teacher_alpha, "mult5", Path(tmp) / "out",
            gate_threshold=0.1, hidden_dim=64, lr=3e-3, batch_size=128,
            max_epochs=80, patience=20, val_frac=0.2, seed=0,
            aug_noise_std=0.0, aug_dropout=0.0, device="cpu",
            image_features=image_features,
        )

    def test_image_features_are_necessary_and_sufficient(self):
        trainer = load_trainer()
        cache_a, teacher_alpha, image_features = self.image_dependent_teacher()
        cache_b = {key: np.copy(value) for key, value in cache_a.items()}
        with tempfile.TemporaryDirectory() as tmp:
            with_feat = self._train(trainer, cache_a, teacher_alpha, tmp, image_features=image_features)
        with tempfile.TemporaryDirectory() as tmp:
            without_feat = self._train(trainer, cache_b, teacher_alpha, tmp)
        # with image features the mapping is learnable...
        self.assertLess(with_feat["best_val_loss"], 0.5 * with_feat["zero_baseline_val_l1"])
        # ...without them it stays near the predict-zero baseline
        self.assertGreater(without_feat["best_val_loss"], 0.7 * without_feat["zero_baseline_val_l1"])
        self.assertEqual(with_feat["config"]["image_feature_dim"], 16)
        self.assertEqual(with_feat["config"]["in_dim"], 65 + 16)

    def test_checkpoint_roundtrip_with_image_features(self):
        trainer = load_trainer()
        cache, teacher_alpha, image_features = self.image_dependent_teacher(num=64)
        with tempfile.TemporaryDirectory() as tmp:
            self._train(trainer, cache, teacher_alpha, tmp, image_features=image_features)
            ckpt = Path(tmp) / "out" / "student.pt"
            alpha, config = student_alpha(cache, ckpt, "cpu", image_features=image_features)
            self.assertEqual(alpha.shape, (64, 5))
            with self.assertRaises(ValueError):
                student_alpha(cache, ckpt, "cpu")  # features required
            with self.assertRaises(ValueError):
                student_alpha(cache, ckpt, "cpu", image_features=image_features[:, :8])  # wrong dim

    def test_plain_checkpoint_rejects_image_features(self):
        trainer = load_trainer()
        cache, teacher_alpha = synthetic_teacher(num=64)
        with tempfile.TemporaryDirectory() as tmp:
            self._train(trainer, cache, teacher_alpha, tmp)
            ckpt = Path(tmp) / "out" / "student.pt"
            with self.assertRaises(ValueError):
                student_alpha(cache, ckpt, "cpu", image_features=np.zeros((64, 16), dtype=np.float32))

    def test_join_image_features_reorders_and_validates(self):
        from ropetrack.refine.alpha_student import join_image_features

        feature_ids = ["b", "a", "c"]
        features = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        joined = join_image_features(["a", "b"], feature_ids, features)
        np.testing.assert_allclose(joined, [[2.0], [1.0]])
        with self.assertRaises(ValueError):
            join_image_features(["a", "z"], feature_ids, features)

    def test_row_count_mismatch_raises(self):
        trainer = load_trainer()
        cache, teacher_alpha, image_features = self.image_dependent_teacher(num=64)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._train(trainer, cache, teacher_alpha, tmp, image_features=image_features[:-1])


class StudentApplyPathTest(unittest.TestCase):
    def load_apply_script(self):
        path = ROOT / "scripts" / "rope_refiner" / "apply_rope_refinement.py"
        spec = importlib.util.spec_from_file_location("apply_rope_refinement_student_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_student_mode_requires_checkpoint(self):
        script = self.load_apply_script()
        with self.assertRaises(ValueError):
            script.main([
                "--rope-labels", "rope.jsonl", "--pred-dir", "pred", "--run-meta", "run_meta.json",
                "--mano-cache", "mano.npz", "--out-dir", "out", "--mode", "student",
            ])

    def test_gate_composition_on_student_alpha(self):
        script = self.load_apply_script()
        trainer = load_trainer()
        cache, teacher_alpha = synthetic_teacher(num=64)
        with tempfile.TemporaryDirectory() as tmp:
            trainer.train_student(
                cache, teacher_alpha, "mult5", Path(tmp),
                gate_threshold=0.1, hidden_dim=32, batch_size=32, max_epochs=10,
                patience=5, val_frac=0.2, seed=0, aug_noise_std=0.0, aug_dropout=0.0, device="cpu",
            )
            alpha, config = student_alpha(cache, Path(tmp) / "student.pt", "cpu")
            gate = script.gate_from_cache(cache, config["gate_threshold"])
            gated_alpha = alpha * script.expand_gate_to_alpha(gate, config["action_space"])
            self.assertTrue(np.all(gated_alpha[~gate] == 0.0))
            refined = apply_action_np(cache["base_hand_pose"], gated_alpha, config["action_space"])
            self.assertEqual(refined.shape, (64, 45))


if __name__ == "__main__":
    unittest.main()
