import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "rope_head" / "extract_feature_cache.py"
    spec = importlib.util.spec_from_file_location("extract_feature_cache", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeCandidate:
    sample_index: int
    bbox_index: int = 0


class FakeBackbone(nn.Module):
    """Emits a [B, C, H, W] map whose value encodes the input's id channel."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        batch = ids.shape[0]
        base = ids.reshape(batch, 1, 1, 1).float()
        grid = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 3)
        return base + grid.expand(batch, 4, 2, 3) * 0.0 + grid.repeat(1, 4, 1, 1) * 0.01


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()

    def forward(self, batch):
        self.backbone(batch["img"])
        return {}


def fake_loader(indices, batch_size=2):
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        yield {
            "img": torch.tensor(chunk),
            "candidate_index": torch.tensor(chunk),
        }


class CandidateSelectionTest(unittest.TestCase):
    def test_first_candidate_per_sample(self):
        script = load_script()
        candidates = [FakeCandidate(0), FakeCandidate(1), FakeCandidate(1, bbox_index=1), FakeCandidate(2)]
        selected = script.first_candidate_per_sample(3, candidates)
        self.assertEqual([c.sample_index for c in selected], [0, 1, 2])
        self.assertEqual(selected[1].bbox_index, 0)  # first occurrence wins

    def test_missing_sample_raises(self):
        script = load_script()
        with self.assertRaises(ValueError):
            script.first_candidate_per_sample(3, [FakeCandidate(0), FakeCandidate(2)])


class PoolingTest(unittest.TestCase):
    def test_mean_pool_channel_first(self):
        script = load_script()
        feat = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).reshape(2, 4, 2, 3)
        pooled, tokens = script.pool_feature_map(feat, "mean")
        self.assertEqual(pooled.shape, (2, 4))
        self.assertEqual(tokens.shape, (2, 6, 4))
        torch.testing.assert_close(pooled, feat.flatten(2).mean(dim=2))

    def test_meanmax_pool_and_token_layout(self):
        script = load_script()
        feat = torch.randn(3, 5, 4, 2)
        pooled, _ = script.pool_feature_map(feat, "meanmax")
        self.assertEqual(pooled.shape, (3, 10))

    def test_token_layout_input(self):
        script = load_script()
        feat = torch.randn(2, 7, 8)  # [B, T, C]
        pooled, tokens = script.pool_feature_map(feat, "mean")
        self.assertEqual(pooled.shape, (2, 8))
        self.assertEqual(tokens.shape, (2, 7, 8))

    def test_tuple_output_unwrapped(self):
        script = load_script()
        feat = (torch.randn(2, 4, 2, 3), "aux")
        pooled, _ = script.pool_feature_map(feat, "mean")
        self.assertEqual(pooled.shape, (2, 4))

    def test_wilor_backbone_tuple_selects_image_feature(self):
        script = load_script()
        img_feat = torch.randn(2, 8, 3, 4)
        output = (
            {"hand_pose": torch.randn(2, 96)},
            torch.randn(2, 3),
            {"cam": torch.randn(2, 3)},
            img_feat,
        )
        pooled, tokens = script.pool_feature_map(output, "mean")
        self.assertEqual(pooled.shape, (2, 8))
        self.assertEqual(tokens.shape, (2, 12, 8))
        torch.testing.assert_close(pooled, img_feat.flatten(2).mean(dim=2))

    def test_dict_feature_key_supported(self):
        script = load_script()
        feat = torch.randn(2, 5, 6)
        pooled, tokens = script.pool_feature_map({"vit_out": feat}, "mean")
        self.assertEqual(pooled.shape, (2, 6))
        self.assertEqual(tokens.shape, (2, 5, 6))

    def test_bad_pooling_raises(self):
        script = load_script()
        with self.assertRaises(ValueError):
            script.pool_feature_map(torch.randn(1, 2, 2, 2), "attention")


class RunExtractionTest(unittest.TestCase):
    def test_features_land_on_candidate_rows(self):
        script = load_script()
        model = FakeModel()
        # shuffled batch order must still land rows by candidate_index
        indices = [3, 1, 0, 2]
        features, tokens = script.run_extraction(model, fake_loader(indices), 4, "cpu")
        self.assertEqual(features.shape, (4, 4))
        self.assertIsNone(tokens)
        # FakeBackbone encodes the sample id into the feature magnitude
        for sample_index in range(4):
            self.assertAlmostEqual(
                float(features[sample_index].mean()), sample_index + 0.025, places=5
            )

    def test_save_tokens_shapes(self):
        script = load_script()
        model = FakeModel()
        features, tokens = script.run_extraction(model, fake_loader([0, 1]), 2, "cpu", save_tokens=True)
        self.assertEqual(features.shape, (2, 4))
        self.assertEqual(tokens.shape, (2, 6, 4))
        self.assertEqual(tokens.dtype, np.float16)

    def test_missing_rows_raise(self):
        script = load_script()
        model = FakeModel()
        with self.assertRaises(RuntimeError):
            script.run_extraction(model, fake_loader([0, 1]), 3, "cpu")

    def test_hookless_backbone_raises(self):
        script = load_script()

        class NoCallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = FakeBackbone()

            def forward(self, batch):
                return {}  # never calls backbone

        with self.assertRaises(RuntimeError):
            script.run_extraction(NoCallModel(), fake_loader([0]), 1, "cpu")


class WriteCacheTest(unittest.TestCase):
    def test_roundtrip(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature_cache.npz"
            features = np.random.default_rng(3).normal(size=(4, 8)).astype(np.float32)
            script.write_feature_cache(path, [f"{i:08d}" for i in range(4)], features, None,
                                       {"backend": "wilor", "pooling": "mean"})
            with np.load(path) as data:
                self.assertEqual(data["sample_id"].tolist(), ["00000000", "00000001", "00000002", "00000003"])
                np.testing.assert_allclose(data["features"], features)
                meta = json.loads(str(data["meta_json"]))
                self.assertEqual(meta["backend"], "wilor")
                self.assertNotIn("tokens", data.files)

    def test_length_mismatch_raises(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                script.write_feature_cache(Path(tmp) / "x.npz", ["a"], np.zeros((2, 4), dtype=np.float32), None, {})


if __name__ == "__main__":
    unittest.main()
