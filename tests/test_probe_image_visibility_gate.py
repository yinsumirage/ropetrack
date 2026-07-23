import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "legacy" / "temporal" / "probe_image_visibility_gate.py"
    spec = importlib.util.spec_from_file_location("probe_image_visibility_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ImageVisibilityGateProbeTest(unittest.TestCase):
    def test_feature_alignment(self):
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.npz"
            np.savez(
                path,
                sample_id=np.asarray(["b", "a"]),
                features=np.asarray([[2.0, 3.0], [0.0, 1.0]], dtype=np.float32),
            )
            features = script.aligned_image_features(path, ["a", "b"])
            np.testing.assert_array_equal(features, [[0.0, 1.0], [2.0, 3.0]])

    def test_minibatch_gate_learns_separable_rows(self):
        script = load_script()
        rng = np.random.default_rng(4)
        features = rng.normal(size=(200, 4)).astype(np.float32)
        labels = features[:, 0] > 0
        model, mean, std = script.fit_gate_minibatch(
            features, labels, np.arange(200), hidden=0, seed=5, steps=200, batch_size=64
        )
        scores = script.probabilities(model, features, mean, std)
        self.assertGreater(script.auc(labels, scores), 0.95)


if __name__ == "__main__":
    unittest.main()
