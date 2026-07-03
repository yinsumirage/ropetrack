import inspect
import unittest
from pathlib import Path


class HandPredictorLayoutTest(unittest.TestCase):
    def test_defaults_use_outer_repo_assets(self):
        from ropetrack.backends.hand_predictor import HandPredictor

        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(Path(HandPredictor.DEFAULT_MANO_RIGHT), repo / "mano_data" / "MANO_RIGHT.pkl")
        self.assertEqual(Path(HandPredictor.DEFAULT_WILOR_CKPT), repo / "pretrained_models" / "anyhand_wilor.ckpt")
        self.assertEqual(Path(HandPredictor.DEFAULT_WILOR_CFG), repo / "pretrained_models" / "model_config_wilor.yaml")
        self.assertEqual(Path(HandPredictor.DEFAULT_HAMER_CKPT), repo / "pretrained_models" / "hamer_ckpts" / "checkpoints" / "anyhand_hamer.ckpt")
        self.assertEqual(Path(HandPredictor.DEFAULT_DETECTOR_PT), repo / "pretrained_models" / "detector.pt")

    def test_backend_loaders_use_outer_submodules(self):
        from ropetrack.backends.hand_predictor import HandPredictor

        source = inspect.getsource(HandPredictor)
        self.assertIn('"third_party" / "wilor"', source)
        self.assertIn('"third_party" / "hamer"', source)
        self.assertNotIn('"third_party" / "anyhand"', source)


if __name__ == "__main__":
    unittest.main()
