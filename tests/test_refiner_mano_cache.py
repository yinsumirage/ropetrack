import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ropetrack.datasets.hand_pose import BBoxItem, FreiHandSample, iter_hand_pose_samples
from ropetrack.eval.config import build_run_args
from ropetrack.eval.pipeline import BatchHandPrediction, write_mano_cache


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RefinerManoCacheTest(unittest.TestCase):
    def test_build_run_args_propagates_output_flags(self):
        args = build_run_args(
            dataset="freihand",
            split="training",
            save_mano_cache=True,
            joint_only_output=True,
        )

        self.assertEqual(args.split, "training")
        self.assertTrue(args.save_mano_cache)
        self.assertTrue(args.joint_only_output)

    def test_iter_freihand_training_samples_uses_training_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training" / "rgb").mkdir(parents=True)
            (root / "training" / "rgb" / "00000000.jpg").write_text("")
            write_json(root / "training_K.json", [
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
            ])
            write_json(root / "training_verts.json", [
                [[0.1, 0.2, 1.0], [0.4, 0.5, 1.0]]
            ])

            sample = next(iter(iter_hand_pose_samples("freihand", root, limit=None, split="training")))

        self.assertEqual(sample.sample_id, "00000000")
        self.assertTrue(sample.image_path.as_posix().endswith("training/rgb/00000000.jpg"))
        self.assertEqual(sample.bbox_xyxy.tolist(), [10.0, 20.0, 40.0, 50.0])

    def test_ho3d_training_split_needs_train_list(self):
        # training split is supported since the HO3D v3 train pipeline
        # (experience/0040); without a train.txt the root is unusable
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list(iter_hand_pose_samples("ho3d", Path(tmp), limit=None, split="training"))

    def test_ho3d_rejects_unknown_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsupported HO3D split"):
                list(iter_hand_pose_samples("ho3d", Path(tmp), limit=None, split="nope"))

    def test_write_mano_cache_preserves_order_and_zero_fills_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = [
                FreiHandSample("00000000", Path("a.jpg"), np.zeros(4, dtype=np.float32)),
                SimpleNamespace(sample_id="00000001", is_right=False),
            ]
            hand = BatchHandPrediction(
                candidate=BBoxItem(0, 0, samples[0], np.zeros(4, dtype=np.float32), True, 1.0, "gt_bbox"),
                vertices=np.zeros((778, 3), dtype=np.float32),
                keypoints_3d=np.zeros((21, 3), dtype=np.float32),
                cam_t=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                base_global_orient=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                base_hand_pose=np.full(45, 4.0, dtype=np.float32),
                base_betas=np.full(10, 5.0, dtype=np.float32),
            )
            path = Path(tmp) / "mano_cache.npz"

            write_mano_cache(path, samples, [hand, None])

            with np.load(path) as data:
                self.assertEqual(data["sample_id"].tolist(), ["00000000", "00000001"])
                self.assertEqual(data["is_right"].tolist(), [True, False])
                np.testing.assert_allclose(data["base_global_orient"][0], [0.1, 0.2, 0.3])
                np.testing.assert_array_equal(data["base_hand_pose"][0], np.full(45, 4.0, dtype=np.float32))
                np.testing.assert_array_equal(data["base_betas"][0], np.full(10, 5.0, dtype=np.float32))
                np.testing.assert_allclose(data["base_cam_t"][0], [1.0, 2.0, 3.0])
                np.testing.assert_array_equal(data["base_hand_pose"][1], np.zeros(45, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
