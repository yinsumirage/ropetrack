import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ropetrack.datasets.hand_pose import BBoxItem, FreiHandSample, iter_hand_pose_samples
from ropetrack.eval.config import build_run_args
from ropetrack.eval.pipeline import BatchHandPrediction, write_mano_cache


ROOT = Path(__file__).resolve().parents[1]


def load_script(rel_path: str):
    path = ROOT / rel_path
    name = rel_path.replace("/", "_").replace("\\", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def freihand_joints(tip_x: float = 4.0):
    joints = [[0.0, 0.0, 1.0] for _ in range(21)]
    for joint_id in range(1, 5):
        joints[joint_id] = [float(joint_id), 0.0, 1.0]
    joints[4] = [tip_x, 0.0, 1.0]
    return joints


class RefinerManoCacheTest(unittest.TestCase):
    def test_build_run_args_propagates_split_and_mano_cache_flag(self):
        args = build_run_args(dataset="freihand", split="training", save_mano_cache=True)

        self.assertEqual(args.split, "training")
        self.assertTrue(args.save_mano_cache)

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

    def test_build_cache_uses_mano_cache_aligned_by_sample_id(self):
        builder = load_script("scripts/rope_refiner/build_freihand_refiner_cache.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "freihand"
            pred_dir = Path(tmp) / "pred"
            rope_labels = Path(tmp) / "rope.jsonl"
            run_meta = Path(tmp) / "run_meta.json"
            mano_cache = Path(tmp) / "mano_cache.npz"
            output = Path(tmp) / "cache.npz"
            pred_dir.mkdir()

            write_json(root / "training_mano.json", [
                [[float(i) for i in range(61)]],
                [[float(i + 100) for i in range(61)]],
            ])
            rows = []
            for sid in ("00000000", "00000001"):
                rows.append({
                    "sample_id": sid,
                    "dataset": "freihand",
                    "rope_norm": [1.0, None, None, None, None],
                    "rope_chain_m": [4.0, None, None, None, None],
                    "rope_valid": [True, False, False, False, False],
                    "normalization": {"fist_ratio": 0.5},
                })
            rope_labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            write_json(pred_dir / "pred.json", [
                [freihand_joints(tip_x=3.0), freihand_joints(tip_x=2.0)],
                [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
            ])
            write_json(run_meta, {"sample_order": ["00000001", "00000000"]})
            np.savez(
                mano_cache,
                sample_id=np.asarray(["00000000", "00000001"]),
                base_hand_pose=np.stack([
                    np.full(45, 7.0, dtype=np.float32),
                    np.full(45, 9.0, dtype=np.float32),
                ]),
            )

            builder.build_cache(
                root,
                rope_labels,
                pred_dir,
                run_meta,
                output,
                split="training",
                base_hand_pose_source="mano_cache",
                base_mano_cache=mano_cache,
            )

            with np.load(output) as data:
                self.assertEqual(data["sample_id"].tolist(), ["00000001", "00000000"])
                np.testing.assert_array_equal(data["base_hand_pose"][0], np.full(45, 9.0, dtype=np.float32))
                np.testing.assert_array_equal(data["base_hand_pose"][1], np.full(45, 7.0, dtype=np.float32))
                self.assertNotEqual(float(data["base_hand_pose"][0, 0]), float(data["target_hand_pose"][0, 0]))

    def test_write_mano_cache_preserves_order_and_zero_fills_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = [
                FreiHandSample("00000000", Path("a.jpg"), np.zeros(4, dtype=np.float32)),
                FreiHandSample("00000001", Path("b.jpg"), np.zeros(4, dtype=np.float32)),
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
                np.testing.assert_allclose(data["base_global_orient"][0], [0.1, 0.2, 0.3])
                np.testing.assert_array_equal(data["base_hand_pose"][0], np.full(45, 4.0, dtype=np.float32))
                np.testing.assert_array_equal(data["base_betas"][0], np.full(10, 5.0, dtype=np.float32))
                np.testing.assert_allclose(data["base_cam_t"][0], [1.0, 2.0, 3.0])
                np.testing.assert_array_equal(data["base_hand_pose"][1], np.zeros(45, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
