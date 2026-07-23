import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ropetrack.datasets.hand_pose import (
    ho3d_projected_hand_bbox,
    iter_hand_pose_samples,
    iter_ho3d_train_samples,
    load_gt_bbox_candidates,
    read_ho3d_train_ids,
)


ROOT = Path(__file__).resolve().parents[1]
K = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def load_script(name: str):
    path = ROOT / "scripts" / "datasets" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def toy_ho3d_joints(center_x: float = 0.0) -> np.ndarray:
    """21 joints in the HO3D (OpenGL) frame: OpenCV z=0.5 -> stored z=-0.5."""
    rng = np.random.default_rng(11)
    joints_opencv = rng.uniform(-0.05, 0.05, size=(21, 3))
    joints_opencv[:, 0] += center_x
    joints_opencv[:, 2] = 0.5 + 0.02 * rng.uniform(size=21)
    return joints_opencv * np.asarray([1.0, -1.0, -1.0])


def make_meta(with_bbox: bool = False) -> dict:
    meta = {
        "handJoints3D": toy_ho3d_joints(),
        "handPose": np.zeros(48),
        "handBeta": np.zeros(10),
        "handTrans": np.zeros(3),
        "camMat": K.copy(),
    }
    if with_bbox:
        meta["handBoundingBox"] = [100.0, 100.0, 400.0, 400.0]
    return meta


def build_train_root(tmp: Path, num_seq: int = 2, frames: int = 6, with_bbox: bool = False) -> list[str]:
    ids = []
    for seq_idx in range(num_seq):
        seq = f"SEQ{seq_idx}"
        rgb = tmp / "train" / seq / "rgb"
        meta_dir = tmp / "train" / seq / "meta"
        rgb.mkdir(parents=True)
        meta_dir.mkdir(parents=True)
        for frame_idx in range(frames):
            frame = f"{frame_idx:04d}"
            from PIL import Image

            Image.new("RGB", (640, 480), (90, 120, 150)).save(rgb / f"{frame}.jpg")
            with (meta_dir / f"{frame}.pkl").open("wb") as handle:
                pickle.dump(make_meta(with_bbox=with_bbox), handle)
            ids.append(f"{seq}/{frame}")
    (tmp / "train.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


class TrainIdsAndSamplesTest(unittest.TestCase):
    def test_stride_and_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = build_train_root(root)
            self.assertEqual(read_ho3d_train_ids(root), ids)
            strided = read_ho3d_train_ids(root, stride=3)
            self.assertEqual(strided, ids[::3])
            self.assertEqual(read_ho3d_train_ids(root, stride=3, limit=2), ids[::3][:2])
            with self.assertRaises(ValueError):
                read_ho3d_train_ids(root, stride=0)

    def test_missing_list_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_ho3d_train_ids(Path(tmp))

    def test_iter_train_samples_resolves_jpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = build_train_root(root)
            samples = list(iter_ho3d_train_samples(root, None))
            self.assertEqual(len(samples), len(ids))
            self.assertTrue(samples[0].image_path.suffix == ".jpg")
            self.assertTrue(samples[0].image_path.exists())
            self.assertTrue(samples[0].meta_path.exists())

    def test_adapter_dispatch_supports_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_train_root(root)
            samples = list(iter_hand_pose_samples("ho3d", root, 3, split="training"))
            self.assertEqual(len(samples), 3)
            with self.assertRaises(ValueError):
                list(iter_hand_pose_samples("ho3d", root, None, split="nope"))


class ProjectedBboxTest(unittest.TestCase):
    def test_bbox_covers_projected_joints(self):
        meta = make_meta()
        bbox = ho3d_projected_hand_bbox(meta)
        self.assertEqual(bbox.shape, (1, 4))
        x1, y1, x2, y2 = bbox[0]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertGreaterEqual(x1, 0.0)
        self.assertLessEqual(x2, 640.0)
        # projected joint center should be inside the bbox
        joints = np.asarray(meta["handJoints3D"]) * np.asarray([1.0, -1.0, -1.0])
        uv = (K @ joints.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        self.assertTrue((uv[:, 0] >= x1).all() and (uv[:, 0] <= x2).all())
        self.assertTrue((uv[:, 1] >= y1).all() and (uv[:, 1] <= y2).all())

    def test_gt_bbox_candidates_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_train_root(root, num_seq=1, frames=2, with_bbox=False)
            samples = list(iter_ho3d_train_samples(root, None))
            candidates = load_gt_bbox_candidates("ho3d", samples)
            self.assertEqual(len(candidates), len(samples))
            for candidate in candidates:
                self.assertEqual(candidate.source, "gt_bbox")
                self.assertLess(candidate.bbox_xyxy[0], candidate.bbox_xyxy[2])

    def test_gt_bbox_candidates_prefer_meta_bbox(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_train_root(root, num_seq=1, frames=1, with_bbox=True)
            samples = list(iter_ho3d_train_samples(root, None))
            candidates = load_gt_bbox_candidates("ho3d", samples)
            np.testing.assert_allclose(candidates[0].bbox_xyxy, [100.0, 100.0, 400.0, 400.0])


class MakeHardImagesTrainTest(unittest.TestCase):
    def test_build_train_hard_root_with_stride(self):
        script = load_script("make_hard_images")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            ids = build_train_root(src)
            out = Path(tmp) / "hard"
            script.build_ho3d_train_hard_root(src, out, "mask", 0.7, None, seed=7, stride=3)

            strided = ids[::3]
            emitted_ids = [line.strip() for line in (out / "train.txt").read_text().splitlines() if line.strip()]
            self.assertEqual(emitted_ids, strided)
            xyz = json.loads((out / "training_xyz.json").read_text(encoding="utf-8"))
            self.assertEqual(len(xyz), len(strided))
            self.assertEqual(np.asarray(xyz[0]).shape, (21, 3))
            manifest = [json.loads(line) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(manifest), len(strided))
            self.assertEqual(manifest[0]["effect"], "mask")
            self.assertEqual(manifest[0]["stride"], 3)
            self.assertEqual(len(manifest[0]["points_xy"]), 5)
            for sample_id in strided:
                seq, frame = sample_id.split("/")
                self.assertTrue((out / "train" / seq / "rgb" / f"{frame}.jpg").exists())
                self.assertTrue((out / "train" / seq / "meta" / f"{frame}.pkl").exists())

    def test_hard_root_feeds_train_iterator(self):
        script = load_script("make_hard_images")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            build_train_root(src)
            out = Path(tmp) / "hard"
            script.build_ho3d_train_hard_root(src, out, "mask", 0.7, None, seed=7, stride=2)
            samples = list(iter_ho3d_train_samples(out, None))
            self.assertEqual(len(samples), 6)  # 12 frames / stride 2
            candidates = load_gt_bbox_candidates("ho3d", samples)
            self.assertEqual(len(candidates), 6)


class MakeRopeLabelsTrainTest(unittest.TestCase):
    def test_train_labels_with_stride(self):
        script = load_script("make_rope_labels")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            ids = build_train_root(src)
            output = Path(tmp) / "rope.jsonl"
            rows = script.write_rope_labels("ho3d", src, output, split="training", stride=3)
            self.assertEqual([row["sample_id"] for row in rows], ids[::3])
            self.assertEqual(rows[0]["dataset"], "ho3d")
            self.assertEqual(len(rows[0]["rope_norm"]), 5)
            self.assertTrue(all(rows[0]["rope_valid"]))

    def test_eval_split_rejects_stride_in_cli(self):
        script = load_script("make_rope_labels")
        with tempfile.TemporaryDirectory() as tmp:
            argv_backup = sys.argv
            sys.argv = [
                "make_rope_labels.py", "--dataset", "ho3d", "--input-root", tmp,
                "--output", str(Path(tmp) / "x.jsonl"), "--stride", "3",
            ]
            try:
                with self.assertRaises(ValueError):
                    script.main()
            finally:
                sys.argv = argv_backup


if __name__ == "__main__":
    unittest.main()
