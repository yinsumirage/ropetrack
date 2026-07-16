import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ropetrack.datasets.hand_pose import (
    Ho3dSample,
    bbox_candidates_from_sample,
    bbox_from_projected_points,
    hand_bbox_from_meta,
    iter_hand_pose_samples,
    load_gt_bbox_candidates,
    resolve_image_path,
    validate_eval_protocol,
    write_eval_gt_subset,
)


class HandPoseDatasetsTest(unittest.TestCase):
    def test_ho3d_iter_samples_prefers_evaluation_txt_order_and_jpg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation" / "SM1" / "rgb").mkdir(parents=True)
            (root / "evaluation" / "SM1" / "meta").mkdir(parents=True)
            (root / "evaluation" / "SM1" / "rgb" / "0000.jpg").write_text("")
            (root / "evaluation.txt").write_text("SM1/0000\n")

            sample = next(iter(iter_hand_pose_samples("ho3d", root, limit=None)))

        self.assertEqual(sample.sample_id, "SM1/0000")
        self.assertTrue(sample.image_path.as_posix().endswith("SM1/rgb/0000.jpg"))

    def test_freihand_iter_samples_builds_bbox_from_projected_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation" / "rgb").mkdir(parents=True)
            (root / "evaluation" / "rgb" / "00000000.jpg").write_text("")
            (root / "evaluation_K.json").write_text(json.dumps([
                [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
            ]))
            (root / "evaluation_verts.json").write_text(json.dumps([
                [[0.1, 0.2, 1.0], [0.4, 0.5, 1.0]]
            ]))

            sample = next(iter(iter_hand_pose_samples("freihand", root, limit=None)))

        self.assertEqual(sample.sample_id, "00000000")
        self.assertEqual(sample.bbox_xyxy.tolist(), [10.0, 20.0, 40.0, 50.0])

    def test_bbox_candidates_allow_multiple_boxes_per_sample(self):
        sample = Ho3dSample("S/0001", Path("img.png"), Path("meta.pkl"))

        candidates = bbox_candidates_from_sample(
            sample_index=3,
            sample=sample,
            boxes=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
            is_right=np.asarray([1.0, 0.0], dtype=np.float32),
            scores=np.asarray([0.8, 0.9], dtype=np.float32),
            source="detector",
        )

        self.assertEqual([c.sample_index for c in candidates], [3, 3])
        self.assertEqual([c.bbox_index for c in candidates], [0, 1])
        self.assertEqual([c.is_right for c in candidates], [True, False])

    def test_egodex_manifest_preserves_hand_side_and_temporal_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation" / "frames").mkdir(parents=True)
            row = {
                "sample_id": "task__0__left/000012",
                "image_path": "evaluation/frames/000012.jpg",
                "bbox_xyxy": [1, 2, 30, 40],
                "is_right": False,
                "episode_id": "task/0",
                "frame_index": 12,
                "intrinsic": [[100, 0, 10], [0, 100, 20], [0, 0, 1]],
                "joint_confidence": [0.5] * 21,
            }
            (root / "evaluation.jsonl").write_text(json.dumps(row) + "\n")

            sample = next(iter(iter_hand_pose_samples("egodex", root, limit=None)))
            candidates = load_gt_bbox_candidates("egodex", [sample])

        self.assertEqual(sample.sample_id, "task__0__left/000012")
        self.assertFalse(candidates[0].is_right)
        self.assertEqual(candidates[0].bbox_xyxy.tolist(), [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(sample.intrinsic.shape, (3, 3))
        self.assertEqual(sample.joint_confidence.shape, (21,))

    def test_load_ho3d_gt_bbox_candidates_reads_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "meta.pkl"
            with meta_path.open("wb") as f:
                pickle.dump({"handBoundingBox": [1, 2, 3, 4]}, f)
            sample = Ho3dSample("S/0001", Path("img.png"), meta_path)

            candidates = load_gt_bbox_candidates("ho3d", [sample])

        self.assertEqual(candidates[0].bbox_xyxy.tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_write_eval_gt_subset_limits_gt_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            out = Path(tmp) / "out"
            root.mkdir()
            (root / "evaluation_xyz.json").write_text(json.dumps([1, 2, 3]))
            (root / "evaluation_verts.json").write_text(json.dumps([4, 5, 6]))

            write_eval_gt_subset("ho3d", root, out, 2)

            self.assertEqual(json.loads((out / "evaluation_xyz.json").read_text()), [1, 2])
            self.assertEqual(json.loads((out / "evaluation_verts.json").read_text()), [4, 5])

    def test_ho3d_protocol_check_rejects_wrong_gt_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "meta.pkl"
            with meta_path.open("wb") as f:
                pickle.dump({"handJoints3D": [[1.0, 0.0, 0.0]]}, f)
            (root / "evaluation_xyz.json").write_text(json.dumps([[[2.0, 0.0, 0.0]]]))
            (root / "evaluation_verts.json").write_text(json.dumps([[[0.0, 0.0, 0.0]]]))
            sample = Ho3dSample("S/0001", Path("img.png"), meta_path)

            with self.assertRaisesRegex(ValueError, "HO3D protocol check failed"):
                validate_eval_protocol("ho3d", root, [sample], 1, 0.001)

    def test_ho3d_protocol_check_accepts_flat_meta_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta_path = root / "meta.pkl"
            with meta_path.open("wb") as f:
                pickle.dump({"handJoints3D": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)}, f)
            (root / "evaluation_xyz.json").write_text(json.dumps([[[1.0, 0.0, 0.0]]]))
            (root / "evaluation_verts.json").write_text(json.dumps([[[0.0, 0.0, 0.0]]]))
            sample = Ho3dSample("S/0001", Path("img.png"), meta_path)

            validate_eval_protocol("ho3d", root, [sample], 1, 0.001)

    def test_small_helpers_remain_available_for_hard_image_builder(self):
        self.assertEqual(hand_bbox_from_meta({"handBoundingBox": [1, 2, 3, 4]}).tolist(), [[1.0, 2.0, 3.0, 4.0]])
        np.testing.assert_allclose(
            bbox_from_projected_points([[0.1, 0.2, 1.0]], np.eye(3), image_size=1),
            [0.1, 0.2, 0.1, 0.2],
        )
        self.assertTrue(resolve_image_path(Path("rgb"), "0000").as_posix().endswith("rgb/0000.png"))


if __name__ == "__main__":
    unittest.main()
