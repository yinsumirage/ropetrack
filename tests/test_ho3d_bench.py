import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_ho3d.py"
    spec = importlib.util.spec_from_file_location("bench_ho3d", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Ho3dBenchGenericTest(unittest.TestCase):
    def test_gt_bbox_candidates_allow_multiple_boxes_per_sample(self):
        bench = load_script()
        sample = bench.Ho3dSample("S/0001", Path("img.png"), Path("meta.pkl"))

        candidates = bench.bbox_candidates_from_sample(
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
        self.assertEqual([round(c.score, 3) for c in candidates], [0.8, 0.9])
        self.assertEqual(candidates[1].bbox_xyxy.tolist(), [5.0, 6.0, 7.0, 8.0])

    def test_select_sample_predictions_zero_fills_missing_and_picks_best_score(self):
        bench = load_script()
        samples = [
            bench.Ho3dSample("A/0000", Path("a.png"), Path("a.pkl")),
            bench.Ho3dSample("B/0001", Path("b.png"), Path("b.pkl")),
        ]
        low = bench.BatchHandPrediction(
            candidate=bench.BBoxItem(0, 0, samples[0], np.asarray([0, 0, 1, 1], dtype=np.float32), True, 0.1, "detector"),
            vertices=np.ones((778, 3), dtype=np.float32),
            keypoints_3d=np.ones((21, 3), dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
        )
        high = bench.BatchHandPrediction(
            candidate=bench.BBoxItem(0, 1, samples[0], np.asarray([0, 0, 2, 2], dtype=np.float32), True, 0.9, "detector"),
            vertices=np.full((778, 3), 2.0, dtype=np.float32),
            keypoints_3d=np.full((21, 3), 2.0, dtype=np.float32),
            cam_t=np.zeros(3, dtype=np.float32),
        )

        selected, failures = bench.select_sample_predictions(samples, [low, high])

        self.assertIs(selected[0], high)
        self.assertIsNone(selected[1])
        self.assertEqual(failures, [{"idx": 1, "sample_id": "B/0001", "error": "RuntimeError('no hand detected')"}])

    def test_detect_bbox_candidates_batched_flattens_per_image_results(self):
        bench = load_script()
        samples = [
            bench.Ho3dSample("A/0000", Path("a.png"), Path("a.pkl")),
            bench.Ho3dSample("B/0001", Path("b.png"), Path("b.pkl")),
        ]

        class ArrayBox:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class Boxes:
            def __init__(self, boxes, cls, conf):
                self.xyxy = ArrayBox(boxes)
                self.cls = ArrayBox(cls)
                self.conf = ArrayBox(conf)

            def __len__(self):
                return len(self.conf.value)

        class Result:
            def __init__(self, boxes):
                self.boxes = boxes

        class Predictor:
            det_conf = 0.3
            det_iou = 0.3

            def _detector(self, images, **_kwargs):
                self.num_images = len(images)
                return [
                    Result(Boxes([[1, 2, 3, 4], [5, 6, 7, 8]], [1, 0], [0.4, 0.9])),
                    Result(Boxes([[9, 10, 11, 12]], [1], [0.7])),
                ]

        old_cv2 = sys.modules.get("cv2")
        sys.modules["cv2"] = types.SimpleNamespace(imread=lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))
        predictor = Predictor()
        try:
            candidates = bench.detect_bbox_candidates_batched(predictor, samples, detector_batch_size=8)
        finally:
            if old_cv2 is None:
                sys.modules.pop("cv2", None)
            else:
                sys.modules["cv2"] = old_cv2

        self.assertEqual(predictor.num_images, 2)
        self.assertEqual([c.sample_index for c in candidates], [0, 0, 1])
        self.assertEqual([c.bbox_index for c in candidates], [0, 1, 0])
        self.assertEqual([c.is_right for c in candidates], [True, False, True])
        self.assertEqual([round(c.score, 3) for c in candidates], [0.4, 0.9, 0.7])

    def test_write_eval_gt_subset_limits_gt_files_to_prediction_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            out = Path(tmp) / "out"
            root.mkdir()
            (root / "evaluation_xyz.json").write_text(json.dumps([1, 2, 3]))
            (root / "evaluation_verts.json").write_text(json.dumps([4, 5, 6]))

            bench = load_script()
            bench.write_eval_gt_subset(root, out, 2)

            self.assertEqual(json.loads((out / "evaluation_xyz.json").read_text()), [1, 2])
            self.assertEqual(json.loads((out / "evaluation_verts.json").read_text()), [4, 5])


if __name__ == "__main__":
    unittest.main()
