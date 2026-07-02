import tempfile
import unittest
from pathlib import Path

from ropetrack.io import read_jsonl, write_jsonl
from ropetrack.metrics import fingertip_error_mm, mean_joint_error_mm
from ropetrack.schema import PredictionRecord, SampleRecord


class SmokeTest(unittest.TestCase):
    def test_schema_and_jsonl_roundtrip(self):
        sample = SampleRecord(
            sample_id="freihand/000001",
            dataset="freihand",
            split="val",
            image_path="data/processed/freihand/images/000001.jpg",
            hand_side="right",
            bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        )
        pred = PredictionRecord(
            sample_id=sample.sample_id,
            backend="hamer_original",
            pred_joints3d_mm_path="predictions/hamer_original/000001_joints3d.npy",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            write_jsonl(path, [sample.to_dict(), pred.to_dict()])
            rows = list(read_jsonl(path))

        self.assertEqual(rows[0]["sample_id"], "freihand/000001")
        self.assertEqual(rows[1]["backend"], "hamer_original")

    def test_basic_metrics(self):
        gt = [(0.0, 0.0, 0.0)] * 21
        pred = [(1.0, 0.0, 0.0)] * 21

        self.assertEqual(mean_joint_error_mm(pred, gt), 1.0)
        self.assertEqual(fingertip_error_mm(pred, gt), 1.0)


if __name__ == "__main__":
    unittest.main()
