import tempfile
import unittest
from pathlib import Path

from ropetrack.io import read_jsonl, write_jsonl


class SmokeTest(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            write_jsonl(path, [{"sample_id": "freihand/000001"}, {"backend": "hamer_original"}])
            rows = list(read_jsonl(path))

        self.assertEqual(rows[0]["sample_id"], "freihand/000001")
        self.assertEqual(rows[1]["backend"], "hamer_original")


if __name__ == "__main__":
    unittest.main()
