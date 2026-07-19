import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DexYcbArtifactTest(unittest.TestCase):
    def test_raw_signature_ignores_declared_derived_directories(self):
        script = load_script("verify_dexycb_artifacts")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            root.mkdir()
            (root / "subject").mkdir()
            (root / "subject" / "label.npz").write_bytes(b"label")
            (root / "img_feats").mkdir()
            (root / "img_feats" / "old.cache").write_bytes(b"ignored")
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            script.raw_signature(root, first)
            (root / "img_feats" / "old.cache").write_bytes(b"changed")
            script.raw_signature(root, second)
            self.assertEqual(first.read_text(), second.read_text())


if __name__ == "__main__":
    unittest.main()
