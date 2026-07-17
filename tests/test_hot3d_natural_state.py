from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "hot3d_natural_state.py"
    spec = importlib.util.spec_from_file_location("hot3d_natural_state", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Hot3DNaturalStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = load_script()

    def manifest(self, low_count=3):
        return [
            {"episode_id": "ep", "phase": "context", "phase_index": index + 1}
            for index in range(30)
        ] + [
            {"episode_id": "ep", "phase": "low_visibility", "phase_index": index + 1}
            for index in range(low_count)
        ]

    def test_variable_length_episode_and_last_context_state(self):
        episodes = self.script.natural_episode_rows(self.manifest(3))
        pose = np.arange(33 * 45, dtype=np.float32).reshape(33, 45)
        state, low = self.script.last_context_state(pose, episodes)
        np.testing.assert_array_equal(low, [30, 31, 32])
        np.testing.assert_array_equal(state[:30], pose[:30])
        np.testing.assert_array_equal(state[30:], np.repeat(pose[29:30], 3, axis=0))

    def test_protocol_rejects_noncontiguous_low_phase(self):
        manifest = self.manifest(2)
        manifest[-1]["phase_index"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous from one"):
            self.script.natural_episode_rows(manifest)

    def test_shuffle_stays_inside_low_rows(self):
        episodes = self.script.natural_episode_rows(self.manifest(4))
        values = np.arange(34, dtype=np.float32)[:, None]
        cache = {"input_rope_norm": values.copy(), "rope_valid": values.astype(bool)}
        out = self.script.shuffle_low_rope(cache, episodes, seed=1)
        np.testing.assert_array_equal(out["input_rope_norm"][:30], values[:30])
        self.assertEqual(set(out["input_rope_norm"][30:, 0]), set(values[30:, 0]))


if __name__ == "__main__":
    unittest.main()
