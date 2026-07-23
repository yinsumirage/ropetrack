import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "legacy" / "temporal" / "temporal_oracle_state.py"
    spec = importlib.util.spec_from_file_location("temporal_oracle_state", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TemporalOracleStateTest(unittest.TestCase):
    def test_zero_constant_and_damped_motion_are_bounded(self):
        script = load_script()
        pose = np.zeros((5, 45), dtype=np.float32)
        pose[:, 2] = np.arange(5) * 0.1

        zero = script.extrapolate_pose(pose, 60, "last_clean")
        cv1 = script.extrapolate_pose(pose, 1, "constant_velocity")
        cv60 = script.extrapolate_pose(pose, 60, "constant_velocity")
        damped60 = script.extrapolate_pose(pose, 60, "damped_velocity")

        self.assertAlmostEqual(float(zero[2]), 0.4, places=5)
        self.assertAlmostEqual(float(cv1[2]), 0.5, places=5)
        self.assertAlmostEqual(float(cv60[2]), 0.9, places=5)
        self.assertAlmostEqual(float(damped60[2]), 0.9, places=5)

    def test_states_and_prefix_beta_change_only_masked_rows(self):
        script = load_script()
        pose = np.zeros((120, 45), dtype=np.float32)
        pose[25:30, 2] = np.arange(5) * 0.1
        betas = np.repeat(np.arange(120, dtype=np.float32)[:, None], 10, axis=1)

        states, fixed, masked = script.build_oracle_states(pose, betas, (np.arange(120),))

        np.testing.assert_array_equal(masked, np.arange(30, 90))
        np.testing.assert_array_equal(states["last_clean"][:30], pose[:30])
        np.testing.assert_array_equal(states["last_clean"][90:], pose[90:])
        np.testing.assert_allclose(states["last_clean"][30:90, 2], 0.4, atol=1e-5)
        np.testing.assert_array_equal(fixed[:30], betas[:30])
        np.testing.assert_array_equal(fixed[90:], betas[90:])
        np.testing.assert_allclose(fixed[30:90], 14.5)

    def test_rope_shuffle_is_mask_local_and_distribution_preserving(self):
        script = load_script()
        values = np.arange(120 * 5, dtype=np.float32).reshape(120, 5)
        cache = {
            "input_rope_norm": values,
            "rope_valid": values.astype(np.int64) % 2 == 0,
        }

        shuffled = script.shuffle_masked_rope(cache, (np.arange(120),), seed=20260710)

        np.testing.assert_array_equal(shuffled["input_rope_norm"][:30], values[:30])
        np.testing.assert_array_equal(shuffled["input_rope_norm"][90:], values[90:])
        self.assertFalse(np.array_equal(shuffled["input_rope_norm"][30:90], values[30:90]))
        self.assertEqual(
            sorted(shuffled["input_rope_norm"][30:90, 0].tolist()),
            sorted(values[30:90, 0].tolist()),
        )


if __name__ == "__main__":
    unittest.main()
