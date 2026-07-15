import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "temporal_state_followups.py"
    spec = importlib.util.spec_from_file_location("temporal_state_followups", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TemporalStateFollowupsTest(unittest.TestCase):
    def test_states_and_finger_mix(self):
        script = load_script()
        pose = np.zeros((120, 45), dtype=np.float32)
        pose[:, 0] = np.arange(120)
        pose[25:30, 3] = 1.0
        states = script.build_followup_states(pose, (np.arange(120),))

        self.assertEqual(float(states["state_age5"][30, 0]), 25.0)
        self.assertEqual(float(states["state_age15"][30, 0]), 15.0)
        self.assertEqual(float(states["state_age30"][30, 0]), 0.0)
        self.assertEqual(float(states["delayed_freeze3"][30, 0]), 30.0)
        self.assertEqual(float(states["delayed_freeze3"][33, 0]), 32.0)
        self.assertEqual(float(states["false_update_h15"][43, 0]), 29.0)
        self.assertEqual(float(states["false_update_h15"][44, 0]), 44.0)

        current = np.zeros((2, 45), dtype=np.float32)
        state = np.ones((2, 45), dtype=np.float32)
        gate = np.zeros((2, 5), dtype=bool)
        gate[0, 0] = True
        mixed = script.mix_finger_poses(current, state, gate)
        self.assertEqual(int(mixed[0].sum()), 9)
        self.assertEqual(int(mixed[1].sum()), 0)

    def test_rotation_aggregates_are_finite(self):
        script = load_script()
        poses = np.zeros((5, 45), dtype=np.float32)
        poses[:, 2] = np.asarray([0.0, 0.1, 0.2, 0.3, 2.0])
        mean = script.rotation_mean_pose(poses)
        medoid = script.rotation_medoid_pose(poses)
        self.assertTrue(np.isfinite(mean).all())
        self.assertAlmostEqual(float(medoid[2]), 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
