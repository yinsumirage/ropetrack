import unittest

import numpy as np
import torch

from ropetrack.refine.actions import (
    FINGER_POSE_GROUPS,
    JOINT_TO_FINGER,
    alpha_dim,
    apply_action_np,
    apply_action_torch,
    per_finger_alpha_abs,
    per_finger_pose_magnitude,
)


def legacy_apply_finger_curl_alpha(base_hand_pose, alpha):
    """Reference mult5 implementation from experience/0027."""
    refined = np.asarray(base_hand_pose, dtype=np.float32).copy()
    alpha = np.asarray(alpha, dtype=np.float32)
    for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
        dims = np.asarray([3 * joint + axis for joint in joints for axis in range(3)])
        refined[:, dims] += alpha[:, finger_idx:finger_idx + 1] * refined[:, dims]
    return refined


class ActionSpaceTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.base = rng.normal(size=(4, 45)).astype(np.float32)

    def test_alpha_dim(self):
        self.assertEqual(alpha_dim("mult5"), 5)
        self.assertEqual(alpha_dim("mult15"), 15)
        self.assertEqual(alpha_dim("flex15"), 15)
        self.assertEqual(alpha_dim("flex5"), 5)
        with self.assertRaises(ValueError):
            alpha_dim("free45")

    def test_joint_to_finger_covers_all_joints(self):
        self.assertEqual(sorted(np.concatenate([np.asarray(g) for g in FINGER_POSE_GROUPS]).tolist()), list(range(15)))
        for finger_idx, joints in enumerate(FINGER_POSE_GROUPS):
            for joint in joints:
                self.assertEqual(JOINT_TO_FINGER[joint], finger_idx)

    def test_mult5_matches_legacy_implementation(self):
        rng = np.random.default_rng(5)
        alpha = rng.uniform(-0.5, 0.5, size=(4, 5)).astype(np.float32)
        np.testing.assert_allclose(
            apply_action_np(self.base, alpha, "mult5"),
            legacy_apply_finger_curl_alpha(self.base, alpha),
            atol=1e-6,
        )

    def test_mult15_changes_only_selected_joint(self):
        alpha = np.zeros((4, 15), dtype=np.float32)
        alpha[:, 7] = 0.3
        refined = apply_action_np(self.base, alpha, "mult15")
        changed = np.flatnonzero(np.abs(refined[0] - self.base[0]) > 1e-7)
        np.testing.assert_array_equal(changed, [21, 22, 23])
        np.testing.assert_allclose(refined[:, 21:24], 1.3 * self.base[:, 21:24], atol=1e-6)

    def test_flex15_adds_alpha_along_directions(self):
        rng = np.random.default_rng(7)
        directions = rng.normal(size=(4, 15, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=2, keepdims=True)
        alpha = np.zeros((4, 15), dtype=np.float32)
        alpha[:, 2] = 0.25
        refined = apply_action_np(self.base, alpha, "flex15", directions)
        expected = self.base.copy()
        expected[:, 6:9] += 0.25 * directions[:, 2]
        np.testing.assert_allclose(refined, expected, atol=1e-6)

    def test_flex15_requires_directions(self):
        alpha = np.zeros((4, 15), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_action_np(self.base, alpha, "flex15")

    def test_flex5_broadcasts_alpha_over_finger_joints(self):
        rng = np.random.default_rng(19)
        directions = rng.normal(size=(4, 15, 3)).astype(np.float32)
        alpha = np.zeros((4, 5), dtype=np.float32)
        alpha[:, 0] = 0.2  # thumb
        refined = apply_action_np(self.base, alpha, "flex5", directions)
        expected = self.base.copy()
        for joint in FINGER_POSE_GROUPS[0]:
            expected[:, 3 * joint : 3 * joint + 3] += 0.2 * directions[:, joint]
        np.testing.assert_allclose(refined, expected, atol=1e-6)
        # other fingers untouched
        untouched = [3 * j + a for f in range(1, 5) for j in FINGER_POSE_GROUPS[f] for a in range(3)]
        np.testing.assert_allclose(refined[:, untouched], self.base[:, untouched], atol=1e-7)

    def test_flex5_requires_directions(self):
        alpha = np.zeros((4, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            apply_action_np(self.base, alpha, "flex5")

    def test_alpha_shape_validation(self):
        with self.assertRaises(ValueError):
            apply_action_np(self.base, np.zeros((4, 15), dtype=np.float32), "mult5")
        with self.assertRaises(ValueError):
            apply_action_np(self.base, np.zeros((4, 5), dtype=np.float32), "mult15")

    def test_torch_numpy_parity_all_action_spaces(self):
        rng = np.random.default_rng(11)
        directions = rng.normal(size=(4, 15, 3)).astype(np.float32)
        cases = [
            ("mult5", rng.uniform(-0.4, 0.4, size=(4, 5)).astype(np.float32), None),
            ("mult15", rng.uniform(-0.4, 0.4, size=(4, 15)).astype(np.float32), None),
            ("flex15", rng.uniform(-0.4, 0.4, size=(4, 15)).astype(np.float32), directions),
            ("flex5", rng.uniform(-0.4, 0.4, size=(4, 5)).astype(np.float32), directions),
        ]
        for action_space, alpha, dirs in cases:
            with self.subTest(action_space=action_space):
                expected = apply_action_np(self.base, alpha, action_space, dirs)
                actual = apply_action_torch(
                    torch.from_numpy(self.base),
                    torch.from_numpy(alpha),
                    action_space,
                    torch.from_numpy(dirs) if dirs is not None else None,
                ).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_torch_apply_is_differentiable(self):
        alpha = torch.zeros((2, 15), requires_grad=True)
        base = torch.from_numpy(self.base[:2])
        directions = torch.ones((2, 15, 3))
        out = apply_action_torch(base, alpha, "flex15", directions)
        out.sum().backward()
        self.assertIsNotNone(alpha.grad)
        self.assertGreater(float(alpha.grad.abs().sum()), 0.0)

    def test_per_finger_alpha_abs(self):
        alpha5 = np.asarray([[0.1, -0.2, 0.3, -0.4, 0.5]], dtype=np.float32)
        np.testing.assert_allclose(per_finger_alpha_abs(alpha5, "mult5"), np.abs(alpha5), atol=1e-7)
        np.testing.assert_allclose(per_finger_alpha_abs(alpha5, "flex5"), np.abs(alpha5), atol=1e-7)

        alpha15 = np.zeros((1, 15), dtype=np.float32)
        alpha15[0, list(FINGER_POSE_GROUPS[0])] = [0.3, -0.3, 0.3]  # thumb joints
        out = per_finger_alpha_abs(alpha15, "flex15")
        self.assertAlmostEqual(float(out[0, 0]), 0.3, places=6)
        self.assertAlmostEqual(float(out[0, 1]), 0.0, places=6)

    def test_per_finger_pose_magnitude(self):
        base = np.zeros((1, 45), dtype=np.float32)
        thumb_dims = [3 * joint + axis for joint in FINGER_POSE_GROUPS[0] for axis in range(3)]
        base[0, thumb_dims] = 1.0
        out = per_finger_pose_magnitude(base)
        self.assertAlmostEqual(float(out[0, 0]), 3.0, places=5)  # sqrt(9)
        self.assertAlmostEqual(float(out[0, 1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
