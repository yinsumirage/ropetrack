import unittest

import numpy as np
import torch

from ropetrack.eval.protocols import eval_points_from_model, joints_from_vertices
from ropetrack.refine.oracle import (
    FREIHAND_TIP_JOINT_IDS,
    HO3D_TIP_JOINT_IDS,
    oracle_joint_ids,
    oracle_loss_cm2,
    torch_eval_joints_from_vertices,
    torch_eval_points_from_model,
    wrist_relative,
)


class OracleJointIdsTest(unittest.TestCase):
    def test_tip_ids_per_dataset(self):
        self.assertEqual(oracle_joint_ids("freihand", "oracle_tip"), list(FREIHAND_TIP_JOINT_IDS))
        self.assertEqual(oracle_joint_ids("egodex", "oracle_tip"), list(FREIHAND_TIP_JOINT_IDS))
        self.assertEqual(oracle_joint_ids("arctic", "oracle_tip"), list(FREIHAND_TIP_JOINT_IDS))
        self.assertEqual(oracle_joint_ids("ho3d", "oracle_tip"), list(HO3D_TIP_JOINT_IDS))
        self.assertEqual(oracle_joint_ids("ho3d_v2", "oracle_tip"), list(HO3D_TIP_JOINT_IDS))

    def test_chain_ids_exclude_wrist(self):
        ids = oracle_joint_ids("freihand", "oracle_chain")
        self.assertEqual(ids, list(range(1, 21)))

    def test_unknown_objective_raises(self):
        with self.assertRaises(ValueError):
            oracle_joint_ids("freihand", "oracle_verts")


class TorchProtocolParityTest(unittest.TestCase):
    """The torch decode path must match the numpy scoring protocol exactly."""

    def setUp(self):
        rng = np.random.default_rng(13)
        self.verts = rng.normal(scale=0.05, size=(3, 778, 3)).astype(np.float32)
        self.cam_t = rng.normal(scale=0.2, size=(3, 3)).astype(np.float32)
        self.j_regressor = rng.uniform(size=(16, 778)).astype(np.float32)
        self.j_regressor /= self.j_regressor.sum(axis=1, keepdims=True)

    def test_eval_points_parity(self):
        for dataset in ("freihand", "ho3d", "egodex"):
            with self.subTest(dataset=dataset):
                expected = np.stack([
                    eval_points_from_model(dataset, self.verts[i], self.cam_t[i], "m")
                    for i in range(len(self.verts))
                ])
                actual = torch_eval_points_from_model(
                    dataset, torch.from_numpy(self.verts), torch.from_numpy(self.cam_t)
                ).numpy()
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_eval_joints_parity(self):
        for dataset in ("freihand", "ho3d", "egodex"):
            with self.subTest(dataset=dataset):
                verts_eval = np.stack([
                    eval_points_from_model(dataset, self.verts[i], self.cam_t[i], "m")
                    for i in range(len(self.verts))
                ])
                expected = np.stack([
                    joints_from_vertices(dataset, verts_eval[i], self.j_regressor)
                    for i in range(len(verts_eval))
                ])
                actual = torch_eval_joints_from_vertices(
                    dataset, torch.from_numpy(verts_eval), torch.from_numpy(self.j_regressor)
                ).numpy()
                self.assertEqual(actual.shape, (3, 21, 3))
                np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_arctic_vertex_regression_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kinematic model_keypoints"):
            torch_eval_joints_from_vertices(
                "arctic", torch.from_numpy(self.verts), torch.from_numpy(self.j_regressor)
            )


class OracleLossTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.gt = torch.from_numpy(rng.normal(scale=0.05, size=(2, 21, 3)).astype(np.float32))

    def test_zero_when_equal(self):
        loss = oracle_loss_cm2(self.gt.clone(), self.gt, list(range(1, 21)))
        self.assertAlmostEqual(float(loss), 0.0, places=8)

    def test_translation_invariance(self):
        shifted = self.gt + torch.tensor([0.3, -0.2, 0.5])
        loss = oracle_loss_cm2(shifted, self.gt, [4, 8, 12, 16, 20])
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_known_displacement_scale(self):
        pred = self.gt.clone()
        pred[:, 4] += torch.tensor([0.01, 0.0, 0.0])  # 1 cm on one tip
        loss = oracle_loss_cm2(pred, self.gt, [4])
        # squared error 1 cm^2 on the x component, averaged over 3 components.
        self.assertAlmostEqual(float(loss), 1.0 / 3.0, places=5)

    def test_wrist_relative(self):
        joints = self.gt.clone()
        rel = wrist_relative(joints)
        self.assertTrue(torch.allclose(rel[:, 0], torch.zeros(2, 3), atol=1e-7))


if __name__ == "__main__":
    unittest.main()
