import unittest

import numpy as np

from ropetrack.eval.protocols import (
    canonical_dataset,
    eval_points_from_model,
    joints_from_vertices,
)


class ProtocolTest(unittest.TestCase):
    def test_ho3d_eval_points_adds_camera_translation_and_flips_yz(self):
        pts = eval_points_from_model(
            "ho3d_v2",
            points=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            cam_t=[10.0, 20.0, 30.0],
            units="m",
        )

        self.assertEqual(pts.tolist(), [[11.0, -22.0, -33.0], [14.0, -25.0, -36.0]])

    def test_freihand_eval_points_keeps_camera_axes(self):
        pts = eval_points_from_model(
            "freihand",
            points=[[1.0, 2.0, 3.0]],
            cam_t=[10.0, 20.0, 30.0],
            units="m",
        )

        self.assertEqual(pts.tolist(), [[11.0, 22.0, 33.0]])

    def test_eval_points_converts_mm_to_metres(self):
        pts = eval_points_from_model("freihand", [[1000.0, 0.0, 0.0]], [0.0, 0.0, 0.0], "mm")

        self.assertEqual(pts.tolist(), [[1.0, 0.0, 0.0]])

    def test_ho3d_joints_keep_mano_order_and_hamer_tips(self):
        verts = np.asarray([[float(i), float(i + 1), float(i + 2)] for i in range(778)], dtype=np.float32)
        regressor = np.zeros((16, 778), dtype=np.float32)
        for joint_id in range(16):
            regressor[joint_id, joint_id] = 1.0

        joints = joints_from_vertices("ho3d_v3", verts, regressor)

        self.assertEqual(joints.shape, (21, 3))
        self.assertEqual(joints[16].tolist(), [744.0, 745.0, 746.0])
        self.assertEqual(joints[17].tolist(), [320.0, 321.0, 322.0])
        self.assertEqual(joints[18].tolist(), [443.0, 444.0, 445.0])
        self.assertEqual(joints[19].tolist(), [554.0, 555.0, 556.0])
        self.assertEqual(joints[20].tolist(), [671.0, 672.0, 673.0])

    def test_freihand_joints_reorder_mano_and_use_freihand_tips(self):
        verts = np.asarray([[float(i), float(i + 1), float(i + 2)] for i in range(778)], dtype=np.float32)
        regressor = np.zeros((16, 778), dtype=np.float32)
        for joint_id in range(16):
            regressor[joint_id, joint_id] = 1.0

        joints = joints_from_vertices("freihand", verts, regressor)

        self.assertEqual(joints.shape, (21, 3))
        self.assertEqual(joints[1].tolist(), [13.0, 14.0, 15.0])
        self.assertEqual(joints[4].tolist(), [744.0, 745.0, 746.0])
        self.assertEqual(joints[8].tolist(), [320.0, 321.0, 322.0])
        self.assertEqual(joints[20].tolist(), [672.0, 673.0, 674.0])

    def test_unknown_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_dataset("dexycb")


if __name__ == "__main__":
    unittest.main()
