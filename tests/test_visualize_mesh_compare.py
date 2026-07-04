import unittest

import numpy as np

from ropetrack.visualize.mesh_compare import MeshTriplet, align_mesh_to_gt, mesh_error, select_triplets


class MeshCompareTest(unittest.TestCase):
    def test_alignment_removes_translation_for_same_shape(self):
        gt = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        pred = gt + np.asarray([10.0, -5.0, 2.0])

        aligned = align_mesh_to_gt(gt, pred)

        self.assertLess(mesh_error(gt, aligned), 1e-8)

    def test_select_triplets_can_pick_middle_and_low_degradation(self):
        triplets = [
            MeshTriplet(i, str(i), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), 0.0, value)
            for i, value in enumerate([0.0, 1.0, 2.0, 3.0, 4.0])
        ]

        self.assertEqual([t.index for t in select_triplets(triplets, 2, "middle_degradation")], [1, 2])
        self.assertEqual([t.index for t in select_triplets(triplets, 2, "low_degradation")], [0, 1])


if __name__ == "__main__":
    unittest.main()
