import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_egodex.py"
    spec = importlib.util.spec_from_file_location("prepare_egodex", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareEgoDexTest(unittest.TestCase):
    def test_joint_order_is_wrist_then_five_fingers(self):
        module = load_script()
        names = module.joint_names("right")
        self.assertEqual(len(names), 21)
        self.assertEqual(names[:5], (
            "rightHand",
            "rightThumbKnuckle",
            "rightThumbIntermediateBase",
            "rightThumbIntermediateTip",
            "rightThumbTip",
        ))
        self.assertNotIn("rightIndexFingerMetacarpal", names)

    def test_camera_conversion_matches_official_projection_frame(self):
        module = load_script()
        names = module.joint_names("left")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.hdf5"
            with h5py.File(path, "w") as h5:
                transforms = h5.create_group("transforms")
                transforms.create_dataset("camera", data=np.eye(4, dtype=np.float32)[None])
                for idx, name in enumerate(names):
                    value = np.eye(4, dtype=np.float32)[None]
                    value[0, :3, 3] = [idx, 2.0, -4.0]
                    transforms.create_dataset(name, data=value)
            with h5py.File(path, "r") as h5:
                joints = module.camera_joints(h5, names, 0)

        np.testing.assert_allclose(joints[0], [0.0, 2.0, -4.0])
        np.testing.assert_allclose(joints[-1], [20.0, 2.0, -4.0])

    def test_bbox_projects_and_pads(self):
        module = load_script()
        joints = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]], dtype=np.float32)
        intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
        bbox = module.padded_bbox(joints, intrinsic, 200, 200, margin=0.1)
        np.testing.assert_allclose(bbox, [48.0, 58.0, 62.0, 82.0])

    def test_two_hands_share_split_group_but_not_temporal_frame(self):
        module = load_script()
        left = module.temporal_sample_id("task", "0", "left", 12)
        right = module.temporal_sample_id("task", "0", "right", 12)

        self.assertEqual(left, "task__0/0000012")
        self.assertEqual(right, "task__0/1000012")


if __name__ == "__main__":
    unittest.main()
