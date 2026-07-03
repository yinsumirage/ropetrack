import unittest
from pathlib import Path

from ropetrack.eval.config import build_run_args


class EvalConfigTest(unittest.TestCase):
    def test_freihand_defaults_to_model_keypoints_like_hamer_eval(self):
        args = build_run_args("freihand", "hamer_anyhand")

        self.assertEqual(args.dataset, "freihand")
        self.assertEqual(args.adapter, "freihand")
        self.assertEqual(args.freihand_root, Path("/data/wentao/ropetrack/FreiHAND"))
        self.assertEqual(args.backend, "hamer")
        self.assertEqual(args.joint_source, "model_keypoints")

    def test_ho3d_defaults_to_mano_vertices_protocol(self):
        args = build_run_args("ho3d_v3", "wilor_anyhand")

        self.assertEqual(args.adapter, "ho3d")
        self.assertEqual(args.ho3d_root, Path("/data/wentao/ropetrack/HO3D_v3"))
        self.assertEqual(args.backend, "wilor")
        self.assertEqual(args.joint_source, "mano_vertices")
        self.assertEqual(args.wilor_ckpt, Path(__file__).resolve().parents[1] / "pretrained_models" / "anyhand_wilor.ckpt")

    def test_out_dir_override_wins_over_config_template(self):
        args = build_run_args("ho3d_v2", "wilor_anyhand", out_dir=Path("custom_out"))

        self.assertEqual(args.out_dir, Path("custom_out"))


if __name__ == "__main__":
    unittest.main()
