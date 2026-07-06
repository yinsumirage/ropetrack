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

    def test_original_methods_use_original_checkpoints(self):
        repo = Path(__file__).resolve().parents[1]

        wilor = build_run_args("ho3d_v2", "wilor_original")
        hamer = build_run_args("ho3d_v2", "hamer_original")

        self.assertEqual(wilor.backend, "wilor")
        self.assertEqual(wilor.wilor_ckpt, repo / "pretrained_models" / "wilor_final.ckpt")
        self.assertEqual(hamer.backend, "hamer")
        self.assertEqual(hamer.hamer_ckpt, repo / "pretrained_models" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt")

    def test_out_dir_override_wins_over_config_template(self):
        args = build_run_args("ho3d_v2", "wilor_anyhand", out_dir=Path("custom_out"))

        self.assertEqual(args.out_dir, Path("custom_out"))

    def test_default_limit_is_full_dataset(self):
        args = build_run_args("freihand", "wilor_original")

        self.assertEqual(args.limit, 0)

    def test_hard_split_dataset_config_keeps_base_adapter(self):
        args = build_run_args("ho3d_v2_mask70", "wilor_anyhand")

        self.assertEqual(args.adapter, "ho3d")
        self.assertEqual(args.ho3d_root, Path("/data/wentao/ropetrack/hard/ho3d_v2/mask70"))

    def test_finger_end_hard_split_dataset_config_keeps_base_protocol(self):
        args = build_run_args("ho3d_v2_finger_end80", "wilor_anyhand")

        self.assertEqual(args.adapter, "ho3d")
        self.assertEqual(args.ho3d_root, Path("/data/wentao/ropetrack/hard/ho3d_v2/finger_end80"))
        self.assertEqual(args.joint_source, "mano_vertices")


if __name__ == "__main__":
    unittest.main()
