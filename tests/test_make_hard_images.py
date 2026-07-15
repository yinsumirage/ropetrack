import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_hard_images.py"
    spec = importlib.util.spec_from_file_location("make_hard_images", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MakeHardImagesTest(unittest.TestCase):
    def test_mask_effect_changes_only_inside_bbox(self):
        hard = load_script()
        image = Image.new("RGB", (10, 10), (255, 0, 0))

        out = hard.apply_hard_effect(image, [2, 2, 8, 8], effect="mask", severity=0.5, seed=1)

        self.assertEqual(out.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(out.getpixel((5, 5)), (0, 0, 0))

    def test_tip_square_masks_given_fingertip_points(self):
        hard = load_script()
        image = Image.new("RGB", (20, 20), (255, 255, 255))

        out = hard.apply_hard_effect(
            image,
            [0, 0, 20, 20],
            effect="tip_square",
            severity=0.2,
            seed=1,
            points_xy=[(5, 5)],
        )

        self.assertEqual(out.getpixel((5, 5)), (0, 0, 0))
        self.assertEqual(out.getpixel((15, 15)), (255, 255, 255))

    def test_finger_end_masks_between_projected_joint_pairs(self):
        hard = load_script()
        image = Image.new("RGB", (30, 30), (255, 255, 255))

        out = hard.apply_hard_effect(
            image,
            [0, 0, 30, 30],
            effect="finger_end",
            severity=0.5,
            seed=1,
            segments_xy=[((5, 10), (20, 10)), ((5, 20), (20, 20))],
        )

        self.assertEqual(out.getpixel((12, 10)), (0, 0, 0))
        self.assertEqual(out.getpixel((12, 20)), (0, 0, 0))
        self.assertEqual(out.getpixel((12, 2)), (255, 255, 255))

    def test_project_fingertips_from_joints_uses_tip_indices(self):
        hard = load_script()
        joints = [[0.0, 0.0, 1.0] for _ in range(21)]
        for joint_id in (4, 8, 12, 16, 20):
            joints[joint_id] = [float(joint_id), float(joint_id + 1), 1.0]
        K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]

        tips = hard.project_fingertips_from_joints(joints, K)

        self.assertEqual(tips[0], (41.0, 52.0))
        self.assertEqual(tips[-1], (201.0, 212.0))

    def test_project_freihand_finger_end_segments_cover_tip_to_pip_to_mcp(self):
        hard = load_script()
        joints = [[0.0, 0.0, 1.0] for _ in range(21)]
        joints[1] = [1.0, 2.0, 1.0]
        joints[4] = [3.0, 4.0, 1.0]
        joints[5] = [5.0, 6.0, 1.0]
        joints[6] = [7.0, 8.0, 1.0]
        joints[8] = [9.0, 10.0, 1.0]
        K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]

        segments = hard.project_finger_end_segments_from_joints(joints, K)

        self.assertEqual(segments[0], ((31.0, 42.0), (11.0, 22.0)))
        self.assertEqual(segments[1], ((91.0, 102.0), (71.0, 82.0)))
        self.assertEqual(segments[2], ((71.0, 82.0), (51.0, 62.0)))
        self.assertEqual(len(segments), 9)

    def test_project_ho3d_fingertips_uses_ho3d_tip_indices_and_camera_axes(self):
        hard = load_script()
        joints = [[0.0, 0.0, -1.0] for _ in range(21)]
        joints[4] = [9.0, 9.0, 1.0]
        joints[16] = [1.0, -2.0, -1.0]
        K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]

        tips = hard.project_ho3d_fingertips_from_joints(joints, K)

        self.assertEqual(tips[0], (11.0, 22.0))

    def test_project_ho3d_finger_end_segments_use_thumb_once_and_other_fingers_twice(self):
        hard = load_script()
        joints = [[0.0, 0.0, -1.0] for _ in range(21)]
        joints[13] = [1.0, -2.0, -1.0]
        joints[16] = [3.0, -4.0, -1.0]
        joints[1] = [5.0, -6.0, -1.0]
        joints[2] = [7.0, -8.0, -1.0]
        joints[17] = [11.0, -12.0, -1.0]
        K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]

        segments = hard.project_ho3d_finger_end_segments_from_joints(joints, K)

        self.assertEqual(segments[0], ((31.0, 42.0), (11.0, 22.0)))
        self.assertEqual(segments[1], ((111.0, 122.0), (71.0, 82.0)))
        self.assertEqual(segments[2], ((71.0, 82.0), (51.0, 62.0)))
        self.assertEqual(len(segments), 9)

    def test_build_freihand_subset_writes_hard_root_and_manifest(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "freihand"
            out = Path(tmp) / "hard"
            (src / "evaluation" / "rgb").mkdir(parents=True)
            Image.new("RGB", (224, 224), (255, 255, 255)).save(src / "evaluation" / "rgb" / "00000000.jpg")
            Image.new("RGB", (224, 224), (255, 255, 255)).save(src / "evaluation" / "rgb" / "00000001.jpg")
            K = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
            verts = [[[0.2, 0.2, 1.0], [0.8, 0.8, 1.0]], [[0.1, 0.1, 1.0], [0.3, 0.3, 1.0]]]
            xyz = [[[0.2, 0.2, 1.0]], [[0.1, 0.1, 1.0]]]
            (src / "evaluation_K.json").write_text(json.dumps([K, K]))
            (src / "evaluation_verts.json").write_text(json.dumps(verts))
            (src / "evaluation_xyz.json").write_text(json.dumps(xyz))

            hard.build_freihand_hard_root(src, out, effect="mask", severity=0.5, limit=1, seed=3)

            self.assertTrue((out / "evaluation" / "rgb" / "00000000.jpg").exists())
            self.assertEqual(len(json.loads((out / "evaluation_xyz.json").read_text())), 1)
            self.assertEqual(len(json.loads((out / "evaluation_verts.json").read_text())), 1)
            self.assertEqual(len(json.loads((out / "evaluation_K.json").read_text())), 1)
            rows = [(json.loads(line)) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "00000000")
            self.assertEqual(rows[0]["effect"], "mask")

    def test_build_ho3d_subset_writes_evaluation_txt_and_subset_gt(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0000.png")
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0001.png")
            for frame in ("0000", "0001"):
                with (meta / f"{frame}.pkl").open("wb") as f:
                    pickle.dump({"handBoundingBox": [10, 10, 50, 50], "handJoints3D": [[0, 0, 0]]}, f)
            (src / "evaluation.txt").write_text("AP10/0000\nAP10/0001\n")
            (src / "evaluation_xyz.json").write_text(json.dumps([[[0, 0, 0]], [[1, 1, 1]]]))
            (src / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]], [[1, 1, 1]]]))

            hard.build_ho3d_hard_root(src, out, effect="mask", severity=0.5, limit=1, seed=3)

            self.assertEqual((out / "evaluation.txt").read_text(), "AP10/0000\n")
            self.assertTrue((out / "evaluation" / "AP10" / "rgb" / "0000.png").exists())
            self.assertTrue((out / "evaluation" / "AP10" / "meta" / "0000.pkl").exists())
            self.assertEqual(len(json.loads((out / "evaluation_xyz.json").read_text())), 1)
            rows = [(json.loads(line)) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["sample_id"], "AP10/0000")

    def test_build_ho3d_subset_can_use_run_meta_sample_order(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            order_path = Path(tmp) / "run_meta.json"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0000.png")
            with (meta / "0000.pkl").open("wb") as f:
                pickle.dump({"handBoundingBox": [10, 10, 50, 50], "handJoints3D": [[0, 0, 0]]}, f)
            (src / "evaluation_xyz.json").write_text(json.dumps([[[0, 0, 0]]]))
            (src / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]]]))
            order_path.write_text(json.dumps({"sample_order": ["AP10/0000"]}))

            hard.build_ho3d_hard_root(src, out, effect="mask", severity=0.5, limit=1, seed=3, sample_order_file=order_path)

            self.assertEqual((out / "evaluation.txt").read_text(), "AP10/0000\n")

    def test_build_ho3d_uses_evaluation_xyz_for_fingertip_points(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta.mkdir(parents=True)
            Image.new("RGB", (64, 64), (255, 255, 255)).save(rgb / "0000.png")
            K = [[10.0, 0.0, 1.0], [0.0, 10.0, 2.0], [0.0, 0.0, 1.0]]
            with (meta / "0000.pkl").open("wb") as f:
                pickle.dump({"handBoundingBox": [10, 10, 50, 50], "handJoints3D": [0, 0, 1], "camMat": K}, f)
            joints = [[0.0, 0.0, 1.0] for _ in range(21)]
            joints[16] = [4.0, -5.0, -1.0]
            (src / "evaluation_xyz.json").write_text(json.dumps([joints]))
            (src / "evaluation_verts.json").write_text(json.dumps([[[0, 0, 0]]]))
            (src / "evaluation.txt").write_text("AP10/0000\n")

            hard.build_ho3d_hard_root(src, out, effect="tip_square", severity=0.5, limit=1, seed=3)

            rows = [(json.loads(line)) for line in (out / "hard_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["points_xy"][0], [41.0, 52.0])

    def test_build_ho3d_episode_root_masks_only_complete_masked_phase(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            rgb = src / "evaluation" / "AP10" / "rgb"
            meta_dir = src / "evaluation" / "AP10" / "meta"
            rgb.mkdir(parents=True)
            meta_dir.mkdir(parents=True)
            K = np.eye(3, dtype=np.float32)
            joints = np.zeros((21, 3), dtype=np.float32)
            joints[:, 2] = -1.0
            source_mtime = 1_600_000_000_000_000_000
            for frame in ("0000", "0001", "0002", "0003"):
                image_path = rgb / f"{frame}.png"
                Image.new("RGB", (32, 32), "white").save(image_path)
                os.utime(image_path, ns=(source_mtime, source_mtime))
                with (meta_dir / f"{frame}.pkl").open("wb") as handle:
                    pickle.dump(
                        {
                            "handBoundingBox": [4, 4, 28, 28],
                            "handJoints3D": joints,
                            "camMat": K,
                        },
                        handle,
                    )
            ids = [f"AP10/{frame}" for frame in ("0000", "0001", "0002", "0003")]
            (src / "evaluation.txt").write_text("\n".join(ids) + "\n")
            (src / "evaluation_xyz.json").write_text(
                json.dumps([joints.tolist()] * len(ids))
            )
            (src / "evaluation_verts.json").write_text(
                json.dumps([[[0, 0, 0]]] * len(ids))
            )

            hard.build_ho3d_hard_root(
                src,
                out,
                "mask",
                0.7,
                None,
                7,
                episode_context=1,
                episode_mask=1,
                episode_recovery=1,
            )

            rows = [
                json.loads(line)
                for line in (out / "hard_manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["episode_phase"] for row in rows],
                ["context", "masked", "recovery", "tail"],
            )
            self.assertEqual(
                [row["episode_id"] for row in rows],
                ["AP10:0:0", "AP10:0:0", "AP10:0:0", None],
            )
            self.assertEqual([row["episode_offset"] for row in rows], [0, 1, 2, 0])
            self.assertEqual({row["segment_id"] for row in rows}, {"AP10:0"})
            self.assertEqual({row["effect"] for row in rows}, {"mask"})
            for row in rows:
                self.assertTrue(
                    {"episode_id", "episode_phase", "episode_offset", "segment_id"}
                    <= row.keys()
                )

            output_images = [out / "evaluation" / "AP10" / "rgb" / f"{frame}.png" for frame in ("0000", "0001", "0002", "0003")]
            self.assertEqual(
                [image.getpixel((16, 16)) for image in map(Image.open, output_images)],
                [(255, 255, 255), (0, 0, 0), (255, 255, 255), (255, 255, 255)],
            )
            for index in (0, 2, 3):
                source_image = rgb / output_images[index].name
                self.assertEqual(
                    output_images[index].stat().st_mtime_ns,
                    source_image.stat().st_mtime_ns,
                )

    def test_build_ho3d_training_episode_root_uses_same_schedule(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "ho3d"
            out = Path(tmp) / "hard"
            rgb = src / "train" / "ABF10" / "rgb"
            meta_dir = src / "train" / "ABF10" / "meta"
            rgb.mkdir(parents=True)
            meta_dir.mkdir(parents=True)
            K = np.asarray([[10.0, 0.0, 16.0], [0.0, 10.0, 16.0], [0.0, 0.0, 1.0]])
            joints = np.zeros((21, 3), dtype=np.float32)
            joints[:, 2] = -1.0
            ids = [f"ABF10/{frame}" for frame in ("0000", "0001", "0002")]
            source_mtime = 1_600_000_000_000_000_000
            for sample_id in ids:
                frame = sample_id.split("/")[1]
                image_path = rgb / f"{frame}.png"
                Image.new("RGB", (32, 32), "white").save(image_path)
                os.utime(image_path, ns=(source_mtime, source_mtime))
                with (meta_dir / f"{frame}.pkl").open("wb") as handle:
                    pickle.dump({"handJoints3D": joints, "camMat": K}, handle)
            (src / "train.txt").write_text("\n".join(ids) + "\n")

            hard.build_ho3d_train_hard_root(
                src,
                out,
                "mask",
                0.7,
                None,
                7,
                episode_context=1,
                episode_mask=1,
                episode_recovery=1,
            )

            rows = [
                json.loads(line)
                for line in (out / "hard_manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["episode_phase"] for row in rows],
                ["context", "masked", "recovery"],
            )
            self.assertEqual({row["segment_id"] for row in rows}, {"ABF10:0"})
            self.assertEqual([row["episode_offset"] for row in rows], [0, 1, 2])
            self.assertEqual({row["episode_id"] for row in rows}, {"ABF10:0:0"})
            self.assertEqual({row["effect"] for row in rows}, {"mask"})
            for row in rows:
                self.assertTrue(
                    {"episode_id", "episode_phase", "episode_offset", "segment_id"}
                    <= row.keys()
                )

            output_images = [
                out / "train" / "ABF10" / "rgb" / f"{frame}.png"
                for frame in ("0000", "0001", "0002")
            ]
            self.assertEqual(
                [image.getpixel((16, 16)) for image in map(Image.open, output_images)],
                [(255, 255, 255), (0, 0, 0), (255, 255, 255)],
            )
            for index in (0, 2):
                source_image = rgb / output_images[index].name
                self.assertEqual(
                    output_images[index].stat().st_mtime_ns,
                    source_image.stat().st_mtime_ns,
                )

    def test_episode_arguments_must_be_complete_and_ho3d_only(self):
        hard = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "positive context/masked"):
                hard.build_ho3d_hard_root(
                    Path(tmp) / "missing",
                    Path(tmp) / "out",
                    "mask",
                    0.7,
                    None,
                    7,
                    episode_context=1,
                )

            argv = [
                "make_hard_images.py",
                "--dataset",
                "freihand",
                "--input-root",
                str(Path(tmp) / "missing"),
                "--output-root",
                str(Path(tmp) / "out"),
                "--episode-context",
                "1",
                "--episode-mask",
                "1",
                "--episode-recovery",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(
                ValueError, "HO3D"
            ):
                hard.main()

    def test_episode_cli_lengths_default_to_disabled(self):
        hard = load_script()
        argv = [
            "make_hard_images.py",
            "--dataset",
            "ho3d",
            "--input-root",
            "in",
            "--output-root",
            "out",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = hard.parse_args()
        self.assertEqual(
            (args.episode_context, args.episode_mask, args.episode_recovery),
            (0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
