import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ropetrack.eval.protocols import FREIHAND_TIP_VERTEX_IDS
from ropetrack.refine.actions import FINGER_POSE_GROUPS
from ropetrack.rope import FINGER_CHAINS


ROOT = Path(__file__).resolve().parents[1]


def load_apply_script():
    path = ROOT / "scripts" / "rope_refiner" / "apply_rope_refinement.py"
    spec = importlib.util.spec_from_file_location("apply_rope_refinement", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def freihand_joints(tip_x: float = 4.0):
    joints = [[0.0, 0.0, 1.0] for _ in range(21)]
    for joint_id in range(1, 5):
        joints[joint_id] = [float(joint_id), 0.0, 1.0]
    joints[4] = [tip_x, 0.0, 1.0]
    return joints


def ho3d_joints(tip_x: float = 4.0):
    joints = [[0.0, 0.0, 1.0] for _ in range(21)]
    for joint_id in (13, 14, 15):
        joints[joint_id] = [1.0, 0.0, 1.0]
    joints[16] = [tip_x, 0.0, 1.0]
    return joints


BONE = 0.02


class FakeMano:
    """Two-segment toy hand: each fingertip = v + R(first finger joint) @ v.

    Wrist stays at the origin, so the rope distance of finger f is
    |v + R v| = 2 * BONE * cos(theta / 2) for a rotation of angle theta:
    curling the first joint shortens the rope, exactly like a real finger.
    Fingertip vertices mirror the tip joints so the oracle decode path works.
    """

    def __call__(self, global_orient=None, hand_pose=None, betas=None, pose2rot=False):
        batch = hand_pose.shape[0]
        bone = torch.tensor([BONE, 0.0, 0.0], dtype=hand_pose.dtype, device=hand_pose.device)
        joints = torch.zeros(batch, 21, 3, dtype=hand_pose.dtype, device=hand_pose.device)
        verts = torch.zeros(batch, 778, 3, dtype=hand_pose.dtype, device=hand_pose.device)
        for finger_idx, chain in enumerate(FINGER_CHAINS["freihand"]):
            first_joint = FINGER_POSE_GROUPS[finger_idx][0]
            rot = hand_pose[:, first_joint]
            tip = bone + torch.einsum("bij,j->bi", rot, bone)
            joints[:, chain[-1]] = tip
            verts[:, int(FREIHAND_TIP_VERTEX_IDS[finger_idx])] = tip
        return SimpleNamespace(joints=joints, vertices=verts)


class FakeManoTwoJoint:
    """Three-segment toy finger: tip = v + R_j0 @ (v + R_j1 @ v).

    Both the first and second pose joints of each finger move the rope
    distance, which distinguishes per-finger coupled normalization (flex5)
    from per-joint normalization (flex15).
    """

    def __call__(self, global_orient=None, hand_pose=None, betas=None, pose2rot=False):
        batch = hand_pose.shape[0]
        bone = torch.tensor([BONE, 0.0, 0.0], dtype=hand_pose.dtype, device=hand_pose.device)
        joints = torch.zeros(batch, 21, 3, dtype=hand_pose.dtype, device=hand_pose.device)
        verts = torch.zeros(batch, 778, 3, dtype=hand_pose.dtype, device=hand_pose.device)
        for finger_idx, chain in enumerate(FINGER_CHAINS["freihand"]):
            first, second, _ = FINGER_POSE_GROUPS[finger_idx]
            inner = bone + torch.einsum("bij,j->bi", hand_pose[:, second], bone)
            tip = bone + torch.einsum("bij,bj->bi", hand_pose[:, first], inner)
            joints[:, chain[-1]] = tip
            verts[:, int(FREIHAND_TIP_VERTEX_IDS[finger_idx])] = tip
        return SimpleNamespace(joints=joints, vertices=verts)


def toy_pose(num_samples: int, theta: float) -> np.ndarray:
    """Every finger's first joint rotated by theta around z."""
    pose = np.zeros((num_samples, 45), dtype=np.float32)
    for joints in FINGER_POSE_GROUPS:
        pose[:, 3 * joints[0] + 2] = theta
    return pose


def toy_rope_norm(theta: float, chain_m: float = 2.0 * BONE, fist_ratio: float = 0.5) -> float:
    dist = 2.0 * BONE * math.cos(theta / 2.0)
    lmin = fist_ratio * chain_m
    return (dist - lmin) / (chain_m - lmin)


def toy_cache(num_samples: int, base_theta: float, target_theta: float) -> dict:
    chain_m = 2.0 * BONE
    return {
        "sample_id": np.asarray([f"{i:08d}" for i in range(num_samples)]),
        "base_hand_pose": toy_pose(num_samples, base_theta),
        "base_rope_norm": np.full((num_samples, 5), toy_rope_norm(base_theta), dtype=np.float32),
        "input_rope_norm": np.full((num_samples, 5), toy_rope_norm(target_theta), dtype=np.float32),
        "gt_rope_norm": np.full((num_samples, 5), toy_rope_norm(target_theta), dtype=np.float32),
        "rope_chain_m": np.full((num_samples, 5), chain_m, dtype=np.float32),
        "rope_valid": np.ones((num_samples, 5), dtype=bool),
        "fist_ratio": np.full(num_samples, 0.5, dtype=np.float32),
    }


def write_toy_mano_cache(path: Path, num_samples: int) -> None:
    np.savez(
        path,
        sample_id=np.asarray([f"{i:08d}" for i in range(num_samples)]),
        base_hand_pose=toy_pose(num_samples, 0.0),
        base_global_orient=np.zeros((num_samples, 3), dtype=np.float32),
        base_betas=np.zeros((num_samples, 10), dtype=np.float32),
        base_cam_t=np.zeros((num_samples, 3), dtype=np.float32),
    )


def rope_residual_for_pose(script, pose_np: np.ndarray, target: np.ndarray) -> float:
    fake = FakeMano()
    with torch.no_grad():
        hand_pose = torch.from_numpy(pose_np.astype(np.float32))
        out = fake(hand_pose=script.torch_aa_to_rotmat(hand_pose.reshape(-1, 15, 3)))
        chain = torch.full((pose_np.shape[0], 5), 2.0 * BONE)
        fist = torch.full((pose_np.shape[0],), 0.5)
        pred = script.torch_rope_norm(out.joints, FINGER_CHAINS["freihand"], chain, fist)
    return float(np.mean(np.abs(pred.numpy() - target)))


class ApplyRopeRefinementTest(unittest.TestCase):
    def test_apply_finger_curl_alpha_only_changes_selected_finger_groups(self):
        script = load_apply_script()
        base = np.arange(45, dtype=np.float32).reshape(1, 45)
        alpha = np.asarray([[0.1, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        refined = script.apply_finger_curl_alpha(base, alpha)

        changed = np.flatnonzero(np.abs(refined[0] - base[0]) > 1e-6)
        expected = np.asarray([36, 37, 38, 39, 40, 41, 42, 43, 44])
        np.testing.assert_array_equal(changed, expected)

    def test_scipy_and_torch_rodrigues_agree(self):
        # the scipy version feeds the final decode, the torch version feeds
        # the optimizer; the same poses pass through both — pin their parity
        script = load_apply_script()
        rng = np.random.default_rng(29)
        aa = rng.normal(scale=1.2, size=(16, 15, 3)).astype(np.float32)
        expected = script.aa_to_rotmat(aa)
        actual = script.torch_aa_to_rotmat(torch.from_numpy(aa)).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_aa_to_rotmat_shapes_global_and_hand_pose(self):
        script = load_apply_script()

        global_rot = script.aa_to_rotmat(np.zeros((2, 3), dtype=np.float32))
        hand_rot = script.aa_to_rotmat(np.zeros((2, 15, 3), dtype=np.float32))

        self.assertEqual(global_rot.shape, (2, 3, 3))
        self.assertEqual(hand_rot.shape, (2, 15, 3, 3))
        np.testing.assert_allclose(global_rot[0], np.eye(3), atol=1e-6)

    def test_build_inference_cache_uses_run_meta_order_and_has_no_target_pose(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            pred_dir.mkdir()
            rope_labels = root / "rope.jsonl"
            run_meta = root / "run_meta.json"
            mano_cache = root / "mano_cache.npz"
            out = root / "cache.npz"

            rows = []
            for sid in ("00000000", "00000001"):
                rows.append({
                    "sample_id": sid,
                    "dataset": "freihand",
                    "rope_norm": [1.0, None, None, None, None],
                    "rope_chain_m": [4.0, None, None, None, None],
                    "rope_valid": [True, False, False, False, False],
                    "normalization": {"fist_ratio": 0.5},
                })
            rope_labels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            (pred_dir / "pred.json").write_text(json.dumps([
                [freihand_joints(3.0), freihand_joints(2.0)],
                [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
            ]), encoding="utf-8")
            run_meta.write_text(json.dumps({"sample_order": ["00000001", "00000000"]}), encoding="utf-8")
            np.savez(
                mano_cache,
                sample_id=np.asarray(["00000000", "00000001"]),
                base_hand_pose=np.stack([
                    np.full(45, 7.0, dtype=np.float32),
                    np.full(45, 9.0, dtype=np.float32),
                ]),
            )

            script.build_inference_cache("freihand", rope_labels, pred_dir, run_meta, mano_cache, out)

            with np.load(out) as data:
                self.assertEqual(data["sample_id"].tolist(), ["00000001", "00000000"])
                self.assertNotIn("target_hand_pose", data.files)
                np.testing.assert_array_equal(data["base_hand_pose"][0], np.full(45, 9.0, dtype=np.float32))
                self.assertEqual(data["base_rope_norm"].shape, (2, 5))
                self.assertEqual(data["input_rope_norm"].shape, (2, 5))
                self.assertIn("fist_ratio", data.files)
                np.testing.assert_allclose(data["fist_ratio"], [0.5, 0.5], atol=1e-7)

    def test_build_inference_cache_uses_ho3d_rope_chain(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred_dir = root / "pred"
            pred_dir.mkdir()
            rope_labels = root / "rope.jsonl"
            run_meta = root / "run_meta.json"
            mano_cache = root / "mano_cache.npz"
            out = root / "cache.npz"

            row = {
                "sample_id": "SM1/0000",
                "dataset": "ho3d",
                "rope_norm": [1.0, None, None, None, None],
                "rope_chain_m": [4.0, None, None, None, None],
                "rope_valid": [True, False, False, False, False],
                "normalization": {"fist_ratio": 0.5},
            }
            rope_labels.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (pred_dir / "pred.json").write_text(json.dumps([
                [ho3d_joints(4.0)],
                [[[0.0, 0.0, 0.0]]],
            ]), encoding="utf-8")
            run_meta.write_text(json.dumps({"sample_order": ["SM1/0000"]}), encoding="utf-8")
            np.savez(
                mano_cache,
                sample_id=np.asarray(["SM1/0000"]),
                base_hand_pose=np.zeros((1, 45), dtype=np.float32),
            )

            script.build_inference_cache("ho3d", rope_labels, pred_dir, run_meta, mano_cache, out)

            with np.load(out) as data:
                self.assertAlmostEqual(float(data["base_rope_norm"][0, 0]), 1.0)


class ParseArgsTest(unittest.TestCase):
    REQUIRED = [
        "--rope-labels", "rope.jsonl",
        "--pred-dir", "pred",
        "--run-meta", "run_meta.json",
        "--mano-cache", "mano_cache.npz",
        "--out-dir", "out",
    ]

    def test_defaults_are_published_working_recipe(self):
        script = load_apply_script()
        args = script.parse_args(self.REQUIRED)
        self.assertEqual(args.mode, "optimize")
        self.assertEqual(args.objective, "rope")
        self.assertEqual(args.action_space, "mult5")
        self.assertEqual(args.opt_steps, 120)
        self.assertAlmostEqual(args.opt_lr, 2.0)
        self.assertAlmostEqual(args.opt_alpha_l2, 0.001)
        self.assertAlmostEqual(args.opt_max_alpha, 0.5)
        self.assertIsNone(args.gate_residual_threshold)

    def test_pose45_and_bias_scale_cli_parse(self):
        script = load_apply_script()
        args = script.parse_args(self.REQUIRED + [
            "--action-space", "pose45",
            "--rope-noise-bias-std", "0.05",
            "--rope-noise-bias-fixed", "-0.05",
            "--rope-noise-scale-range", "0.1",
        ])
        self.assertEqual(args.action_space, "pose45")
        self.assertAlmostEqual(args.rope_noise_bias_std, 0.05)
        self.assertAlmostEqual(args.rope_noise_bias_fixed, -0.05)
        self.assertAlmostEqual(args.rope_noise_scale_range, 0.1)

    def test_oracle_requires_optimize_mode(self):
        script = load_apply_script()
        with self.assertRaises(ValueError):
            script.main(self.REQUIRED + ["--mode", "student", "--objective", "oracle_tip"])

    def test_oracle_requires_gt_xyz(self):
        script = load_apply_script()
        with self.assertRaises(ValueError):
            script.main(self.REQUIRED + ["--mode", "optimize", "--objective", "oracle_tip"])


class OptimizeAlphaToyTest(unittest.TestCase):
    """End-to-end optimization on a differentiable toy hand (no MANO files)."""

    NUM = 4
    BASE_THETA = 0.8
    TARGET_THETA = 1.1

    # Toy lr is much larger than the published real-data recipe (lr=2.0)
    # because the loss is a batch mean: per-sample gradients scale with
    # 1/(batch * 5 fingers), and the toy batch is tiny.
    def _run(self, script, tmp, action_space, objective="rope", gt_xyz=None, j_regressor=None, lr=20.0, steps=120, dataset="freihand", gate_threshold=None, cache=None):
        if cache is None:
            cache = toy_cache(self.NUM, self.BASE_THETA, self.TARGET_THETA)
        mano_cache = Path(tmp) / "mano_cache.npz"
        write_toy_mano_cache(mano_cache, self.NUM)
        return script.optimize_alpha(
            cache,
            mano_cache,
            "cpu",
            steps,
            lr,
            0.0005,
            0.5,
            8,
            dataset,
            action_space=action_space,
            objective=objective,
            gt_xyz=gt_xyz,
            j_regressor=j_regressor,
            gate_threshold=gate_threshold,
            mano_module=FakeMano(),
        ), cache

    def test_mult5_rope_objective_reduces_residual(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            (refined, alpha, directions), cache = self._run(script, tmp, "mult5")
            self.assertEqual(alpha.shape, (self.NUM, 5))
            self.assertIsNone(directions)
            target = cache["input_rope_norm"]
            base_res = rope_residual_for_pose(script, cache["base_hand_pose"], target)
            refined_res = rope_residual_for_pose(script, refined, target)
            self.assertLess(refined_res, 0.25 * base_res)
            # curling further means positive alpha under multiplicative scaling
            self.assertGreater(float(alpha.mean()), 0.0)

    def test_flex15_rope_objective_reduces_residual(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            (refined, alpha, directions), cache = self._run(script, tmp, "flex15")
            self.assertEqual(alpha.shape, (self.NUM, 15))
            self.assertEqual(directions.shape, (self.NUM, 15, 3))
            target = cache["input_rope_norm"]
            base_res = rope_residual_for_pose(script, cache["base_hand_pose"], target)
            refined_res = rope_residual_for_pose(script, refined, target)
            self.assertLess(refined_res, 0.5 * base_res)

    def test_flex5_rope_objective_reduces_residual(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            (refined, alpha, directions), cache = self._run(script, tmp, "flex5")
            self.assertEqual(alpha.shape, (self.NUM, 5))
            self.assertEqual(directions.shape, (self.NUM, 15, 3))
            target = cache["input_rope_norm"]
            base_res = rope_residual_for_pose(script, cache["base_hand_pose"], target)
            refined_res = rope_residual_for_pose(script, refined, target)
            self.assertLess(refined_res, 0.5 * base_res)

    def test_pose45_rope_objective_reduces_residual(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            (refined, alpha, directions), cache = self._run(script, tmp, "pose45", lr=20.0, steps=80)
            self.assertEqual(alpha.shape, (self.NUM, 45))
            self.assertIsNone(directions)
            target = cache["input_rope_norm"]
            base_res = rope_residual_for_pose(script, cache["base_hand_pose"], target)
            refined_res = rope_residual_for_pose(script, refined, target)
            self.assertLess(refined_res, 0.7 * base_res)

    def test_per_finger_direction_normalization(self):
        # With two active joints per finger, flex5 normalizes the finger's
        # concatenated 9-dim gradient to unit length, so individual joint
        # vectors are shorter than 1; flex15 normalizes each joint to 1.
        script = load_apply_script()
        base_pose = torch.from_numpy(toy_pose(self.NUM, self.BASE_THETA))
        common = (base_pose, torch.zeros(self.NUM, 3), torch.zeros(self.NUM, 10), FakeManoTwoJoint())
        per_joint = script.compute_flex_directions(*common, batch_size=8, per_finger=False)
        per_finger = script.compute_flex_directions(*common, batch_size=8, per_finger=True)
        for finger_idx in range(5):
            first, second, _ = FINGER_POSE_GROUPS[finger_idx]
            pj_norms = torch.linalg.norm(per_joint[:, [first, second]], dim=2)
            self.assertTrue(torch.allclose(pj_norms, torch.ones(self.NUM, 2), atol=1e-4))
            finger_vec = per_finger[:, list(FINGER_POSE_GROUPS[finger_idx])].reshape(self.NUM, -1)
            self.assertTrue(torch.allclose(torch.linalg.norm(finger_vec, dim=1), torch.ones(self.NUM), atol=1e-4))
            pf_norms = torch.linalg.norm(per_finger[:, [first, second]], dim=2)
            self.assertTrue(bool((pf_norms < 0.999).all()))
            self.assertTrue(bool((pf_norms > 1e-3).all()))

    def test_gating_freezes_ungated_fingers(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            # only the thumb has a rope residual; other fingers' targets equal
            # their base value, so a 0.05 threshold gates thumb only
            cache = toy_cache(self.NUM, self.BASE_THETA, self.TARGET_THETA)
            base_val = toy_rope_norm(self.BASE_THETA)
            target_row = np.asarray([toy_rope_norm(self.TARGET_THETA)] + [base_val] * 4, dtype=np.float32)
            cache["input_rope_norm"] = np.tile(target_row, (self.NUM, 1))
            (refined, alpha, _), cache = self._run(script, tmp, "mult5", gate_threshold=0.05, cache=cache)

            self.assertTrue(np.all(alpha[:, 1:] == 0.0))
            self.assertTrue(np.all(np.abs(alpha[:, 0]) > 1e-4))
            thumb_dims = [3 * j + a for j in FINGER_POSE_GROUPS[0] for a in range(3)]
            other_dims = [d for d in range(45) if d not in thumb_dims]
            np.testing.assert_allclose(refined[:, other_dims], cache["base_hand_pose"][:, other_dims], atol=1e-7)
            self.assertGreater(float(np.abs(refined[:, thumb_dims] - cache["base_hand_pose"][:, thumb_dims]).max()), 1e-4)

    def test_gating_masks_flex15_joint_columns(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            cache = toy_cache(self.NUM, self.BASE_THETA, self.TARGET_THETA)
            base_val = toy_rope_norm(self.BASE_THETA)
            target_row = np.asarray([toy_rope_norm(self.TARGET_THETA)] + [base_val] * 4, dtype=np.float32)
            cache["input_rope_norm"] = np.tile(target_row, (self.NUM, 1))
            (refined, alpha, _), cache = self._run(script, tmp, "flex15", gate_threshold=0.05, cache=cache)

            thumb_joints = list(FINGER_POSE_GROUPS[0])
            other_joints = [j for j in range(15) if j not in thumb_joints]
            self.assertTrue(np.all(alpha[:, other_joints] == 0.0))
            self.assertGreater(float(np.abs(alpha[:, thumb_joints]).max()), 1e-5)

    def test_gating_masks_pose45_finger_dims(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            cache = toy_cache(self.NUM, self.BASE_THETA, self.TARGET_THETA)
            base_val = toy_rope_norm(self.BASE_THETA)
            target_row = np.asarray([toy_rope_norm(self.TARGET_THETA)] + [base_val] * 4, dtype=np.float32)
            cache["input_rope_norm"] = np.tile(target_row, (self.NUM, 1))
            (refined, alpha, _), cache = self._run(script, tmp, "pose45", gate_threshold=0.05, cache=cache, lr=20.0, steps=80)

            thumb_dims = [3 * joint + axis for joint in FINGER_POSE_GROUPS[0] for axis in range(3)]
            other_dims = [d for d in range(45) if d not in thumb_dims]
            self.assertTrue(np.all(alpha[:, other_dims] == 0.0))
            self.assertGreater(float(np.abs(alpha[:, thumb_dims]).max()), 1e-5)

    def test_gate_from_cache_and_expand(self):
        script = load_apply_script()
        cache = {
            "base_rope_norm": np.asarray([[0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
            "input_rope_norm": np.asarray([[0.8, 0.52, 0.5, 0.5, 0.5]], dtype=np.float32),
            "rope_valid": np.asarray([[True, True, False, True, True]]),
        }
        gate = script.gate_from_cache(cache, 0.1)
        np.testing.assert_array_equal(gate, [[True, False, False, False, False]])
        # invalid finger never gated even with a huge residual
        cache["input_rope_norm"][0, 2] = 5.0
        gate = script.gate_from_cache(cache, 0.1)
        self.assertFalse(bool(gate[0, 2]))

        expanded = script.expand_gate_to_alpha(gate, "flex15")
        self.assertEqual(expanded.shape, (1, 15))
        for joint in FINGER_POSE_GROUPS[0]:
            self.assertEqual(expanded[0, joint], 1.0)
        self.assertAlmostEqual(float(expanded.sum()), 3.0)
        expanded45 = script.expand_gate_to_alpha(gate, "pose45")
        self.assertEqual(expanded45.shape, (1, 45))
        self.assertAlmostEqual(float(expanded45.sum()), 9.0)
        np.testing.assert_array_equal(script.expand_gate_to_alpha(gate, "mult5"), gate.astype(np.float32))

    def test_ho3d_dataset_uses_openpose_wrapper_chains(self):
        # Regression: the WiLoR MANO wrapper always emits OpenPose-ordered
        # joints. Indexing them with FINGER_CHAINS["ho3d"] (the old behavior)
        # made 4/5 finger residuals nonsense; optimization must converge for
        # --dataset ho3d exactly as it does for freihand.
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            (refined, alpha, _), cache = self._run(script, tmp, "mult5", dataset="ho3d")
            target = cache["input_rope_norm"]
            base_res = rope_residual_for_pose(script, cache["base_hand_pose"], target)
            refined_res = rope_residual_for_pose(script, refined, target)
            self.assertLess(refined_res, 0.25 * base_res)

    def test_flex_directions_unit_norm_on_active_joints(self):
        script = load_apply_script()
        base_pose = torch.from_numpy(toy_pose(self.NUM, self.BASE_THETA))
        directions = script.compute_flex_directions(
            base_pose,
            torch.zeros(self.NUM, 3),
            torch.zeros(self.NUM, 10),
            FakeMano(),
            batch_size=8,
        )
        norms = torch.linalg.norm(directions, dim=2)
        for finger_idx in range(5):
            first, second, third = FINGER_POSE_GROUPS[finger_idx]
            # only the first joint of each toy finger moves the rope
            self.assertTrue(torch.allclose(norms[:, first], torch.ones(self.NUM), atol=1e-4))
            self.assertTrue(torch.allclose(norms[:, second], torch.zeros(self.NUM), atol=1e-4))
            self.assertTrue(torch.allclose(norms[:, third], torch.zeros(self.NUM), atol=1e-4))

    def test_oracle_tip_objective_moves_tips_toward_gt(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            theta_t = self.TARGET_THETA
            tip_target = np.asarray(
                [BONE + BONE * math.cos(theta_t), BONE * math.sin(theta_t), 0.0], dtype=np.float32
            )
            gt = np.zeros((self.NUM, 21, 3), dtype=np.float32)
            for tip_joint in (4, 8, 12, 16, 20):
                gt[:, tip_joint] = tip_target
            j_regressor = np.zeros((16, 778), dtype=np.float32)

            (refined, alpha, _), cache = self._run(
                script, tmp, "mult5", objective="oracle_tip", gt_xyz=gt, j_regressor=j_regressor, lr=50.0, steps=100
            )

            def mean_tip_error(pose_np):
                fake = FakeMano()
                with torch.no_grad():
                    out = fake(hand_pose=script.torch_aa_to_rotmat(torch.from_numpy(pose_np).reshape(-1, 15, 3)))
                tips = out.joints[:, [4, 8, 12, 16, 20]].numpy()
                return float(np.linalg.norm(tips - gt[:, [4, 8, 12, 16, 20]], axis=2).mean())

            base_err = mean_tip_error(cache["base_hand_pose"])
            refined_err = mean_tip_error(refined)
            self.assertLess(refined_err, 0.7 * base_err)

    def test_unknown_action_space_raises(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._run(script, tmp, "free45")

    def test_oracle_requires_gt(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._run(script, tmp, "mult5", objective="oracle_tip")


class PerturbRopeCacheTest(unittest.TestCase):
    def _cache(self, num=16):
        return {
            "input_rope_norm": np.random.default_rng(3).uniform(0.2, 0.8, size=(num, 5)).astype(np.float32),
            "gt_rope_norm": np.full((num, 5), 0.5, dtype=np.float32),
            "rope_valid": np.ones((num, 5), dtype=bool),
        }

    def test_noop_when_disabled(self):
        script = load_apply_script()
        cache = self._cache()
        before = cache["input_rope_norm"].copy()
        script.perturb_rope_cache(cache, 0.0, 0.0, 7)
        np.testing.assert_array_equal(cache["input_rope_norm"], before)
        self.assertTrue(cache["rope_valid"].all())

    def test_noise_is_seeded_and_clamped(self):
        script = load_apply_script()
        cache_a, cache_b = self._cache(), self._cache()
        clean = cache_a["input_rope_norm"].copy()
        script.perturb_rope_cache(cache_a, 0.3, 0.0, 7)
        script.perturb_rope_cache(cache_b, 0.3, 0.0, 7)
        np.testing.assert_array_equal(cache_a["input_rope_norm"], cache_b["input_rope_norm"])
        self.assertGreater(float(np.abs(cache_a["input_rope_norm"] - clean).mean()), 0.05)
        self.assertGreaterEqual(float(cache_a["input_rope_norm"].min()), 0.0)
        self.assertLessEqual(float(cache_a["input_rope_norm"].max()), 1.0)
        # gt stays clean
        np.testing.assert_array_equal(cache_a["gt_rope_norm"], np.full((16, 5), 0.5, dtype=np.float32))

    def test_bias_scale_are_applied_to_cache(self):
        script = load_apply_script()
        cache = self._cache()
        clean = cache["input_rope_norm"].copy()
        script.perturb_rope_cache(
            cache, 0.0, 0.0, 7, bias_std=0.0, bias_fixed=0.05, scale_range=0.0
        )
        np.testing.assert_allclose(cache["input_rope_norm"], np.clip(clean + 0.05, 0.0, 1.0), atol=1e-7)

    def test_dropout_marks_invalid_and_zeroes_reading(self):
        script = load_apply_script()
        cache = self._cache(num=64)
        script.perturb_rope_cache(cache, 0.0, 0.5, 11)
        valid = cache["rope_valid"]
        frac = float(valid.mean())
        self.assertGreater(frac, 0.3)
        self.assertLess(frac, 0.7)
        self.assertTrue(np.all(cache["input_rope_norm"][~valid] == 0.0))

    def test_dropped_fingers_are_never_gated(self):
        script = load_apply_script()
        cache = {
            "base_rope_norm": np.full((8, 5), 0.2, dtype=np.float32),
            "input_rope_norm": np.full((8, 5), 0.8, dtype=np.float32),
            "gt_rope_norm": np.full((8, 5), 0.8, dtype=np.float32),
            "rope_valid": np.ones((8, 5), dtype=bool),
        }
        script.perturb_rope_cache(cache, 0.0, 0.6, 13)
        gate = script.gate_from_cache(cache, 0.1)
        self.assertFalse(bool(gate[~cache["rope_valid"]].any()))


class LoadManoGlobalsTest(unittest.TestCase):
    def test_reorders_by_sample_id(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mano_cache.npz"
            np.savez(
                path,
                sample_id=np.asarray(["a", "b", "c"]),
                base_global_orient=np.asarray([[0.0] * 3, [1.0] * 3, [2.0] * 3], dtype=np.float32),
                base_betas=np.asarray([[0.0] * 10, [1.0] * 10, [2.0] * 10], dtype=np.float32),
                base_cam_t=np.asarray([[0.0] * 3, [1.0] * 3, [2.0] * 3], dtype=np.float32),
            )
            orient, betas, cam_t = script.load_mano_globals(path, ["c", "a"])
            np.testing.assert_allclose(orient[:, 0], [2.0, 0.0])
            np.testing.assert_allclose(betas[:, 0], [2.0, 0.0])
            np.testing.assert_allclose(cam_t[:, 0], [2.0, 0.0])

    def test_missing_id_raises(self):
        script = load_apply_script()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mano_cache.npz"
            np.savez(
                path,
                sample_id=np.asarray(["a"]),
                base_global_orient=np.zeros((1, 3), dtype=np.float32),
                base_betas=np.zeros((1, 10), dtype=np.float32),
                base_cam_t=np.zeros((1, 3), dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                script.load_mano_globals(path, ["a", "z"])


class RopeResidualReportTest(unittest.TestCase):
    def test_report_from_decoded_joints(self):
        script = load_apply_script()
        cache = toy_cache(2, 0.8, 1.1)
        # decoded joints: put fingertips at the base/refined toy positions
        def joints_for_theta(theta):
            joints = np.zeros((21, 3), dtype=np.float32)
            tip = [BONE + BONE * math.cos(theta), BONE * math.sin(theta), 0.0]
            for tip_joint in (4, 8, 12, 16, 20):
                joints[tip_joint] = tip
            return joints.tolist()

        base_xyz = [joints_for_theta(0.8) for _ in range(2)]
        refined_xyz = [joints_for_theta(1.1) for _ in range(2)]
        summary, arrays = script.rope_residual_report("freihand", base_xyz, refined_xyz, cache)

        self.assertAlmostEqual(summary["refined"]["mean_abs"], 0.0, places=5)
        self.assertGreater(summary["base"]["mean_abs"], 0.1)
        self.assertAlmostEqual(summary["closure_frac"], 1.0, places=4)
        self.assertEqual(arrays["base_rope_residual"].shape, (2, 5))
        self.assertEqual(arrays["refined_rope_residual"].shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
