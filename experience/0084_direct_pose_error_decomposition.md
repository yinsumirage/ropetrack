# DirectPose existing-prediction error decomposition

Date: 2026-07-21

## Decision

**CONTINUE `delta cam_t` only. Do not build a universal
`delta global_orient + delta cam_t` branch.** Keep the current DirectPose head
responsible for local articulation and screen one separate 3D camera-translation
residual output. The previous global-orientation result remains rejected.

This is an oracle consistency result, not a learnability result. It proves that
translation-aligned evaluation removes a large, cross-dataset amount of the
current camera-frame metric and that DirectPose often reduces local residual
without reducing camera error. It does not prove that `delta cam_t` is
predictable from the frozen tokens or rope inputs.

LoRA remains stopped: the existing-prediction geometry provides no new evidence
for backbone adaptation.

## Scope and artifacts

No model was trained, no prediction was regenerated, no GPU was requested, no
new data was downloaded, and neither DexYCB test nor InterHand test was opened.
The analysis used the existing val/evaluation predictions and GT only, aligned
every method by `sample_id`, and left all source files unchanged.

- Remote root: `/data/wentao/ropetrack/runs/direct_pose_error_decomposition_20260721`
- Full CPU analysis: Slurm `189797`, `COMPLETED 0:0`, 4 CPU, peak RSS 3.84 GiB
- Independent verifier: Slurm `189817`, `COMPLETED 0:0`
- Machine-readable outputs: `inventory.json`, `per_sample.npz`,
  `summary.json`, `artifact_verification.json`
- Human/visual outputs: `summary.md`, `headroom.png`
- Bootstrap: 2,000 group resamples, seed `20260721`; units were sequence for
  ARCTIC/HO3D, episode for HOT3D, subject/sequence episode for DexYCB, and
  `frame_group_id` for InterHand so paired hands stayed together.

The inventory found 21 joints in metres for all five datasets. ARCTIC uses the
requested 3,888-row stride-10 GT subset; each prediction contains the full
38,921-row val set, giving exactly 3,888 matched and 35,033 extra IDs with no
missing requested ID. HOT3D (8,147), HO3D v3 (20,137), DexYCB val (46,859), and
InterHand corrected val (19,341) are exact ID matches. InterHand has 11--21
valid joints per row. Only InterHand has paired prediction/GT mesh for every
row; mesh fitting was kept secondary and was not mixed into the joint analysis.

## Metric boundary

All reported errors are the mean per-valid-joint Euclidean distance in mm.
`Camera`, root-relative, centroid-translation, proper Kabsch rigid, and proper
Procrustes candidates were computed per sample. Reflection is forbidden in the
proper transforms.

There is one important mathematical boundary. Centroid/Kabsch/Procrustes
minimize squared residual, whereas the requested MPJPE is a mean Euclidean
objective. Therefore their raw MPJPE candidates are not guaranteed to be
nested. The first strict runs correctly exposed this instead of silently
accepting negative headroom. The durable output retains every raw conventional
candidate and records every adjustment, while the named oracle chain is the
monotone transform-family envelope:

`E_T = min(E_camera, raw centroid-T)`,
`E_RT = min(E_T, raw proper-Kabsch)`, and
`E_Sim3 = min(E_RT, raw proper-Procrustes)`.

This makes `H_T/H_R/H_S/H_local` valid nested **oracle metric reductions**, not
causal components or learnable ceilings. Historical normal-data PA parity is
computed separately with the exact legacy `score_predictions.align_w_scale`
implementation, which can reflect; DexYCB/InterHand parity uses their proper
Procrustes scorer. The primary decomposition always forbids reflection.

## Parity gate

Every historical camera/root/PA target passed the 0.005 mm gate. Fold methods
were scored independently; the table reports each fold, then mean/range, and
never averages joints across checkpoints.

| Dataset | Checks | Max absolute difference (mm) | Status |
|---|---:|---:|---|
| ARCTIC | 24 | 7.2e-15 | PASS |
| HOT3D | 12 | 2.9e-14 | PASS |
| HO3D v3 | 24 | 4.6e-13 | PASS |
| DexYCB S1 val | 9 | 3.0e-6 | PASS |
| InterHand corrected val | 9 | 1.9e-7 | PASS |

## Core decomposition

These are the matched base and the representative DirectPose artifact selected
before inspecting the decomposition. ARCTIC and HO3D are three-checkpoint
means; their full ranges remain in `summary.json`/`summary.md`.

| Dataset | Method | N | Camera | Root | T | RT | Sim3/local |
|---|---|---:|---:|---:|---:|---:|---:|
| ARCTIC | WiLoR | 3,888 | 52.272 | 15.317 | 9.772 | 7.366 | 6.701 |
| ARCTIC | triple mean | 3,888 | 51.632 | 14.858 | 9.328 | 6.343 | 6.007 |
| HOT3D | WiLoR | 8,147 | 138.581 | 66.574 | 30.148 | 9.879 | 8.976 |
| HOT3D | triple stitched | 8,147 | 139.473 | 65.613 | 29.634 | 6.339 | 5.745 |
| HO3D v3 | WiLoR | 20,137 | 1646.927 | 14.571 | 10.436 | 7.552 | 7.130 |
| HO3D v3 | triple mean | 20,137 | 1647.313 | 13.835 | 9.783 | 6.818 | 6.543 |
| DexYCB S1 val | WiLoR | 46,859 | 55.378 | 8.779 | 6.908 | 5.479 | 5.227 |
| DexYCB S1 val | RGB-only | 46,859 | 55.370 | 9.363 | 7.130 | 5.429 | 5.194 |
| InterHand val | WiLoR | 19,341 | 81.994 | 19.062 | 12.229 | 10.521 | 8.801 |
| InterHand val | RGB-only | 19,341 | 82.715 | 18.703 | 11.875 | 10.203 | 8.415 |

For the representative DirectPose methods, the remaining nested headroom is:

| Dataset | Translation | Rotation | Scale | Local residual |
|---|---:|---:|---:|---:|
| ARCTIC | 42.304 | 2.985 | 0.335 | 6.007 |
| HOT3D | 109.838 | 23.296 | 0.593 | 5.745 |
| HO3D v3 | 1637.530 | 2.965 | 0.274 | 6.543 |
| DexYCB S1 val | 48.240 | 1.700 | 0.236 | 5.194 |
| InterHand val | 70.840 | 1.672 | 1.789 | 8.415 |

HO3D camera translation is a convention-dominated diagnostic and is not used
by magnitude to decide a universal branch. The cross-dataset fact is the layer,
not the 1.64 m absolute value. Rotation is a large fraction only on HOT3D;
ARCTIC has a smaller rotation term, and the other datasets do not support a
universal rotation branch.

## DirectPose effect and uncertainty

All deltas are DirectPose minus matched WiLoR base; negative error delta is an
improvement. CIs are the specified group bootstrap.

| Dataset | Method | dCamera [95% CI] | dRoot | dRT | dSim3/local [95% CI] | dH_T | dH_R |
|---|---|---:|---:|---:|---:|---:|---:|
| ARCTIC | triple mean | -0.640 [-0.891,-0.400] | -0.459 | -1.024 | -0.694 [-0.840,-0.555] | -0.195 | +0.579 |
| HOT3D | triple stitched | +0.891 [+0.490,+1.310] | -0.961 | -3.541 | -3.231 [-3.574,-2.890] | +1.405 | +3.027 |
| HO3D v3 | triple mean | +0.386 [-0.002,+0.743] | -0.736 | -0.735 | -0.587 [-0.838,-0.359] | +1.039 | +0.081 |
| DexYCB S1 val | RGB-only | -0.008 [-0.053,+0.040] | +0.583 | -0.049 | -0.034 [-0.100,+0.030] | -0.230 | +0.271 |
| InterHand val | RGB-only | +0.721 [+0.700,+0.741] | -0.359 | -0.318 | -0.386 [-0.401,-0.373] | +1.075 | -0.036 |

ARCTIC improves at every reported error layer, so it does not motivate a new
global branch. HOT3D, HO3D, and InterHand all show significant local/Sim3
improvement without a matching camera improvement; their additional mismatch
lands consistently in translation headroom. This is the three-dataset evidence
for a minimal translation screen. It is not evidence for `delta R`: the added
rotation headroom is large and significant on HOT3D, small/null elsewhere, and
the already-tested orientation head failed ARCTIC transfer.

The required subgroups are machine-readable in `summary.json`. The clearest
one is HOT3D triple: context has `H_T/H_R/Sim3 = 48.509/16.833/5.321` mm,
whereas low visibility has `168.704/29.498/6.152` mm. DexYCB includes subject,
camera, and visibility buckets; InterHand includes side, single/interacting,
MANO-valid/joint-only, capture, and camera. Per-joint camera/root/PA,
fingertip/non-tip, root translation, and reliable palm-orientation diagnostics
are also retained in `summary.json` and `per_sample.npz`.

## Rope and the DexYCB/InterHand reversal

The ideal-rope effect is not purely local, but its sign is dataset-specific:

| Dataset, rope minus RGB | dCamera | dRoot | dT | dRT | dSim3/local | dH_T | dH_R |
|---|---:|---:|---:|---:|---:|---:|---:|
| DexYCB val | -0.218 | -1.279 | -1.195 | -1.256 | -1.304 | +0.976 | +0.062 |
| InterHand val | +1.138 | -0.423 | +0.323 | +0.152 | +0.289 | +0.816 | +0.170 |

On DexYCB, ideal rope substantially improves local articulation and both
translation- and rigid-aligned error, while camera improves much less because
translation headroom grows. On InterHand, rope improves the wrist-root metric
but degrades T, RT, Sim3, and camera; the reversal therefore starts in the
local/shape-aligned layer and is amplified by translation, rather than being a
pure camera-convention effect. This preserves the prior decision to stop the
current InterHand rope recipe and not tune another final/test mixture.

## Minimal next experiment

If continued, the only new output should be a 3-vector `delta cam_t`, attached
to the frozen existing DirectPose/local-articulation path and evaluated as a
small matched validation screen. It should not output `delta global_orient`,
scale, MANO betas, another local-pose residual, temporal state, or any backbone
LoRA. The screen must compare camera, root, RT, and PA together and must not use
the already-observed DexYCB/InterHand test results for tuning.

## Failures and do-not-repeat notes

- Jobs `189761` and `189768` exposed the squared-loss-versus-MPJPE nesting
  mismatch. Do not enforce nesting by an arbitrary numerical tolerance.
- Job `189777` exposed the legacy reflective PA definition in the normal-data
  scorer. Do not weaken the proper primary transform for parity; reproduce the
  historical scorer only in the parity lane.
- Verifier `189815` used a 0.0001 mm summary/per-sample tolerance, tighter than
  float32 serialization at HO3D's camera scale. The final 0.001 mm aggregation
  tolerance remains stricter than the 0.005 mm parity gate and passed in
  verifier `189817`.
