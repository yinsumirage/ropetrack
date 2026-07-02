# 0004 HO3D v2 WiLoR Baseline

Date: 2026-07-02

## Runs

Detector run:

```text
/data/wentao/ropetrack/runs/ho3d_v2_wilor_detector_full_20260702_110436
Slurm: GPU 161474 cancelled after inference, CPU eval 161490 completed
mode: detector
num_samples: 11524
num_failures: 30
```

GT bbox run:

```text
/data/wentao/ropetrack/runs/ho3d_v2_wilor_gtbbox_full_20260702_115722
Slurm: GPU 161493 completed, CPU eval 161494 completed
mode: gt_bbox
num_samples: 11524
num_failures: 0
```

## Scores

HO3D eval writes mean errors in centimetres. The AnyHand paper table appears to
report the same aligned errors as millimetres, so multiply these cm values by
10 when comparing to PA-MPJPE / PA-MPVPE in the paper.

| Mode | PA joint mean | PA joint AUC | PA mesh mean | PA mesh AUC | aligned F@5 | aligned F@15 |
|---|---:|---:|---:|---:|---:|---:|
| detector | 4.544852 cm | 0.235423 | 0.787432 cm | 0.843035 | 0.639297 | 0.980031 |
| gt_bbox | 4.542710 cm | 0.235692 | 0.768836 cm | 0.846304 | 0.645197 | 0.983529 |

AnyHand Table 3 reports WiLoR w/ AnyHand roughly:

```text
AUC_j 0.853, PA-MPJPE 7.355, AUC_v 0.848, PA-MPVPE 7.624, F@5 0.649, F@15 0.984
```

## Interpretation

The aligned mesh metrics match the paper closely:

- `gt_bbox` PA-MPVPE: `0.768836 cm = 7.688 mm`, close to `7.624`.
- `gt_bbox` AUC_v: `0.846304`, close to `0.848`.
- `gt_bbox` aligned F@5/F@15: `0.645197/0.983529`, close to `0.649/0.984`.

The joint metrics do not match:

- `gt_bbox` PA joint mean is `4.542710 cm = 45.427 mm`, far from paper
  `7.355`.
- `gt_bbox` PA joint AUC is `0.235692`, far from paper `0.853`.

Because mesh metrics line up while joint metrics do not, the current vertex
export and aligned mesh evaluation are probably correct, but
`hand.keypoints_3d` is not in the HO3D official joint convention/order. Do not
use current `xyz_*` scores as final WiLoR joint benchmark numbers.

## Coordinate Note

HO3D submissions use an OpenGL-style camera frame:

```text
x: right
y: up
z: camera-forward is negative, so the hand is usually at z < 0
```

Convert ordinary OpenCV camera coordinates with:

```python
M_cv_to_ho3d = np.diag([1.0, -1.0, -1.0]).astype("float32")
points_ho3d = points_cv @ M_cv_to_ho3d.T
```

This matters for raw `xyz_mean3d`, `mesh_mean3d`, training targets, projection,
and any non-PA metric. It does not explain the original `45 mm` PA-MPJPE by
itself, because Procrustes alignment can absorb a global 180-degree rotation.
When PA-MPJPE is bad but PA-MPVPE is good, first suspect joint source/order.

## Joint Fix

Commit `8b6b5a9` changed HO3D export to derive joints from MANO vertices:

```text
16 joints = MANO_RIGHT.pkl J_regressor @ vertices
5 tips = vertices[[744, 333, 444, 555, 672]]
```

This is the HO3D/MANO-order adapter and avoids using AnyHand/WiLoR
`pred_keypoints_3d` directly. The first CPU eval jobs failed only because the
converted output directory lacked `eval_results/`; the metrics were printed
before the write failure.

Observed from the fixed gt-bbox eval log:

```text
xyz_procrustes_al_mean3d: 0.74 cm ~= 7.4 mm
xyz_procrustes_al_auc3d: 0.852
xyz_scale_trans_al_mean3d: 1.46 cm
xyz_scale_trans_al_auc3d: 0.712
mesh_al_mean3d: 0.77 cm
mesh_al_auc3d: 0.846
f_al_score_5: 0.645
f_al_score_15: 0.984
```

This matches the AnyHand table closely and confirms the previous joint result
was a joint-convention bug, not model quality.

## Next Action

Use `joint_source="mano_vertices"` for HO3D exports. Keep the original
AnyHand/WiLoR joints only as backend-native diagnostics, not as HO3D submission
joints.

Wait for the rerun CPU eval jobs to write exact `scores.txt` files:

```text
gt_bbox fixed eval: 161654
detector fixed eval: 161655
```

Original WiLoR final checkpoint jobs are still pending and should be checked
next:

| Mode | GPU job | Time limit | Expected window | Run dir |
|---|---:|---:|---|---|
| detector | 161691 | 01:00:00 | 2026-07-03 08:00-09:00 | `/data/wentao/ropetrack/runs/ho3d_v2_wilor_final_detector_pred_20260702_151639` |
| gt_bbox | 161692 | 01:00:00 | 2026-07-03 12:00-13:00 | `/data/wentao/ropetrack/runs/ho3d_v2_wilor_final_gtbbox_pred_20260702_151639` |

These are pred-only GPU jobs for
`third_party/anyhand/pretrained_models/wilor_final.ckpt`. After they finish,
run the parallel HO3D eval on each `eval_input` directory.

Update: jobs `161691` and `161692` timed out after their 1-hour limit. Retry
jobs were submitted with 3-hour GPU limits and dependent parallel CPU eval:

| Mode | Retry GPU job | CPU eval job | Run dir |
|---|---:|---:|---|
| detector | 161879 | 161880 | `/data/wentao/ropetrack/runs/ho3d_v2_wilor_final_detector_retry_20260702_195752` |
| gt_bbox | 161881 | 161882 | `/data/wentao/ropetrack/runs/ho3d_v2_wilor_final_gtbbox_retry_20260702_195752` |

WiLoR final gt-bbox retry completed:

```text
GPU pred: 161881 completed in 00:19:51
CPU eval: 161882 completed in 00:05:15
num_samples: 11524
num_failures: 0
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.753656 cm ~= 7.54 mm
xyz_procrustes_al_auc3d: 0.849366
xyz_scale_trans_al_mean3d: 1.488247 cm
xyz_scale_trans_al_auc3d: 0.706184
mesh_al_mean3d: 0.777823 cm ~= 7.78 mm
mesh_al_auc3d: 0.844507
f_al_score_5: 0.640977
f_al_score_15: 0.982330
```

WiLoR final detector retry completed:

```text
GPU pred: 161879 completed in 00:59:00
CPU eval: 161880 completed in 00:06:11
num_samples: 11524
num_failures: 30
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.770504 cm ~= 7.71 mm
xyz_procrustes_al_auc3d: 0.846357
xyz_scale_trans_al_mean3d: 1.515787 cm
xyz_scale_trans_al_auc3d: 0.704183
mesh_al_mean3d: 0.795209 cm ~= 7.95 mm
mesh_al_auc3d: 0.841490
f_al_score_5: 0.635685
f_al_score_15: 0.979001
```

Final fixed scores:

| Mode | PA joint mean | PA joint AUC | ST joint mean | ST joint AUC | PA mesh mean | PA mesh AUC | aligned F@5 | aligned F@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| detector fixed | 0.758737 cm | 0.848674 | 1.479608 cm | 0.710721 | 0.787432 cm | 0.843035 | 0.639297 | 0.980031 |
| gt_bbox fixed | 0.740556 cm | 0.851971 | 1.458248 cm | 0.711925 | 0.768836 cm | 0.846304 | 0.645197 | 0.983529 |

In paper units this is approximately:

```text
detector fixed: PA-MPJPE 7.59 mm, PA-MPVPE 7.87 mm, AUC_j 0.849, AUC_v 0.843
gt_bbox fixed:  PA-MPJPE 7.41 mm, PA-MPVPE 7.69 mm, AUC_j 0.852, AUC_v 0.846
```

This is the first credible WiLoR w/ AnyHand HO3D v2 baseline. The original
unfixed joint scores above should be kept only as a failure example.

Keep CPU eval separate from GPU inference. The official HO3D eval is a
single-process CPU loop and took about 36-37 minutes; extra Slurm CPUs do not
help unless the eval code is parallelized.

Parallel eval helper added:

```bash
python scripts/eval_ho3d_parallel.py \
  /data/wentao/ropetrack/runs/<run>/eval_input \
  /data/wentao/ropetrack/runs/<run>/eval_results_parallel \
  --version v2 \
  --num-workers 8
```

It writes `scores.txt` with the same field names as the official HO3D eval,
plus `scores.json`. Keep the official `third_party/ho3d_eval/eval.py` as the
reference implementation and use it for occasional exactness checks.

Verified on the fixed gt-bbox run:

```text
Slurm: CPU 161720 completed
workers: 16
elapsed: 00:04:57
output: /data/wentao/ropetrack/runs/ho3d_v2_wilor_gtbbox_joints_from_verts_20260702/eval_results_parallel_16cpu
check: diff against official eval_results/scores.txt was empty
```
