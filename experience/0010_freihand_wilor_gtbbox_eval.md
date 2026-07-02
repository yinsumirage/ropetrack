# 0010 FreiHAND WiLoR GT BBox Eval

Date: 2026-07-03

## Purpose

Run the first FreiHAND evaluation baseline using AnyHand-WiLoR with ground-truth
bboxes.

## Local Code

Added:

```text
scripts/bench_freihand.py
tests/test_freihand_bench.py
```

The script intentionally supports only `gt_bbox` for now. It reuses the generic
cross-image crop/model batch path from `scripts/bench_ho3d.py` and writes the
same eval input shape:

```text
eval_input/pred.json
eval_input/evaluation_xyz.json
eval_input/evaluation_verts.json
```

## Protocol Fix

The first smoke job failed before inference:

```text
job: 162208
error: FreiHAND MANO joint protocol mismatch, max_err=0.163591m
```

Root cause: FreiHAND uses a different MANO joint order than the HO3D adapter,
and the index/middle fingertip vertices match `320` and `443`, not the HO3D
`333` and `444`.

FreiHAND adapter:

```text
tip ids: [744, 320, 443, 555, 672]
order: [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
```

After the fix, protocol check max error was:

```text
4.1733645161912136e-07 m
```

## Smoke

```text
job: 162209
run: /data/wentao/ropetrack/runs/freihand_wilor_gtbbox_smoke_20260703_012912_162209
state: COMPLETED
elapsed: 00:01:05
samples: 32
failures: 0
candidates: 32
```

## Full Prediction

```text
job: 162210
run: /data/wentao/ropetrack/runs/freihand_wilor_gtbbox_full_20260703_013213_162210
state: COMPLETED
elapsed: 00:03:56
samples: 3960
failures: 0
candidates: 3960
checkpoint: anyhand_wilor.ckpt
```

## CPU Eval

```text
job: 162224
out: /data/wentao/ropetrack/runs/freihand_wilor_gtbbox_full_20260703_013213_162210/eval_results_parallel_8cpu
state: COMPLETED
elapsed: 00:02:28
workers: 8
```

Scores:

```text
xyz_mean3d: 56.108925
xyz_auc3d: 0.001712
xyz_procrustes_al_mean3d: 0.495642
xyz_procrustes_al_auc3d: 0.901236
xyz_scale_trans_al_mean3d: 1.000077
xyz_scale_trans_al_auc3d: 0.801800
mesh_mean3d: 56.111460
mesh_auc3d: 0.001708
mesh_al_mean3d: 0.518074
mesh_al_auc3d: 0.896648
f_score_5: 0.001316
f_al_score_5: 0.817835
f_score_15: 0.004001
f_al_score_15: 0.993083
```

The evaluator reports mean distances in centimetres. The aligned scores are
approximately:

```text
PA-MPJPE: 4.96 mm
PA-MPVPE: 5.18 mm
```

The raw absolute errors are very large, so do not treat raw MPJPE/MPVPE as a
settled camera-coordinate metric yet. The first useful FreiHAND baseline signal
is the PA-aligned quality.

## Raw Coordinate Audit

Follow-up transform sweep on the saved full prediction:

```text
job: 162258, 162260
run: /data/wentao/ropetrack/runs/freihand_wilor_gtbbox_full_20260703_013213_162210
```

The sweep tested all 48 axis permutations and sign flips. The best fixed
rotation/reflection was identity:

```text
identity: xyz_mean3d 56.1089 cm, mesh_mean3d 56.1115 cm
HO3D-like sign flips/permutations: worse, often around 145 cm
```

Root-depth evidence:

```text
GT xyz root mean z:   0.712052 m
Pred xyz root mean z: 1.273396 m
GT verts root mean z:   0.708712 m
Pred verts root mean z: 1.270026 m
```

Simple diagnostics:

```text
global offset: xyz 17.3312 cm, mesh 17.3357 cm
global scale:  xyz 11.4218 cm, mesh 11.4530 cm
per-sample root translation: xyz 1.1413 cm, mesh 1.1268 cm
```

Interpretation: raw error is not a coordinate-axis rotation problem. FreiHAND is
already closest in the non-flipped OpenCV-style camera axes. The large raw
metric is dominated by absolute translation/depth from the model camera
recovery, likely focal/crop/camera-depth calibration. Do not add the HO3D
OpenGL y/z flip to FreiHAND. Treat PA metrics as the current valid comparison
and keep raw metrics marked unresolved until camera translation is audited.
