# 0005 HO3D v3 WiLoR Jobs

Date: 2026-07-02

## Purpose

Run HO3D v3 evaluation predictions for:

- AnyHand WiLoR checkpoint, detector bbox.
- AnyHand WiLoR checkpoint, GT bbox.
- Original WiLoR checkpoint, detector bbox.
- Original WiLoR checkpoint, GT bbox.

## Data Notes

HO3D v3 root:

```text
/data/wentao/ropetrack/HO3D_v3
```

The evaluation split has 20137 samples and uses `.jpg` images. The benchmark
export script was updated to resolve `.png`, `.jpg`, or `.jpeg`.

The GT JSON files must be complete before dependent CPU eval jobs start:

```text
/data/wentao/ropetrack/HO3D_v3/evaluation_xyz.json
/data/wentao/ropetrack/HO3D_v3/evaluation_verts.json
```

The CPU eval scripts validate both JSON lengths against 20137 and fail fast if
the upload is incomplete.

## Jobs

Submitted with GPU time limit shortened to 2 hours.

| Experiment | Run dir | GPU job | CPU eval job |
|---|---|---:|---:|
| AnyHand WiLoR detector | `/data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_detector_20260702_161953` | 161779 | 161780 |
| AnyHand WiLoR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_gtbbox_20260702_161953` | 161781 | 161782 |
| WiLoR final detector | `/data/wentao/ropetrack/runs/ho3d_v3_wilor_final_detector_20260702_161953` | 161783 | 161784 |
| WiLoR final gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_wilor_final_gtbbox_20260702_161953` | 161785 | 161786 |

Each CPU eval job has an `afterok` dependency on its corresponding GPU job and
runs:

```bash
python scripts/eval_ho3d_parallel.py \
  <run>/eval_input \
  <run>/eval_results_parallel_16cpu \
  --version v3 \
  --num-workers 16 \
  --chunksize 16
```

## Retry Note

The original gt-bbox submissions used the invalid mode string `gtbbox`; the
script only accepts `gt_bbox`. The bad GPU jobs failed immediately and their
dependent CPU eval jobs were cancelled:

```text
161781 anyhand_wilor/gtbbox failed: invalid --mode gtbbox
161782 cancelled: dependency never satisfied
161785 wilor_final/gtbbox failed: invalid --mode gtbbox
161786 cancelled: dependency never satisfied
```

Retry jobs with the correct `--mode gt_bbox` and 3-hour GPU limits:

| Experiment | Run dir | GPU job | CPU eval job |
|---|---|---:|---:|
| AnyHand WiLoR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_gtbbox_retry_20260702_195752` | 161883 | 161884 |
| WiLoR final gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_wilor_final_gtbbox_retry_20260702_195752` | 161885 | 161886 |

## Completed Results

AnyHand WiLoR detector completed:

```text
run: /data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_detector_20260702_161953
GPU pred: 161779 completed in 01:21:24
CPU eval: 161780 completed in 00:08:14
num_samples: 20137
num_failures: 66
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.714346 cm ~= 7.14 mm
xyz_procrustes_al_auc3d: 0.857561
xyz_scale_trans_al_mean3d: 1.429216 cm
xyz_scale_trans_al_auc3d: 0.721412
mesh_al_mean3d: 0.668057 cm ~= 6.68 mm
mesh_al_auc3d: 0.866959
f_al_score_5: 0.694082
f_al_score_15: 0.979677
```

AnyHand WiLoR gt-bbox completed:

```text
run: /data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_gtbbox_retry_20260702_195752
GPU pred: 161883 completed in 00:47:31
CPU eval: 161884 completed in 00:07:23
num_samples: 20137
num_failures: 0
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.693829 cm ~= 6.94 mm
xyz_procrustes_al_auc3d: 0.861273
xyz_scale_trans_al_mean3d: 1.400026 cm
xyz_scale_trans_al_auc3d: 0.723425
mesh_al_mean3d: 0.646108 cm ~= 6.46 mm
mesh_al_auc3d: 0.870803
f_al_score_5: 0.700943
f_al_score_15: 0.983929
```

WiLoR final detector completed:

```text
run: /data/wentao/ropetrack/runs/ho3d_v3_wilor_final_detector_20260702_161953
GPU pred: 161783 completed in 01:32:54
CPU eval: 161784 completed in 00:08:34
num_samples: 20137
num_failures: 66
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.731226 cm ~= 7.31 mm
xyz_procrustes_al_auc3d: 0.854195
xyz_scale_trans_al_mean3d: 1.463014 cm
xyz_scale_trans_al_auc3d: 0.715313
mesh_al_mean3d: 0.683100 cm ~= 6.83 mm
mesh_al_auc3d: 0.863956
f_al_score_5: 0.684130
f_al_score_15: 0.979185
```

WiLoR final gt-bbox completed:

```text
run: /data/wentao/ropetrack/runs/ho3d_v3_wilor_final_gtbbox_retry_20260702_195752
GPU pred: 161885 completed in 00:47:31
CPU eval: 161886 completed in 00:07:24
num_samples: 20137
num_failures: 0
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.708798 cm ~= 7.09 mm
xyz_procrustes_al_auc3d: 0.858267
xyz_scale_trans_al_mean3d: 1.430366 cm
xyz_scale_trans_al_auc3d: 0.717750
mesh_al_mean3d: 0.659566 cm ~= 6.60 mm
mesh_al_auc3d: 0.868113
f_al_score_5: 0.691721
f_al_score_15: 0.983520
```
