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

## Next Fix

Regenerate `pred.json` joint predictions from the exported MANO vertices using
the HO3D/MANO joint convention instead of AnyHand/WiLoR `pred_keypoints_3d`.
This can likely be done from the existing `pred.json` vertices without
rerunning GPU inference.

Keep CPU eval separate from GPU inference. The official HO3D eval is a
single-process CPU loop and took about 36-37 minutes; extra Slurm CPUs do not
help unless the eval code is parallelized.
