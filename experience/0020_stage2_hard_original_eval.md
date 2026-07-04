# 0020 Stage 2 Hard Splits Original Checkpoint Eval

Date: 2026-07-04

## Purpose

Generate the first full Stage 2 hard-image roots for HO3D v2 and FreiHAND, then
evaluate original WiLoR and original HaMeR checkpoints under GT-bbox protocol.

Hard roots:

- `/data/wentao/ropetrack/hard/ho3d_v2/mask70`
- `/data/wentao/ropetrack/hard/ho3d_v2/tip_square80`
- `/data/wentao/ropetrack/hard/ho3d_v2/finger_end80`
- `/data/wentao/ropetrack/hard/freihand/mask70`
- `/data/wentao/ropetrack/hard/freihand/tip_square80`
- `/data/wentao/ropetrack/hard/freihand/finger_end80`

## Jobs

Generation used CPU Slurm array job `164063`. The first serial generation job
`164055` was cancelled after confirming it was too slow for six sequential
splits.

Evaluation used GPU Slurm array job `164104`, with `--array=0-11%4` and one GPU
per task. It depended on `afterok:164063`.

Run root:

```text
/data/wentao/ropetrack/runs/hard_original_20260704_042200
```

All generation and evaluation tasks completed with `ExitCode 0:0`.

## Data Check

| Root | Images | Manifest rows |
|---|---:|---:|
| HO3D v2 mask70 | 11524 | 11524 |
| HO3D v2 tip_square80 | 11524 | 11524 |
| HO3D v2 finger_end80 | 11524 | 11524 |
| FreiHAND mask70 | 3960 | 3960 |
| FreiHAND tip_square80 | 3960 | 3960 |
| FreiHAND finger_end80 | 3960 | 3960 |

All 12 eval runs had zero prediction failures.

## HO3D v2 Scores

Mean errors are reported in millimetres. AUC/F-score values are unitless.

| Split | Method | AUCj | PA-MPJPE | AUCv | PA-MPVPE | F@5 | F@15 |
|---|---|---:|---:|---:|---:|---:|---:|
| mask70 | WiLoR original | 0.807757 | 9.623 | 0.807387 | 9.637 | 0.530544 | 0.962321 |
| mask70 | HaMeR original | 0.809748 | 9.518 | 0.807546 | 9.626 | 0.523190 | 0.962402 |
| tip_square80 | WiLoR original | 0.834616 | 8.274 | 0.829900 | 8.508 | 0.588893 | 0.975178 |
| tip_square80 | HaMeR original | 0.832187 | 8.394 | 0.827973 | 8.603 | 0.583444 | 0.974633 |
| finger_end80 | WiLoR original | 0.818791 | 9.067 | 0.818361 | 9.086 | 0.558373 | 0.967742 |
| finger_end80 | HaMeR original | 0.816132 | 9.199 | 0.814736 | 9.267 | 0.545208 | 0.966805 |

Relative to clean original baselines:

- `mask70` is strongest on HO3D v2: +2.09 mm PA-MPJPE for WiLoR and +1.78 mm
  for HaMeR.
- `finger_end80` is also meaningful: +1.53 mm for WiLoR and +1.46 mm for
  HaMeR.
- `tip_square80` is mild: about +0.7 mm for both methods.

## FreiHAND Scores

FreiHAND reports aligned paper-table metrics only.

| Split | Method | PA-MPJPE | PA-MPVPE | F@5 | F@15 |
|---|---|---:|---:|---:|---:|
| mask70 | WiLoR original | 10.485 | 10.051 | 0.580457 | 0.942570 |
| mask70 | HaMeR original | 11.148 | 10.852 | 0.537730 | 0.934168 |
| tip_square80 | WiLoR original | 7.295 | 6.887 | 0.713333 | 0.980966 |
| tip_square80 | HaMeR original | 8.289 | 7.958 | 0.649986 | 0.972993 |
| finger_end80 | WiLoR original | 10.363 | 9.974 | 0.568605 | 0.946413 |
| finger_end80 | HaMeR original | 11.457 | 11.114 | 0.523262 | 0.933952 |

Relative to clean original baselines:

- `mask70` and `finger_end80` are both strong on FreiHAND, adding about
  +5.4 to +5.9 mm PA-MPJPE.
- `tip_square80` is milder but still non-trivial, adding about +2.3 mm for
  WiLoR and +2.7 mm for HaMeR.

## Takeaway

The hard splits now show the intended behavior: corrupting the image inside the
GT-bbox lowers original-model performance with zero prediction failures.

For the next benchmark round, use:

- `mask70` as the broad severe occlusion split.
- `finger_end80` as the rope-like finger occlusion split.
- `tip_square80` as a mild/localized ablation, not the main hard condition.

## Diagnostic Visualizations

Reusable script:

```text
scripts/visualize_mesh_comparison.py
```

It reads a clean run, a hard run, the GT root, and optional clean/hard image
roots, then writes sheets with:

```text
clean image | hard image | GT mesh | clean pred mesh | hard pred mesh | overlay
```

Meshes are Procrustes-aligned for visualization, matching the aligned metric
interpretation and avoiding camera-frame drift.

Generated WiLoR original diagnostic sheets:

```text
/data/wentao/ropetrack/debug/mesh_compare/ho3d_v2_wilor_original_hard_sheets
/data/wentao/ropetrack/debug/mesh_compare/freihand_wilor_original_hard_sheets
```

Local copies:

```text
.local_checks/mesh_compare/ho3d_v2_wilor_original_hard_sheets
.local_checks/mesh_compare/freihand_wilor_original_hard_sheets
```

Each folder has 9 sheets:

```text
mask70_degradation.png
mask70_middle_degradation.png
mask70_low_degradation.png
tip_square80_degradation.png
tip_square80_middle_degradation.png
tip_square80_low_degradation.png
finger_end80_degradation.png
finger_end80_middle_degradation.png
finger_end80_low_degradation.png
```

Use the `degradation` sheets for clear failure cases, `middle_degradation` for
moderate examples, and `low_degradation` for controls where the perturbation
does not change the aligned mesh much.

The PNG sheets are intentionally not tracked by Git; keep only the script and
the run/debug paths in the repo.
