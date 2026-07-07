# 0047 P2 Multi-v2 And P3 Feature Cache Results

Date: 2026-07-07

## Scope

This records two completed remote batches:

- P2 multi-v2: five-teacher student that adds HO3D v3 finger_end80 train teacher.
- P3 feature cache: frozen WiLoR backbone features for FreiHAND mask70 eval/train.

## Job Status

All jobs completed with exit code `0:0`.

| Job | Name | Elapsed | Purpose |
|---:|---|---:|---|
| 170342 | p2v2_r1d | 00:43:10 | HO3D v3 finger_end80 train hard root |
| 170343 | p2v2_r1t | 00:38:21 | HO3D v3 finger_end80 train teacher |
| 170344 | p2v2_r2 | 00:31:02 | HO3D v2 finger_end80 eval export + teacher |
| 170345 | p2v2_tr | 00:03:18 | five-teacher student train |
| 170346 | p2v2_ev | 00:08:31 | student eval cells |
| 170347 | p2v2_sc | 00:31:13 | score/slice summaries |
| 170401 | p3feat_eval | 00:04:53 | FreiHAND mask70 eval feature cache |
| 170402 | p3feat_train | 00:16:06 | FreiHAND mask70 train feature cache |

P2 multi-v2 run root:

```text
/data/wentao/ropetrack/runs/rope_p2_multi_v2_20260707_190953
```

P3 feature cache run root:

```text
/data/wentao/ropetrack/runs/rope_p3_feature_cache_20260707_200344
```

Local pulled summaries:

```text
.local_checks/remote_results_20260707_p2v2_p3feat/
```

## Five-teacher Training Health

Main student:

```text
epochs_run=78
best_epoch=57
best_val_loss=0.0097317
zero_baseline_val_l1=0.0511855
beats_zero_baseline=True
```

Shuffle control:

```text
epochs_run=158
best_epoch=137
best_val_loss=0.0479061
zero_baseline_val_l1=0.0511855
beats_zero_baseline=True
```

Teacher sources:

| Source | Samples |
|---|---:|
| FreiHAND mask70 train teacher | 32,560 |
| FreiHAND finger_end80 train teacher | 32,560 |
| HaMeR mask70 train teacher | 32,560 |
| HO3D v3 mask70 stride4 train teacher | 20,832 |
| HO3D v3 finger_end80 stride4 train teacher | 20,832 |
| Total | 139,344 |

## Multi-v2 Results

Values below are all-joint PA-aligned deltas in mm. Negative is better.

| Eval cell | Multi-v2 | Four-teacher multi (0044) | Interpretation |
|---|---:|---:|---|
| FreiHAND mask70 | -1.636 | -1.636 | tied |
| FreiHAND finger_end80 | -1.619 | -1.623 | tied/slightly worse |
| HO3D v2 mask70 | -0.834 | -0.972 | worse |
| HO3D v2 finger_end80 | -0.524 | - | new axis; positive but modest |
| HaMeR mask70 | -1.685 | -1.697 | tied/slightly worse |
| mask70 noise0.05 | -1.536 | -1.539 | tied |
| shuffled mask70 | -0.059 | -0.056 | collapsed as expected |

Other useful slices:

| Eval cell | Occluded-tip delta mm | Clean-tip delta mm | Closure | Gated fraction |
|---|---:|---:|---:|---:|
| v2_mask70 | -5.292 | -2.307 | 0.441 | 0.397 |
| v2_freihand_finger_end80 | -3.338 | - | 0.462 | 0.421 |
| v2_ho3d_mask70 | -1.953 | -0.808 | 0.400 | 0.537 |
| v2_ho3d_finger_end80 | -0.735 | - | 0.343 | 0.499 |
| v2_hamer_mask70 | -5.582 | -2.292 | 0.460 | 0.411 |
| v2_mask70_noise0p05 | -5.021 | -2.084 | 0.437 | 0.427 |
| v2_shuffled_mask70 | -0.091 | -0.090 | 0.011 | 0.397 |

## Interpretation

- Adding the fifth HO3D v3 finger_end80 teacher does **not** improve the
  generalist student. It preserves FreiHAND and noise performance, but loses
  the earlier HO3D mask70 gain from 0044.
- The new HO3D v2 finger_end80 eval cell is positive but modest:
  all-joint gain is -0.524 mm and occluded-tip gain is -0.735 mm.
- The shuffle control still collapses, so the student is still using rope
  signal rather than base-pose memorization.
- Keep the four-teacher augmented multi student from 0044 as the release/report
  model. Treat multi-v2 as a completed ablation, not the new default.

## P3 Feature Cache Results

Both feature caches were written successfully:

```text
/data/wentao/ropetrack/features/freihand_mask70_eval_wilor.npz   20M
/data/wentao/ropetrack/features/freihand_mask70_train_wilor.npz 160M
```

Logged shapes:

```text
FreiHAND mask70 eval:  features=(3960, 1280)
FreiHAND mask70 train: features=(32560, 1280)
```

These are P3 assets only. They are not a P3 model result yet.
