# 0035 P1 Batch B Strong-Recipe Jobs

Date: 2026-07-07

## Trigger

Hourly monitor found both prerequisite runs complete:

- P1-0 convergence scan: 12/12 summaries and 12/12 scores.
- Batch A flex5/gate scan: 8/8 summaries and 8/8 scores.

No existing `rope_p1_batch_b*` run was present before submission.

## P1-0 Recipe Choice

Selected recipe for Batch B:

```text
steps=400
lr=32
alpha_l2=0.001
max_alpha=0.5
```

Reason: `lr32_s400` was the clear convergence winner. The regularized and
unregularized variants were effectively tied; keep `alpha_l2=0.001` because
Batch B includes the higher-capacity `flex15`.

P1-0 key rows on FreiHAND mask70:

| Key | Closure | PA mean3d cm | All-joint delta cm | Clean-tip delta cm | Occluded-tip delta cm |
|---|---:|---:|---:|---:|---:|
| lr32_s400_l2_0 | 0.418428 | 0.866910 | -0.139864 | -0.217650 | -0.409507 |
| lr32_s400_l2_0p001 | 0.418293 | 0.866925 | -0.139849 | -0.217619 | -0.409455 |
| lr32_s120_l2_0p001 | 0.300581 | 0.899711 | -0.107064 | -0.160756 | -0.309093 |
| lr8_s400_l2_0p001 | 0.278306 | 0.906570 | -0.100205 | -0.149111 | -0.288685 |
| lr2_s120_l2_0p001 | 0.038380 | 0.990590 | -0.016184 | -0.020039 | -0.046782 |

The selected recipe passes the clean-finger guard: clean fingertip and clean
finger-joint deltas are still improvements, not regressions.

Batch A gating self-check also passed:

```text
mask70 gate010 frac_fingers_gated      0.397424
finger_end80 gate010 frac_fingers_gated 0.421111
```

## Batch B Run

Run root:

```text
/data/wentao/ropetrack/runs/rope_p1_batch_b_strong_20260707_061237
```

Cells:

```text
rope x {mult5, flex5, flex15} x {gateoff, gate010} x {mask70, finger_end80}
```

Inputs:

```text
mask70 export       /data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932/exports/mask70_wilor
finger_end80 export /data/wentao/ropetrack/runs/rope_optimization_freihand_extra_20260705/finger_end80_wilor/export
rope labels         /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/labels/freihand_rope.jsonl
```

Jobs:

| Split | Kind | Job | Dependency |
|---|---|---:|---:|
| mask70 | apply GPU | 169295 | - |
| mask70 | score CPU | 169296 | 169295 |
| finger_end80 | apply GPU | 169297 | - |
| finger_end80 | score CPU | 169298 | 169297 |

Both GPU jobs request `02:00:00`; CPU score jobs request `04:00:00`.

## Readout

When Batch B finishes, compare against 0032 and Batch A:

- Whether `flex5` beats both `mult5` and `flex15` under the strong recipe.
- Whether `gate010` preserves or improves clean-finger slices while keeping
  occluded-tip gains.
- Whether `gate010` keeps `gating.frac_fingers_gated` in the 15%-50% sanity
  range.
