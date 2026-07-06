# 0036 P1 Batch B Strong-Recipe Results

Date: 2026-07-07

Run root:

```text
/data/wentao/ropetrack/runs/rope_p1_batch_b_strong_20260707_061237
```

Batch B used the P1-0-selected strong recipe:

```text
steps=400 lr=32 alpha_l2=0.001 max_alpha=0.5
```

All 12 cells completed with `summary.json`, `scores/scores.txt`, and
`sliced/sliced_scores.json`.

## Main Result

The under-optimization hypothesis was correct: strong rope optimization turns
the old small P0 rope gain into a large gain.

Compared with the 0032 default rope/mult5 row:

| Split | 0032 rope/mult5 PA cm | Batch B best PA cm | Delta vs base cm | Occluded-tip delta cm |
|---|---:|---:|---:|---:|
| mask70 | 0.990590 | 0.838920 | -0.167855 | -0.547249 |
| finger_end80 | 0.982047 | 0.831759 | -0.165899 | -0.344943 |

Best Batch B cell on both splits is `rope_flex15_gate010`, not `flex5`.

## Mask70

| Cell | Closure | PA mean3d cm | All-joint delta cm | Clean-tip delta cm | Occluded-tip delta cm | Gated fingers |
|---|---:|---:|---:|---:|---:|---:|
| rope_flex15_gate010 | 0.485126 | 0.838920 | -0.167855 | -0.237319 | -0.547249 | 0.397424 |
| rope_mult5_gate010 | 0.401761 | 0.851991 | -0.154784 | -0.231079 | -0.471492 | 0.397424 |
| rope_flex5_gate010 | 0.407337 | 0.864864 | -0.141910 | -0.198809 | -0.449107 | 0.397424 |
| rope_mult5_gateoff | 0.418293 | 0.866925 | -0.139849 | -0.217619 | -0.409455 | - |
| rope_flex15_gateoff | 0.497426 | 0.867306 | -0.139469 | -0.191445 | -0.459069 | - |
| rope_flex5_gateoff | 0.437264 | 0.887732 | -0.119042 | -0.162085 | -0.384105 | - |

Gating passes the sanity check: `frac_fingers_gated=0.397424`, within the
15%-50% target band. It also improves every action-space on PA and occluded
tips under the strong recipe.

## Finger End80

All five fingers are occluded in this split, so clean-finger slices are empty.

| Cell | Closure | PA mean3d cm | All-joint delta cm | Occluded-tip delta cm | Gated fingers |
|---|---:|---:|---:|---:|---:|
| rope_flex15_gate010 | 0.506111 | 0.831759 | -0.165899 | -0.344943 | 0.421111 |
| rope_mult5_gate010 | 0.421116 | 0.843402 | -0.154256 | -0.320319 | 0.421111 |
| rope_flex5_gate010 | 0.430844 | 0.853210 | -0.144447 | -0.295862 | 0.421111 |
| rope_flex15_gateoff | 0.514350 | 0.855219 | -0.142439 | -0.294717 | - |
| rope_mult5_gateoff | 0.433041 | 0.857749 | -0.139909 | -0.290682 | - |
| rope_flex5_gateoff | 0.454470 | 0.872742 | -0.124916 | -0.256368 | - |

Gating also passes here: `frac_fingers_gated=0.421111`.

## Interpretation

Confirmed:

- P1-0 was necessary. The default recipe only closed about 3.8% rope residual;
  the selected recipe closes about 40%-51%.
- Gating matters once optimization is strong. It improves PA for all three
  action spaces and does not cause clean-finger regression on mask70.
- Rope-only test-time optimization is now close to, and on mask70 slightly
  better than, the default-recipe oracle_tip/flex15 row from 0032.

Rejected:

- The specific prediction that `flex5` would beat both `mult5` and `flex15`.
  It does not. `flex5` beats its weak default Batch A version, but under the
  strong recipe it is still behind `mult5_gate010` and `flex15_gate010` on PA.

Current best teacher candidate for distillation:

```text
rope + flex15 + gate_residual_threshold=0.1 + steps=400/lr=32/alpha_l2=0.001
```

One caveat: `flex15` has higher closure than `mult5`, but the best PA comes
from gated `flex15`, not ungated `flex15`. The gate should stay attached if
this teacher is exported or distilled.
