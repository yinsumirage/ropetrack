# 0038 P1 Guardrails, HO3D Winner, And Train-Teacher Results

Date: 2026-07-07

## Runs

Guardrails:

```text
/data/wentao/ropetrack/runs/rope_p1_guardrails_20260707_120955
```

HO3D winner:

```text
/data/wentao/ropetrack/runs/rope_p1_ho3d_winner_20260707_121314
```

Train teacher:

```text
/data/wentao/ropetrack/runs/rope_p2_train_teacher_20260707_121353
```

All jobs completed with exit code `0:0`.

## Noise Guardrail

Reference Batch B winner on FreiHAND mask70:

```text
rope + flex15 + gate010, steps=400 lr=32 alpha_l2=0.001
PA 0.838920 cm, all-joint delta -0.167855 cm, occluded-tip delta -0.547249 cm
```

Noise/dropout cells:

| Noise | Dropout | PA cm | All-joint delta cm | Retained all-joint gain | Occluded-tip delta cm | Gated fingers |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0 | 0.868348 | -0.138427 | 82.5% | -0.492320 | 0.427020 |
| 0.05 | 0.2 | 0.906015 | -0.100760 | 60.0% | -0.367266 | 0.427341 |
| 0.10 | 0.0 | 0.928993 | -0.077782 | 46.3% | -0.369216 | 0.490253 |
| 0.10 | 0.2 | 0.959842 | -0.046933 | 28.0% | -0.262481 | 0.491610 |
| 0.20 | 0.0 | 1.044612 | +0.037837 | fails | -0.128672 | 0.579848 |
| 0.20 | 0.2 | 1.063041 | +0.056267 | fails | -0.058482 | 0.582030 |

The P2 noise gate passes: `noise_std=0.05` keeps more than 60% of the clean
winner gain. With 20% dropout it is right on the threshold, so the report should
state the robustness claim as "moderate noise is acceptable, heavy noise is
not" rather than implying arbitrary sensor noise is safe.

At `noise_std=0.2`, all-joint PA regresses even though occluded-tip delta stays
slightly negative. This is the practical noise ceiling for the current strong
optimizer.

## Strong Oracle Ceiling

Strong-recipe oracle is now the valid ceiling, replacing the default-recipe
oracle rows from 0032.

| Split | Action | PA cm | All-joint delta cm | Occluded-tip delta cm | Closure |
|---|---|---:|---:|---:|---:|
| mask70 | flex15 | 0.809831 | -0.196944 | -0.793549 | 0.417441 |
| mask70 | mult5 | 0.824247 | -0.182528 | -0.569791 | 0.279297 |
| finger_end80 | flex15 | 0.749785 | -0.247873 | -0.582097 | 0.459291 |
| finger_end80 | mult5 | 0.782013 | -0.215644 | -0.447062 | 0.327372 |

Against the Batch B rope winner:

- mask70: rope/flex15/gate010 is `0.838920` PA cm, about `0.0291` cm
  (`0.29` mm) behind strong oracle/flex15. It keeps about 85% of the oracle
  all-joint improvement.
- finger_end80: rope/flex15/gate010 is `0.831759` PA cm, about `0.0820` cm
  (`0.82` mm) behind strong oracle/flex15. It keeps about 67% of the oracle
  all-joint improvement.

So rope is strong but not saturated. P2 still has meaningful headroom,
especially on finger_end80 and occluded-tip slices.

## Clean Split

Clean FreiHAND eval split:

| Cell | PA cm | Closure | Gated fingers | Alpha mean abs |
|---|---:|---:|---:|---:|
| rope_flex15_gateoff | 0.489705 | 0.448111 | - | 0.051025 |
| rope_flex15_gate010 | 0.505370 | 0.344425 | 0.294293 | 0.041012 |

The clean WiLoR baseline from 0011 is `0.495642` PA cm. With that reference,
gateoff is slightly better and gate010 is slightly worse on clean images.
This does not block the hard-split teacher, but it is a caveat for deployment:
the winner recipe should be presented as a hard/occluded teacher, not as a
universal clean-image improvement.

## HaMeR FreiHAND Backend

The same winner recipe works on HaMeR hard exports:

| Split | PA cm | All-joint delta cm | Occluded-tip delta cm | Closure | Gated fingers |
|---|---:|---:|---:|---:|---:|
| mask70 | 0.912448 | -0.169905 | -0.563564 | 0.512787 | 0.410657 |
| finger_end80 | 0.928930 | -0.184000 | -0.374615 | 0.525652 | 0.454596 |

This supports the claim that the effect is not WiLoR-only.

## HO3D Winner

The same FreiHAND-selected winner recipe was run on HO3D v2 mask70 without
retuning:

| Backend | PA cm | All-joint delta cm | Clean-tip delta cm | Occluded-tip delta cm | Closure | Gated fingers |
|---|---:|---:|---:|---:|---:|---:|
| WiLoR | 0.838888 | -0.104735 | -0.191955 | -0.208928 | 0.541515 | 0.536706 |
| HaMeR | 0.866616 | -0.065586 | -0.157064 | -0.081920 | 0.553965 | 0.500330 |

For comparison, the fixed-code P0 default rope rows on HO3D only closed about
1.4%-1.6% residual and improved PA by about 0.05-0.06 mm:

| Backend | P0 cell | PA cm | All-joint delta cm | Closure |
|---|---|---:|---:|---:|
| WiLoR | rope/mult5 | 0.937382 | -0.006241 | 0.016127 |
| WiLoR | rope/flex15 | 0.937678 | -0.005945 | 0.015523 |
| HaMeR | rope/mult5 | 0.927500 | -0.004702 | 0.014182 |
| HaMeR | rope/flex15 | 0.927466 | -0.004736 | 0.014876 |

The HO3D result confirms the P1 conclusion out of split: the winner recipe
transfers and the earlier small HO3D gains were also an optimization-strength
issue.

## Train Teacher

FreiHAND mask70 training teacher generation completed:

| Samples | Closure | Gated fingers | Alpha mean abs |
|---:|---:|---:|---:|
| 32560 | 0.503591 | 0.287918 | 0.052586 |

Teacher directory:

```text
/data/wentao/ropetrack/runs/rope_p2_train_teacher_20260707_121353/teacher/rope_flex15_gate010
```

This satisfies the train-teacher prerequisite for P2 distillation.

## P2 Gate Status

All three gates from 0037 pass:

1. HO3D supports the winner recipe without retuning.
2. Train-split teacher artifacts were produced successfully.
3. `noise_std=0.05` keeps at least 60% of the clean Batch B winner gain.

Recommended next step: start P2 distillation, but keep two controls mandatory:

- shuffled-rope student should lose the gain;
- clean-split eval should be monitored separately because gate010 is not a
  clean-image improvement in this run.
