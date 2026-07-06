# 0032 P0 WiLoR FreiHAND Results

Date: 2026-07-07

## Inputs

Run root:

```text
/data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932
```

Scope:

- backend: WiLoR original;
- splits: FreiHAND `mask70`, `finger_end80`;
- objectives: `rope`, `oracle_tip`;
- action spaces: `mult5`, `mult15`, `flex15`.

All jobs completed successfully. Each cell has `summary.json`,
`scores/scores.json`, `sliced/sliced_scores.json`, and
`deadzone/alpha_deadzone.json`.

## Headline Scores

Values below are PA-MPJPE in cm; deltas are against the base prediction from
the same apply output.

### mask70

| Objective | Action | PA cm | PA delta mm | Occluded fingertip delta mm | Rope closure |
|---|---|---:|---:|---:|---:|
| rope | mult5 | 0.990590 | -0.162 | -0.468 | 0.038 |
| rope | mult15 | 0.998195 | -0.086 | -0.233 | 0.020 |
| rope | flex15 | 0.997723 | -0.091 | -0.249 | 0.021 |
| oracle_tip | mult5 | 0.851743 | -1.550 | -4.580 | 0.245 |
| oracle_tip | mult15 | 0.869325 | -1.374 | -3.885 | 0.181 |
| oracle_tip | flex15 | 0.841720 | -1.651 | -5.210 | 0.222 |

### finger_end80

| Objective | Action | PA cm | PA delta mm | Occluded fingertip delta mm | Rope closure |
|---|---|---:|---:|---:|---:|
| rope | mult5 | 0.982047 | -0.156 | -0.295 | 0.037 |
| rope | mult15 | 0.989558 | -0.081 | -0.143 | 0.019 |
| rope | flex15 | 0.988428 | -0.092 | -0.175 | 0.021 |
| oracle_tip | mult5 | 0.831423 | -1.662 | -3.412 | 0.268 |
| oracle_tip | mult15 | 0.857668 | -1.400 | -2.749 | 0.190 |
| oracle_tip | flex15 | 0.806531 | -1.911 | -3.853 | 0.245 |

## Interpretation

- `oracle_tip` is much stronger than `rope`: roughly 10x-12x larger all-joint
  PA improvement, so the action spaces have real correction capacity.
- `flex15` has the best oracle ceiling on both splits, so additive flexion is
  not a dead end.
- Current `rope` objective does not exploit `flex15`; `rope/mult5` remains the
  best practical teacher in this run.
- Rope residual closure is low for rope objectives (`~2%-4%`) and much higher
  for oracle objectives (`~18%-27%`), implying the current rope objective /
  regularization / gating is the bottleneck, not merely the pose action space.
- Sliced gains are consistently larger on occluded fingertips than on all
  joints, supporting the report framing around local occluded-finger recovery
  rather than only all-joint averages.

## Decision

Do not train a rope-conditioned head from the current `flex15` rope teacher.
First run P1 objective/gating work:

1. keep `mult5` as the stable baseline teacher;
2. add residual/finger gating;
3. retest `flex15` under that gating;
4. only train after rope/flex or rope/mult5 becomes a stable teacher with
   stronger residual closure.
