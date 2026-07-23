# 0052 - E1 Calibration Tolerance and E2 Action-Space Scissors

Date: 2026-07-09

## Scope

Ran the approved E1/E2 batch; this note is the retained terminal specification
and result record.
E3 was not run.

Code prep was minimal:

- Added `pose45` to the shared action-space helpers.
- Added default-off rope calibration perturbations: `bias_std`,
  `bias_fixed`, and `scale_range`.
- Plumbed those perturbations through apply-time teacher/student mode and
  student-training augmentation options.
- Extended `summarize_runs.py` with bias/scale columns.

No `third_party/` code was edited.

## Verification

Local:

```text
python -m unittest discover -s tests
Ran 223 tests in 29.645s
OK
```

HPC login-node light check:

```text
python -m unittest discover -s tests
Ran 223 tests in 20.531s
OK
```

## HPC Jobs

Run roots:

- E1: `/data/wentao/ropetrack/runs/rope_e1_bias_20260709_041239`
- E2: `/data/wentao/ropetrack/runs/rope_e2_scissors_20260709_041239`

Jobs:

| Batch | Apply job | Score job |
|---|---:|---:|
| E1 calibration | 172522 | 172523 |
| E2 scissors | 172524 | 172525 |

Archived launchers:

- `~/project/ropetrack/.local_checks/submit_e1_calibration.sh`
- `~/project/ropetrack/.local_checks/submit_e2_scissors.sh`

Pulled local tables:

- `.local_checks/e1_bias_20260709_041239/tables/`
- `.local_checks/e2_scissors_20260709_041239/tables/`
- `.local_checks/e1_e2_summary_20260709_041239/`

## E2 Result: Action-Space Scissors

FreiHAND mask70, strong recipe fixed.

| Objective | Action | Gain (mm) | Occluded-tip gain (mm) | Closure |
|---|---:|---:|---:|---:|
| rope | mult5 | 1.548 | 4.715 | 0.402 |
| rope | flex15 | 1.679 | 5.472 | 0.485 |
| rope | pose45 | 1.769 | 5.888 | 0.526 |
| oracle_tip | mult5 | 1.825 | 5.698 | 0.279 |
| oracle_tip | flex15 | 1.969 | 7.935 | 0.417 |
| oracle_tip | pose45 | 3.147 | 11.983 | 0.568 |
| oracle_chain | mult5 | 1.586 | 4.901 | 0.233 |
| oracle_chain | flex15 | 1.825 | 7.198 | 0.374 |
| oracle_chain | pose45 | 3.388 | 11.640 | 0.464 |
| oracle_chain pose45, alpha_l2=0.004 | pose45 | 3.388 | 11.640 | 0.464 |

Decision:

- The `rope/pose45` gain is only +0.091 mm over `rope/flex15`, inside the
  pre-registered ±0.15 mm band. Extra action-space DOF does not materially
  help the 5-rope sensor.
- The oracle line opens strongly at 45D: `oracle_chain/pose45` reaches
  3.388 mm, about +1.62 mm over `rope/pose45`.
- The stronger P1 gate, `oracle_chain/pose45` beating `oracle_tip/flex15` by
  at least 0.3 mm, passed: 3.388 - 1.969 = 1.419 mm.
- The regularization sensitivity cell was identical at report precision:
  `alpha_l2=0.004` did not change the `oracle_chain/pose45` result.

Conclusion: 5-rope is the information bottleneck. A bigger correction module
alone is not the next lever; richer sensing or localized image evidence is.

## E1 Result: Calibration Tolerance

FreiHAND mask70 eval. Teacher baseline reference is the frozen winner
(`~1.679 mm` clean gain in this run family). Student baseline reference is the
release model (`1.636 mm` report-pack gain).

| Cell | Teacher gain (mm) | Student gain (mm) | Student retention |
|---|---:|---:|---:|
| `bias_std=0.025` | 1.595 | 1.609 | 98.3% |
| `bias_std=0.05` | 1.384 | 1.539 | 94.1% |
| `bias_std=0.075` | 1.087 | 1.411 | 86.3% |
| `bias_std=0.10` | 0.778 | 1.223 | 74.8% |
| `bias_fixed=+0.05` | 1.685 | 1.609 | 98.3% |
| `bias_fixed=-0.05` | 1.266 | 1.498 | 91.6% |
| `scale_range=0.05` | 1.618 | 1.617 | 98.9% |
| `scale_range=0.10` | 1.447 | 1.561 | 95.4% |
| `bias_std=0.05 + scale_range=0.05` | 1.329 | 1.511 | 92.3% |

Observations:

- Student retention at `bias_std=0.05` is 94.1%, well above the 60% mandatory
  retrain gate.
- The release student is more tolerant than the 400-step teacher under
  random bias and combined bias+scale in this batch.
- Coherent bias is sign-asymmetric: `+0.05` is nearly harmless, while `-0.05`
  weakens both paths. The student still retains 91.6%.
- Gated fraction is the fraction of valid fingers allowed to move. It rises
  with perturbation magnitude (`bias_std=0.025`: 0.406, `0.05`: 0.427,
  `0.075`: 0.459, `0.10`: 0.490), consistent with calibration error
  manufacturing residual and opening more gates.

Conclusion: no Phase-2 bias/scale augmentation retrain is required before a
controlled hardware mini-demo. Still calibrate sign/zero carefully because
negative coherent bias is the worst tested DC case.

## Next

Proceed to a controlled hardware mini-demo if the advisor wants hardware
evidence now. Keep E3 hardware-v2 pricing separate; it still needs explicit
user approval.
