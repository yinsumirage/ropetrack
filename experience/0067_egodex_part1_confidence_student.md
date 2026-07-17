# EgoDex Part 1 confidence-filtered student pilot

Date: 2026-07-17

## Question and immutable inputs

This pilot asks whether EgoDex is useful for training the Flex15 alpha student,
and whether a simple fingertip-confidence filter prevents the earlier zero-shot
release student regression. Raw data under `/data/wentao/datasets/egodex` was
read only. All selection, images, labels, caches, checkpoints, and predictions
were written under `/data/wentao/ropetrack`.

The deterministic Part 1 view selected up to 50 episodes with native confidence
per task, prioritizing different sessions:

- 26 tasks, 1,227 episodes, and 779 task-session selections.
- 23,977 decoded stride-30 frames and 45,276 aligned hand rows/rope labels.
- This is 2.65% of Part 1's 46,234 episodes, not a full-Part-1 run.
- WiLoR produced all 45,276 rows with zero failures.

The frozen Flex15 teacher used the release recipe: rope objective, 400 steps,
learning rate 32, residual gate 0.1. Its mean rope residual fell from 0.2213 to
0.1111, with 49.8% closure and 87.9% of fingers improved in rope space.

## Training cells

All cells used the same teacher and split by episode sequence, with zero train
and validation sequence overlap.

| Student | Teacher rows | Fraction kept | Train / val rows | Train / val sequences |
|---|---:|---:|---:|---:|
| `all` | 45,276 | 100.0% | 40,807 / 4,469 | 1,102 / 123 |
| `min025` | 42,060 | 92.9% | 37,906 / 4,154 | 1,100 / 123 |
| `min050` | 18,538 | 40.9% | 16,478 / 2,060 | 1,051 / 117 |

Every student beat its zero-alpha validation baseline. The binary filters are
intentionally the smallest ablation; no confidence weighting, task balancing,
or full five-part training was added.

## Frozen EgoDex test result

All numbers are PA joint error in mm on the same 162,191-row stride-10 test and
the same WiLoR base.

| Method | Full test | Delta vs base | Part 1 tasks delta | Other 85 tasks delta |
|---|---:|---:|---:|---:|
| WiLoR base | 11.168 | - | - | - |
| `all` | 10.888 | -0.280 | -0.323 | -0.269 |
| `min025` | **10.868** | **-0.300** | **-0.344** | **-0.288** |
| `min050` | 10.875 | -0.293 | -0.323 | -0.285 |

The earlier frozen release student scored 11.552 mm on this test, or +0.384 mm
worse than base. Training a new student from the Part 1 EgoDex teacher therefore
changes the sign of the result. The gain also survives on the 85 tasks absent
from Part 1, so it is not explained only by task-name overlap.

Confidence filtering itself is not the main win: `min025` is only 0.020 mm
better than `all`, and `min050` is slightly worse than `min025` despite removing
59.1% of rows. Do not spend another experiment grid on nearby hard thresholds.

## Where the apparent gain comes from

| Native mean tip confidence | Rows | WiLoR base | `min025` delta |
|---|---:|---:|---:|
| `[0.00, 0.25)` | 9,881 | 14.138 | -1.573 |
| `[0.25, 0.50)` | 69,980 | 11.615 | -0.458 |
| `[0.50, 0.75)` | 60,461 | 10.211 | +0.081 |
| `[0.75, 0.90)` | 6,263 | 9.647 | +0.051 |
| `[0.90, 1.00]` | 478 | 8.922 | +0.292 |
| missing native confidence | 15,128 | 11.680 | -0.424 |

The student improves low/mid-confidence and missing-confidence rows but still
slightly degrades every native-confidence bin above 0.5. This is consistent
with learning a useful correction for hard hands, but it is also consistent
with fitting EgoDex's tracking-error distribution. Because EgoDex has no
independent MANO/mocap truth, the aggregate improvement is evidence of better
agreement with its ARKit skeleton, not proof of better physical hand pose.

## Decision

Continue with the `min025` checkpoint only as the pilot winner. Before scaling
to all 111 tasks or five full parts, apply this frozen checkpoint without
retraining to the existing FreiHAND and HO3D bases. If it preserves those
trusted protocols while improving EgoDex, then expand to a task-balanced
50-episode-per-task view across all five parts. If it regresses either trusted
protocol, stop: more EgoDex data would primarily teach dataset-specific label
bias rather than a general hand correction.

Do not start full five-part training yet, do not treat native confidence as a
deployment-time gate, and do not claim that the 0.300 mm EgoDex delta validates
MANO geometry.

## Run and recovery notes

- Outputs: `/data/wentao/ropetrack/runs/egodex_part1_pilot_20260717`.
- Preprocess `184591`; successful base `184721`; teacher `184722`; students
  `184723_0..2`; final score `184783`; raw integrity check `184791`.
- The post-run read-only check reproduced exactly 46,234 HDF5 files, 46,234
  MP4 files, and 360,483,493,561 bytes for Part 1, matching the pre-run audit.
- The first base attempt failed because the frozen code archive omitted the
  repo-local `mano_data` asset. Frozen snapshots must link both
  `pretrained_models` and `mano_data` before submission.
- The first score attempt completed standard scores but its ignored local-check
  helper could not import `ropetrack`. Snapshot-local helpers must set the
  snapshot root in `PYTHONPATH`.
