# Frozen visibility-gate domain transfer

Date: 2026-07-15

## Question

Does the minimal image-linear visibility gate from `0057` preserve its causal
state gain when the masked phase changes from the training-time black mask to
different corruption families, without retraining the gate or pose refiner?

## Protocol

- Frozen gate checkpoint:
  `/data/wentao/ropetrack/runs/temporal_state_followups_20260715/image_visibility_gate/image_linear.pt`.
- Gate threshold: the validation zero-false-clean threshold `0.20424594`.
- Transfer run root:
  `/data/wentao/ropetrack/runs/temporal_visibility_transfer_20260715`.
- Shift screen run root:
  `/data/wentao/ropetrack/runs/temporal_visibility_shift_20260715`.
- Evaluation keeps the same HO3D v3 sample order and 30 clean, 60 masked,
  30 recovery episode schedule as `0056` and `0057`.
- Clean and recovery rows use the original image-gate scores. Only masked rows
  receive scores extracted from the shifted images. Phase is used to construct
  this controlled transfer test, not by the causal state application.
- The causal state updates on a predicted-clean frame and otherwise holds the
  last predicted-clean pose. Current rope plus the frozen dense K1 refiner
  corrects flexion on held frames.
- All pose reports use 153 complete episodes, 13 sequences, and 2,000 paired
  sequence bootstrap draws.

Implementation commits:

- `ba6176ddab5110f4fa4a41ecbc55c64996473055`: frozen shift evaluator.
- `a4965918408bc29c688b76cd1631ef18bcbdb650`: phase-consistent shifted-score
  composition and single-gate state application.
- `da971def3a2b944f7168aeec1eaf32c5c6004dd8`: represent localized fingertip
  metrics as undefined for non-localized effects such as crop instead of
  aborting all phase-aware scoring.

## Frozen-gate screen

The first screen applies each corruption to every eval row. False-clean means
the gate would incorrectly update trusted state from a corrupted image.

| Shift | Default false-clean | Default three-frame runs | Zero-FN false-clean | Zero-FN three-frame runs | Decision |
|---|---:|---:|---:|---:|---|
| mask40 | 60.42% | 13 sequences | 39.76% | 13 sequences | stop |
| mask70 | 0.005% | 0 | 0 | 0 | control pass |
| mask90 | 0 | 0 | 0 | 0 | pass |
| finger_end80 | 0.348% | 6 sequences | 0 | 0 | promote zero-FN |
| tip_square80 | 58.03% | 13 sequences | 42.28% | 13 sequences | stop |
| blur70 | 45.77% | 13 sequences | 34.88% | 10 sequences | stop |
| crop70 | 0.050% | 0 | 0.005% | 0 | promote zero-FN |

Only `finger_end80` and `crop70` received new WiLoR exports and full temporal
pose evaluation. The failed shifts were not rescued with a threshold grid or
larger classifier.

## Gate behavior in temporal episodes

| Shift | Masked false-clean | First-mask miss | Episodes with any false-clean | Three-frame runs | Clean false-freeze |
|---|---:|---:|---:|---:|---:|
| finger_end80 | 0 | 0 | 0 | 0 | 9.39% |
| crop70 | 0.0109% | 0 | 1 | 0 | 9.39% |

The crop miss is one isolated masked update, not a consecutive pollution run.
Both domains detect every first masked frame and retain about 90.6% of clean
frames for state refresh.

## Full temporal pose results

### Finger-end 80

| Method | Overall | Context | Masked | Recovery | Masked tips | Masked velocity | Masked acceleration | Masked jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 8.1275 | 7.2066 | 9.2097 | 7.1424 | 13.9655 | 121.19 | 5794 | 5686 |
| K1 | 7.6051 | 6.8695 | 8.3680 | 6.7900 | 11.7291 | 124.15 | 5985 | 5901 |
| frozen gate state + K1 | **7.0510** | **6.7683** | **7.1175** | 6.7974 | **10.0828** | **110.93** | **5289** | **5175** |
| state + shuffled rope | 8.1472 | 6.9245 | 9.3040 | 7.0136 | 14.9805 | 245.74 | 12748 | 12659 |

The state gain over K1 is 1.2505 mm masked PA, with state-minus-K1 CI
`[-1.5941, -0.9359]` mm. Masked occluded tips improve by 1.6463 mm with CI
`[-2.1604, -1.2074]` mm. Masked velocity, acceleration, and jitter improve by
10.64%, 11.62%, and 12.30%, respectively. Recovery PA is statistically tied:
delta +0.0074 mm, CI `[-0.1240, +0.1419]` mm.

| Masked horizon | K1 | Frozen gate state | Gain |
|---:|---:|---:|---:|
| 1 | 8.5449 | 6.7509 | 1.7940 mm |
| 5 | 8.6590 | 6.8098 | 1.8492 mm |
| 15 | 8.7456 | 6.9825 | 1.7631 mm |
| 30 | 8.0801 | 7.0418 | 1.0382 mm |
| 60 | 8.3614 | 7.4945 | 0.8669 mm |

The benefit remains positive through frame 60. Shuffling rope worsens masked
PA by 2.1865 mm relative to the correct-rope state, paired CI
`[+1.9071, +2.4090]` mm, so the gain is not pose copying alone.

### Crop 70

| Method | Overall | Context | Masked | Recovery | Masked velocity | Masked acceleration | Masked jitter |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 8.5380 | 7.2066 | 10.1103 | 7.1424 | 567.74 | 30639 | 30597 |
| K1 | 7.9048 | 6.8695 | 9.0254 | 6.7900 | 556.36 | 30003 | 29969 |
| frozen gate state + K1 | **7.0527** | **6.7683** | **7.1211** | 6.7974 | **537.85** | **28986** | **28951** |
| state + shuffled rope | 8.1505 | 6.9263 | 9.3104 | 7.0134 | 615.55 | 32768 | 32728 |

The state gain over K1 is 1.9043 mm masked PA, with state-minus-K1 CI
`[-2.3558, -1.5421]` mm. Masked velocity, acceleration, and jitter improve by
3.33%, 3.39%, and 3.40%; they do not reach the preferred 10% improvement but
do not regress. Recovery PA is tied with K1 using the same CI as finger-end.

| Masked horizon | K1 | Frozen gate state | Gain |
|---:|---:|---:|---:|
| 1 | 8.8572 | 6.7497 | 2.1075 mm |
| 5 | 9.1567 | 6.8284 | 2.3283 mm |
| 15 | 9.1893 | 7.0098 | 2.1795 mm |
| 30 | 8.8221 | 7.0390 | 1.7832 mm |
| 60 | 9.1120 | 7.5208 | 1.5912 mm |

Correct rope beats shuffled rope by 2.1893 mm masked PA, paired CI
`[+1.9004, +2.4104]` mm. Crop is not a finger-localized effect, so a masked
occluded-tip number is not identifiable from its manifest. The report records
`masked_occluded_tip_pa_mpjpe_mm: null` and
`masked_occluded_tip_defined: false`; it does not relabel all tips as occluded.

## Decision

The state-management claim transfers without retraining from black `mask70`
to both a localized `finger_end80` corruption and a global `crop70`
corruption. Both exceed the 0.15 mm promotion bar by more than 1.2 mm, their
paired CIs exclude zero, every measured horizon through frame 60 improves,
and dynamics improve rather than trading accuracy for excess smoothing.
Correct rope is essential in both domains.

This establishes a real upper path for a minimal learned tracker: frozen image
quality decides whether to refresh one explicit trusted pose; motion history
and large sequence models are unnecessary; current rope only corrects flexion
after state selection. The current gate is not generally robust: light mask,
tip-square, and blur shifts have catastrophic false-clean runs. Do not claim
physical deployment and do not scale the pose model. The next model change, if
pursued, should be confined to visibility training coverage for those failed
appearance families, followed by the same frozen transfer gate.

## Jobs and failures

- Screen: `181373_[0-4] -> 181374_[0-4] -> 181375_[0-4]` and
  `181383_[0-1] -> 181384_[0-1] -> 181385_[0-1]`.
- Temporal hard roots: `181435_[0-1]`.
- Full transfer: `181486_[0-1] -> 181487_[0-1] -> 181488_[0-1] -> 181489_[0-1]`.
- `181477_[0-1]` failed at a submodule-revision preflight that queried the
  parent worktree. The retry checks the submodule gitdir directly; no model or
  data setting changed.
- `181489_1` failed because crop has no defined per-finger occlusion. CPU-only
  scorer retry `181582_1` completed after the explicit-null protocol fix; no GPU
  artifact was regenerated.

## Verification

- All final transfer jobs completed with exit code `0:0`.
- `python -m unittest tests.test_score_temporal_predictions`: 19 tests passed
  before the crop scorer retry.
- `python -m unittest discover -s tests`: 351 tests passed.
- `git diff --check`: passed.
