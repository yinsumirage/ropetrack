# Existing-prediction per-finger gate and InterHand input-only follow-up

Date: 2026-07-22

## Decision

- **Continue the current frozen-token `DirectPoseHead`; do not rewrite the
  architecture yet.** Existing predictions contain repeatable finger-level
  heterogeneity, but the extra offline headroom from finger-wise rather than
  frame-wise selection is only about `0.14-0.23 mm` on the main comparisons.
- **Do not form a five-dataset joint-training mixture.** The current combined
  model is only ARCTIC+HOT3D+HO3D, and 0079 already rejected promotion or
  retuning of that fixed triple. DexYCB and InterHand remain separate
  diagnostic domains.
- **Keep InterHand as a two-hand/occlusion stress test, not a default training
  source.** Single and interacting samples show the same direction of average
  input-only rope gain; interaction is harder but is not the sole failure
  mechanism.
- **Do not implement a visibility classifier or reliability-aware fusion from
  this result alone.** The oracle is optimistic and uses separately PA-aligned
  method errors. Revisit only after the AVP/external-camera pilot or another
  physical-sensor boundary shows the same finger-specific pattern.

## Existing five-domain diagnostic

This CPU-only analysis reuses the verified 0084 `per_sample.npz`; it does not
rerun inference, select checkpoints, or read DexYCB/InterHand test. Negative
values improve.

| Evaluation comparison | Mean PA delta (mm) | P95 ref -> candidate (mm) | Improved samples | Extra finger-wise vs frame-wise oracle (mm) |
|---|---:|---:|---:|---:|
| ARCTIC triple mean - WiLoR | -0.694 | 12.037 -> 9.980 | 65.8% | -0.171 |
| HOT3D triple stitched - WiLoR | -3.233 | 16.820 -> 10.039 | 89.3% | -0.162 |
| HO3D v3 triple mean - WiLoR | -0.598 | 11.059 -> 10.451 | 69.9% | -0.137 |
| DexYCB S1-val RGB+rope - RGB-only | -1.305 | 8.551 -> 6.269 | 95.1% | -0.065 |
| InterHand val old RGB+rope - RGB-only | +0.283 | 13.912 -> 13.014 | 33.7% | -0.218 |

The first three rows measure the whole DirectPose effect and do not isolate
rope, because those frozen artifacts have no matched RGB-only/shuffled
prediction. DexYCB shows broad five-finger benefit, so a complex gate has
little room there. The old InterHand rope objective improves the severe tail
while harming most samples, which motivated the matched input-only follow-up.

## InterHand input-only official-val follow-up

The already-trained `rope_weight=0` checkpoints from 0083 were applied to the
same 19,341-sample official-val boundary. No checkpoint was retrained or
selected from this result, and test was not read.

| Method | Standard PA (mm) | Root-relative (mm) | Camera (mm) |
|---|---:|---:|---:|
| WiLoR | 8.811 | 19.062 | 81.994 |
| RGB-only | 8.428 | 18.703 | 82.715 |
| correct-trained + correct rope | **8.296** | **18.083** | 83.648 |
| correct-trained + shuffled rope | 13.376 | 23.018 | 85.911 |
| shuffled-trained + correct rope | 8.385 | 18.427 | 82.968 |

Correct-trained/correct-rope improves PA over RGB-only by `-0.132 mm`, paired
frame-group bootstrap 95% CI `[-0.157,-0.107]`, and root-relative error by
`-0.619 mm`. Camera error worsens by `+0.933 mm`, which stays outside the local
articulation product scope. Its PA P95 improves from `13.912` to `12.883 mm`,
and the mean change on the worst 10% of RGB-only samples is `-3.002 mm`, even
though only 49.4% of all samples improve.

The effect is not uniform by finger:

| Finger | Correct rope - RGB-only PA joint error (mm) | 95% CI |
|---|---:|---:|
| thumb | +0.131 | [+0.069,+0.189] |
| index | -0.340 | [-0.378,-0.303] |
| middle | -0.228 | [-0.264,-0.191] |
| ring | +0.142 | [+0.107,+0.177] |
| pinky | -0.197 | [-0.226,-0.166] |

The overall input-only PA delta is `-0.171 mm` on single-hand rows and
`-0.091 mm` on interacting rows; both CIs exclude zero. On interacting rows,
thumb worsens by `+0.445 mm` while index improves by `-0.535 mm`. Thus the
two-hand subset strengthens the finger-specific trade-off but does not reverse
the overall direction.

Correct input beats inference-time shuffled input by `-5.081 mm` PA, CI
`[-5.146,-5.015]`, proving that this checkpoint uses the paired rope rather
than ignoring it. Correct-trained/correct-input also beats the shuffled-trained
control under correct input by only `-0.089 mm`, CI `[-0.114,-0.066]`; this is
the narrower sensor-specific training effect. The catastrophic arbitrary
shuffle result is a dependency/safety warning, not a calibrated physical-noise
claim.

The finger-wise selection oracle is `0.226 mm` better than a frame-wise oracle
overall (`0.283 mm` on interacting rows, `0.172 mm` on single rows). This is a
real diagnostic signal but still too small and optimistic to pay for a large
fusion rewrite before physical evidence.

## Artifacts and verification

- Run root:
  `/data/wentao/ropetrack/runs/direct_pose_per_finger_20260722`.
- Five-domain existing-prediction report:
  `report.{json,md}` at the run root.
- InterHand predictions:
  `interhand_apply/{correct_train_correct_input,correct_train_shuffled_input,shuffled_train_correct_input}`.
- InterHand decomposition and finger report:
  `interhand_decomposition/` and `interhand_finger/`.
- Jobs: five-domain CPU `191649`; apply array `191650-191652`; first CPU
  attempt `191653` failed before analysis because the pinned code snapshot did
  not contain the 0084 script; corrected CPU `191655`; verifier `191656`.
- Verifier: **PASS** for parity, sample IDs, bootstrap units, aggregate
  reconstruction, immutable inputs, required outputs, and no InterHand test
  read.

The analysis launchers stay under ignored `.local_checks/`; no model or
production code was changed.
