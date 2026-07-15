# Clean-prefix temporal state oracles

Date: 2026-07-15

## Question

Does a trustworthy clean visual state remain useful during a later 60-frame
occlusion, and is the useful mechanism pose hold, motion extrapolation, fixed
MANO shape, or current rope correction?

## Protocol

- Oracle implementation: `4c84c6bbf21e10dc25dec6ff47e41888e840ccd5`
- Perfect-prefix analysis: `971d252359697bf9d24751282c4be16eac0074da`
- Decoded visual-prefix control: `3e3419b758a6ea2762766e6be42ae2f32a133e87`
- WiLoR: `fcb911312a38fa8badd30d9656a167485d61b8f9`
- Run root: `/data/wentao/ropetrack/runs/temporal_oracle_state_20260715`
- Evaluation: 153 complete HO3D v3 episodes, each with 30 clean context,
  60 mask70, and 30 clean recovery frames. Known phase is used only as an
  oracle gate and for scoring.
- All methods reuse the dense WiLoR/MANO cache. No WiLoR inference or tracker
  training was repeated.
- Scores use 2,000 paired sequence bootstrap samples. Lower is better.

Jobs `180460_[0-1]` produced the train/eval perfect-prefix controls. Job
`180426` decoded the MANO state oracles. CPU jobs `180969` and `180970`
produced the visual-prefix control and full phase-aware score. Initial CPU jobs
`180966` and `180427` exited before computation because of stale CLI argument
names and an over-strict submodule cleanliness check; their corrected retries
completed without changing prediction payloads.

## Clean-prefix ceilings

| Prefix state | Train masked PA | Eval masked PA |
|---|---:|---:|
| perfect GT last-clean | 4.9761 mm | 4.2889 mm |
| perfect GT constant velocity | 5.4941 mm | 4.4052 mm |
| perfect GT damped velocity | 5.4587 mm | 4.3831 mm |
| decoded WiLoR last-clean | n/a | 7.7349 mm |
| decoded WiLoR constant velocity | n/a | 8.0311 mm |
| decoded WiLoR damped velocity | n/a | 8.0129 mm |
| dense K1 | n/a | 8.7622 mm |

The perfect-state ceiling is large and persists through frame 60: eval
last-clean is 6.7108 mm at frame 60 versus K1 at 8.7446 mm. More importantly,
the deployable-quality decoded clean visual state also beats K1 by 1.0274 mm;
its paired interval is `[-1.6188, -0.4428]` mm. An ideal per-frame selector
between decoded last-clean and K1 reaches 7.1249 mm, leaving additional
visibility/quality-gating headroom.

Both train and eval reject generic motion extrapolation: zero-velocity hold is
better than constant or damped velocity. Do not add a velocity damping grid.

## Cached MANO oracle result

| Method | Overall | Context | Masked | Recovery | Masked tips |
|---|---:|---:|---:|---:|---:|
| base WiLoR | 8.4531 | 7.2066 | 9.9240 | 7.1424 | 15.9459 |
| framewise | 7.9527 | 6.9366 | 9.1389 | 6.8414 | 13.4101 |
| dense K1 | 7.7848 | 6.8695 | 8.7622 | 6.7900 | 12.5808 |
| last-clean only | 7.3205 | 6.8695 | 7.7437 | 6.7900 | 11.9460 |
| last-clean + current rope/K1 | **7.0728** | 6.8695 | **7.2003** | 6.7900 | **10.1113** |
| constant velocity + rope/K1 | 7.1504 | 6.8695 | 7.3707 | 6.7900 | 10.3227 |
| damped velocity + rope/K1 | 7.1446 | 6.8695 | 7.3578 | 6.7900 | 10.2995 |
| fixed beta + K1 | 7.7912 | 6.8695 | 8.7761 | 6.7900 | 12.5656 |
| fixed beta + last-clean + K1 | 7.0832 | 6.8695 | 7.2231 | 6.7900 | 10.1248 |

The winning deterministic state rule improves masked PA by 1.5619 mm
(`17.83%`) relative to K1, with paired CI `[-1.9912, -1.1824]` mm. Masked
occluded tips improve by 2.4696 mm (`19.63%`), with CI
`[-3.3413, -1.6256]` mm.

| Masked horizon | K1 | Last-clean + rope/K1 | Gain |
|---:|---:|---:|---:|
| 1 | 8.8063 | 6.8474 | 1.9589 mm |
| 5 | 8.8729 | 6.9591 | 1.9138 mm |
| 15 | 9.0723 | 7.0990 | 1.9733 mm |
| 30 | 8.7010 | 7.1119 | 1.5891 mm |
| 60 | 8.7446 | 7.6127 | 1.1320 mm |

The state benefit therefore survives the full two-second mask, although it
decays after frame 30.

## Rope, shape, and motion controls

- Correct current rope labels improve last-clean + K1 from the shuffled-rope
  control at 7.5776 to 7.2003 mm. The 0.3772 mm gap grows to 1.1577 mm on
  occluded tips, so the rope contribution is localized and label-dependent.
- Fixed prefix beta alone is neutral/slightly harmful: 8.7761 versus 8.7622 mm,
  with CI `[-0.0082, 0.0349]` mm. It is not an accuracy mechanism.
- Fixed beta with last-clean improves masked acceleration error and jitter by
  10.15% and 10.63% versus K1, but is 0.0228 mm worse in masked PA than the
  same state rule without fixed beta. Treat it only as an optional smoothness
  trade-off.
- Last-clean + rope/K1 improves masked acceleration error and jitter by 8.25%
  and 8.65%. This is real motion improvement, not a frozen-pose-only score win,
  but it is just below the preferred 10% target.
- Recovery PA and recovery lag are unchanged from K1 because oracle visibility
  resumes current visual updates immediately at the known recovery boundary.

## Decision

The clean-prefix state hypothesis passes. History is useful when represented
as an explicit last trusted visual state; the failed K16/K96 result was a state
management failure, not evidence that temporal information has no value.

Proceed only to a minimal learned visibility/quality gate around this state
rule: update the trusted pose on reliable visual frames, hold it during low
visibility, and apply the current rope-conditioned finger update. Do not add a
larger generic temporal network, a velocity family, or learned sequence shape
yet. The first learned test must recover a meaningful fraction of the
7.2003 mm oracle result without using phase labels at inference.
