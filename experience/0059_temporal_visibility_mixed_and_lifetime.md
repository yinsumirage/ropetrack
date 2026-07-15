# Mixed visibility coverage, framewise ablation, and trusted-state lifetime

Date: 2026-07-16

## Questions

1. Can one minimal visibility gate trained on mixed global corruptions cover the
   frozen-gate failures from `0058` without losing the domains that already
   passed?
2. Is dense K1 necessary after trusted-state selection, or does the original
   framewise flex15 MLP recover the same result?
3. For how long does a 30-frame clean prefix remain useful when masking lasts
   120 or 240 frames and there is no artificial recovery phase?

## Fixed protocol

- Dataset and order: HO3D v3 evaluation, the existing 20,137-row dense order.
- Pose backend and refiner: frozen WiLoR export, frozen dense K1 flex15
  checkpoint, and the learned causal state application from `0057`.
- Original safe gate: frozen 1280D mean-pooled WiLoR image feature followed by
  `Linear(1280, 1)`, with validation zero-false-clean threshold `0.20424594`.
- Mixed gate: the same linear architecture, trained on the existing
  deterministic `effect=mixed`, severity `0.7` train root. Only masked rows are
  randomly assigned mask, blur, or crop; context and recovery remain clean.
  The mixed gate's validation zero-false-clean threshold is `0.51930135`.
- Neither gate reads phase, rope, pose, or future frames. Phase is used only to
  build controlled corruptions, audit gate errors, and score slices.
- Paired confidence intervals use 2,000 whole-sequence bootstrap draws.

Run roots:

- mixed gate and blur transfer:
  `/data/wentao/ropetrack/runs/temporal_visibility_mixed_20260716`;
- framewise state ablation:
  `/data/wentao/ropetrack/runs/temporal_state_framewise_ablation_20260716`;
- 120/240-frame lifetime:
  `/data/wentao/ropetrack/runs/temporal_state_lifetime_20260716`.

## Mixed-gate seven-domain screen

False-clean means that a corrupted frame would be accepted as a trusted-state
update. Deltas below are new mixed-gate minus the old mask70-trained gate in
percentage points. The conservative promotion bar remains false-clean at most
0.1% and zero three-frame false-clean runs.

| Domain | Old default | Mixed default | Delta | Old zero-FN | Mixed zero-FN | Delta | Mixed run3 / max run | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mask40 | 60.4161% | 74.8572% | +14.4411 pp | 39.7626% | 75.8951% | +36.1325 pp | 13 / 666 | stop |
| mask70 | 0.0050% | 0.0894% | +0.0844 pp | 0 | 0.1043% | +0.1043 pp | 1 / 8 | regress; do not replace old gate |
| mask90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 / 0 | pass |
| finger_end80 | 0.3476% | 1.7381% | +1.3905 pp | 0 | 1.9367% | +1.9367 pp | 6 / 40 | regress; stop |
| tip_square80 | 58.0325% | 58.4198% | +0.3873 pp | 42.2804% | 59.3038% | +17.0234 pp | 13 / 650 | stop |
| blur70 | 45.7715% | **0** | -45.7715 pp | 34.8811% | **0** | -34.8811 pp | 0 / 0 | new screen pass |
| crop70 | 0.0497% | **0** | -0.0497 pp | 0.0050% | **0** | -0.0050 pp | 0 / 0 | pass |

The mixed gate learns blur coverage but is not a universal replacement: it
damages mask70 and finger-end safety and makes light mask and tip-square worse.
Only the newly passing blur70 domain received a full pose run.

## Blur70 full temporal pose result

The episode remains 30 clean, 60 blur70, 30 clean recovery.

| Method | Overall | Context | Masked | Recovery | Masked tips | Masked velocity | Masked acceleration | Masked jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 7.6743 | 7.2066 | 8.2156 | 7.1424 | 12.3729 | 88.85 | 4184 | 4026 |
| K1 | 7.2752 | 6.8695 | 7.6444 | 6.7900 | 10.7355 | 94.28 | 4502 | 4381 |
| mixed gate state + K1 | **7.0535** | **6.8052** | **7.2006** | **6.7701** | **10.1511** | **86.78** | **4097** | **3960** |
| state + shuffled rope | 8.0819 | 6.9097 | 9.3619 | 6.8548 | 14.9723 | 226.34 | 11840 | 11746 |

State minus K1 is -0.4438 mm masked PA, but its paired CI
`[-0.9620, +0.0109]` narrowly includes zero. Masked tips improve by -0.5844 mm,
CI `[-1.4876, +0.1804]`. Masked velocity, acceleration, and jitter improve by
-7.50 mm/s, -405.64 mm/s2, and -421.22 mm/s2. The accuracy direction is good,
but the preregistered primary CI gate is not passed.

| Horizon | K1 | State | State - K1 | Paired CI |
|---:|---:|---:|---:|---:|
| H1 | 7.7652 | 6.7748 | -0.9904 | `[-1.4369, -0.5895]` |
| H5 | 7.7801 | 6.8778 | -0.9023 | `[-1.3486, -0.5093]` |
| H15 | 7.8587 | 7.0954 | -0.7633 | `[-1.3073, -0.2644]` |
| H30 | 7.4940 | 7.1215 | -0.3725 | `[-0.9684, +0.1834]` |
| H60 | 7.7790 | 7.5812 | -0.1978 | `[-0.6239, +0.2029]` |

Shuffling rope worsens the correct-rope state by +2.1613 mm masked PA, CI
`[+1.9007, +2.3587]`, and +4.8213 mm on masked tips. The early-horizon gain is
not direct pose copying, but blur70 is not formally promoted and receives no
seed expansion.

## Original framewise flex15 inside trusted state

This ablation keeps the frozen mask70 image-linear zero-FN gate and state
logic unchanged. It replaces dense K1 only with the exact nested framewise
flex15 MLP by zeroing the K1 residual head; the nested weights are asserted
equal to the original framewise checkpoint before inference.

| Method | Overall | Context | Masked | Recovery | Masked tips | Masked velocity | Masked acceleration | Masked jitter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 8.4531 | 7.2066 | 9.9240 | 7.1424 | 15.9459 | 127.35 | 6155 | 6038 |
| framewise only | 7.9527 | 6.9366 | 9.1389 | 6.8414 | 13.4100 | 126.05 | 6105 | 6008 |
| K1 only | 7.7848 | 6.8695 | 8.7622 | 6.7900 | 12.5808 | 130.76 | 6367 | 6280 |
| K1 trusted state | **7.0546** | **6.7683** | **7.1253** | **6.7974** | **9.9741** | 120.45 | 5829 | 5723 |
| framewise trusted state | 7.1201 | 6.8377 | 7.2810 | 6.8374 | 10.5321 | **118.63** | **5733** | **5617** |
| framewise state + shuffled rope | 7.7335 | 6.9122 | 8.5163 | 6.9633 | 13.5818 | 196.64 | 10026 | 9926 |

Framewise state minus K1 state is +0.1557 mm masked PA, CI
`[-0.0274, +0.3718]`; most of the large state gain therefore comes from
trusted-state management, not K1. K1 still has a real localized advantage:
framewise state worsens masked tips by +0.5580 mm, CI
`[+0.2371, +0.8970]`, and is significantly worse at H15 (+0.2384 mm) and H60
(+0.3182 mm). Its small dynamics advantage does not compensate for the tip
regression. Keep K1; do not claim that framewise is equivalent.

Framewise rope shuffle worsens its correct-rope state by +1.2353 mm masked PA,
CI `[+0.9486, +1.4703]`, and +3.0496 mm on masked tips.

## No-recovery trusted-state lifetime

Commit `a82d7796e28f05fbbca2d581f5818ed85ffb0259` minimally allows zero recovery,
infers the common phase layout from the manifest, and adds H120/H240 scoring.
Both runs use a 30-frame clean prefix followed by mask70 until the end of each
complete episode. The original mask70 gate and K1 remain frozen.

Gate health remains safe during masking:

| Mask length | Complete episodes | Masked rows | Masked false-clean | Max false-clean run | Context false-freeze |
|---:|---:|---:|---:|---:|---:|
| 120 | 120 | 14,400 | 0 | 0 | 8.25% |
| 240 | 58 | 13,920 | 0 | 0 | 6.38% |

Recovery metrics are correctly `null`; no recovery value is fabricated.

### Absolute aggregate scores

| Mask | Method | Overall | Context | Masked | Masked tips | Masked velocity | Masked acceleration | Masked jitter |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 120 | base | 9.2100 | 7.2149 | 9.9324 | 16.0327 | 126.04 | 6050 | 5928 |
| 120 | K1 | 8.3389 | 6.8988 | 8.7688 | 12.6012 | 129.93 | 6289 | 6201 |
| 120 | trusted state + K1 | **7.7095** | **6.7835** | **7.8701** | **11.3412** | **119.09** | **5705** | **5599** |
| 120 | state + shuffled rope | 9.1985 | 6.9869 | 9.8694 | 16.1298 | 251.38 | 13055 | 12977 |
| 240 | base | 9.0570 | 7.1449 | 9.9058 | 16.0841 | 124.69 | 6008 | 5897 |
| 240 | K1 | 8.2435 | 6.8992 | 8.7981 | 12.7120 | 128.98 | 6266 | 6186 |
| 240 | trusted state + K1 | **8.0354** | **6.8800** | **8.4554** | **12.3259** | **119.10** | **5732** | **5635** |
| 240 | state + shuffled rope | 9.4156 | 6.9922 | 10.4096 | 17.0332 | 251.56 | 13083 | 13008 |

For mask120, state minus K1 is -0.8987 mm masked PA, CI
`[-1.4032, -0.4065]`, and -1.2600 mm on masked tips, CI
`[-2.2697, -0.3513]`. Velocity, acceleration, and jitter improve by
-10.84 mm/s, -584.02 mm/s2, and -601.35 mm/s2.

For mask240, aggregate state minus K1 is -0.3427 mm masked PA, but CI
`[-1.0971, +0.5714]` crosses zero. Masked tips are -0.3862 mm, CI
`[-1.6376, +0.9339]`. Dynamics still improve by -9.88 mm/s, -533.85 mm/s2,
and -550.64 mm/s2.

### Horizon curve and crossover

| Run | Horizon | K1 | Trusted state | State - K1 | Paired CI |
|---:|---:|---:|---:|---:|---:|
| 120 | H1 | 8.5219 | 6.4955 | -2.0264 | `[-2.4316, -1.6292]` |
| 120 | H5 | 8.5896 | 6.5402 | -2.0494 | `[-2.4013, -1.7032]` |
| 120 | H15 | 8.7374 | 6.6876 | -2.0498 | `[-2.3969, -1.6788]` |
| 120 | H30 | 8.7450 | 7.1230 | -1.6221 | `[-2.0679, -1.1151]` |
| 120 | H60 | 8.7757 | 7.9835 | -0.7923 | `[-1.4025, -0.2354]` |
| 120 | H120 | 9.2691 | 9.0674 | -0.2017 | `[-0.6586, +0.2185]` |
| 240 | H1 | 9.0558 | 6.6929 | -2.3629 | `[-2.7257, -2.0430]` |
| 240 | H5 | 9.1982 | 6.6538 | -2.5444 | `[-2.8379, -2.2272]` |
| 240 | H15 | 8.9188 | 6.6264 | -2.2924 | `[-2.7942, -1.8215]` |
| 240 | H30 | 8.3578 | 6.7854 | -1.5724 | `[-1.9833, -1.2109]` |
| 240 | H60 | 8.6643 | 7.7682 | -0.8961 | `[-1.7924, +0.0889]` |
| 240 | H120 | 8.6166 | 8.7843 | +0.1677 | `[-0.9552, +1.4755]` |
| 240 | H240 | 8.9370 | 10.5010 | **+1.5640** | `[+0.8072, +2.4404]` |

The clean prefix has a robust useful lifetime through H60, approximately two
seconds at 30 FPS. H120 is the crossover/uncertainty region. Indefinite state
hold is rejected: by H240 the trusted state is significantly worse than K1.
Correct current rope remains necessary throughout. Relative to correct-rope
state, shuffle worsens masked PA by +1.9993 mm for mask120, CI
`[+1.7318, +2.2475]`, and +1.9542 mm for mask240, CI
`[+1.5785, +2.2817]`.

## Decision

- **Validated:** visibility-aware trusted-state management is the main temporal
  mechanism. It provides large accuracy and dynamics gains through roughly 60
  masked frames, and current rope supplies indispensable flexion updates.
- **Retain:** dense K1 adds a modest but real fingertip and long-horizon benefit
  over the old framewise MLP. Keep it as the rope-conditioned update.
- **Bounded:** a trusted pose is not permanent. A deployable tracker needs
  explicit state age and must expire or fall back near the 120-frame region;
  freezing through 240 frames is wrong.
- **Stop:** one mixed global-corruption linear gate is not a universal gate;
  blur70 pose does not pass the primary paired-CI promotion rule; do not expand
  seeds, model capacity, K16/K96 history, GRUs, Transformers, or velocity grids.
- **Next only when real data is available:** validate the same gate/state-age
  behavior on a genuinely temporal second dataset or physical video. HO3D v2
  must first pass a continuity audit; FreiHAND remains screen-only.

## Jobs and verification

- mixed gate: `182154 -> 182155 -> 182156 -> 182157_[0-6]`;
- blur70 pose: `182301 -> 182302/182303 -> 182304 -> 182305 -> 182306`;
- framewise ablation: `182307 -> 182308`;
- lifetime: `182335_[0-1] -> 182336_[0-1]/182337_[0-1] -> 182338_[0-1]`,
  `182336_[0-1] -> 182339_[0-1]`, then `182340_[0-1] -> 182341_[0-1]`.
- Every listed final job completed with exit code `0:0`.
- `python -m unittest tests.test_temporal_refiner tests.test_make_hard_images tests.test_score_temporal_predictions`: 102 tests passed.
- `python -m unittest discover -s tests`: 353 tests passed.
- `git diff --check`: passed before the lifetime implementation commit and is
  rerun after this note.
