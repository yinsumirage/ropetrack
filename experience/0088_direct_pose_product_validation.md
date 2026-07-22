# DirectPose product perturbation and conflict-attribution audit

Date: 2026-07-23

## Decision

- **Continue DirectPose as a conditional experimental product path, not as the
  formal release or an always-on replacement for WiLoR.** Existing h128 heads
  give a large, causal local-pose gain on HOT3D and a smaller gain on ARCTIC.
  The benefit is larger on the fixed HOT3D low-visibility slice.
- **Stop the proposed equal/near-equal four-core joint update and do not unlock
  the WiLoR local decoder.** The training-only audit in 0087 failed that gate.
  This follow-up attributes much of the important HOT3D--DexYCB conflict to the
  rope-conditioning query and auxiliary rope/delta objectives rather than to a
  universal disagreement in the PA/root task direction.
- **Exact missing/invalid fallback passes.** An explicitly invalid finger, a
  non-finite channel, or a normalized value outside `[0, 1]` returns exactly
  the original WiLoR local joints for that finger. All-invalid returns the full
  original WiLoR local pose; clean checkpoints remain bitwise unchanged.
- **The predeclared product matrix stops new training for now.** Every frozen
  HOT3D gate passes. ARCTIC still beats WiLoR under simulated Gaussian
  `sigma=0.05`, but retains only 34.4% of its clean gain, below the frozen 60%
  gate. A separately frozen ARCTIC noise-dose diagnostic places the simulated
  passing ceiling at `sigma=0.03`; `0.04` fails. This bounds the sensor target
  but does not reopen robust-head training under the failed `0.05` protocol.
- **PALM is not a present blocker or the next experiment.** It may later supply
  a subject-disjoint MANO geometry prior, but it does not validate physical
  rope readings, does not solve missing/drifting sensors, and must not be added
  as an equal fifth domain. Its predominantly non-commercial license also
  prevents treating it as an unquestioned product-training asset.

The formal release remains RopeAlphaStudent. All rope values below are
GT-derived ideal normalized geometry plus simulated corruption, not physical
glove evidence.

## What the failed multi-domain gate means

The 0087 result did **not** show that one DirectPose head can never serve more
than one dataset. It showed that a fresh equal/near-equal update of the current
shared h128 head is not non-regressive: most off-domain one-step PA deltas were
significantly positive. Continuing that recipe can move the same weights in a
direction that improves one domain while degrading another, and adapting the
larger local decoder would expose far more of the WiLoR prior to that risk.

This matters for product use because an always-on shared model must preserve
normal RGB behavior across people, cameras, and interaction distributions. A
mean mixture score can hide a domain or finger regression. It does not mean the
already-trained head loses its existing HOT3D/ARCTIC gains, nor that an exact
fallback wrapper is impossible.

## Conflict attribution

Run root:
`/data/wentao/ropetrack/runs/direct_pose_conflict_attribution_20260723/full`.
The machine-readable protocol was frozen before results (SHA-256
`2ef570dda7d99ac0e7c79d352cef7c58484299408a62ece0c6808d9a68cb4839`).
It reused the fixed training-only update/probe rows and checkpoint from 0087;
no eval/test GT entered gradients or selection.

### Full loss versus task-only direction

Selected mean gradient cosines over deterministic batches:

| Pair | Full correct rope | Task-only correct rope | Full shuffled rope | Interpretation |
|---|---:|---:|---:|---|
| ARCTIC--HOT3D | -0.071 | uncertain around zero | broadly positive | Different paired-rope operating points contribute; matched-rope task direction remains negative in the matched diagnostic. |
| HOT3D--DexYCB | **-0.148** | **+0.204** | broadly positive | PA/root task gradients are compatible; rope/delta auxiliaries reverse the full direction. |
| HOT3D--InterHand | **-0.195** | mixed | broadly positive | InterHand is a stress domain, not a safe HOT3D training source. |
| DexYCB--InterHand | **+0.352** | positive | positive | InterHand is not globally hostile to all core domains. |

For HOT3D--DexYCB, the correct-rope cosine by parameter group is
`-0.509` in `condition_query`, `-0.057` in RGB attention, and `-0.091` in the
residual path. This localizes the strongest conflict to how the shared head
interprets rope-conditioned corrections. The fixed input-minus-base rope means
also differ: HOT3D is positive on every finger (about `+0.080` to `+0.094`),
ARCTIC is negative (about `-0.012` to `-0.041`), and DexYCB is near zero/mixed.

The finger pattern is not a thumb-only issue. HOT3D--DexYCB is strongest on
ring (`-0.404`) and pinky (`-0.413`); ARCTIC--HOT3D is strongest on middle
(`-0.260`), with thumb also negative (`-0.140`). A single scalar dataset weight
therefore cannot be assumed to fix the conflict.

The safe software implication is narrower than PCGrad or dataset-conditioned
heads: preserve the current frozen WiLoR prior and existing h128 head, remove
no auxiliary objective based on final scores, and require a training-only
non-regression proposal before any new shared update.

## Structural product fallback

The outer project wrapper masks each finger's 9D local MANO residual after the
historical head. `third_party/` is unchanged. Local and real-checkpoint checks
cover:

| Condition | Maximum absolute local-pose difference | Result |
|---|---:|---:|
| clean new path versus legacy checkpoint | 0 | PASS |
| one explicit invalid finger versus WiLoR for that finger | 0 | PASS |
| other valid fingers versus clean DirectPose | 0 | PASS |
| all invalid versus full WiLoR local pose | 0 | PASS |
| NaN, +Inf, below 0, or above 1 channel versus WiLoR finger | 0 | PASS |

The real-checkpoint gate used 64 HOT3D rows and is recorded in
`auto_invalid_gate.json`. This detects explicit status and obvious numerical or
range faults. It cannot reliably detect a plausible in-range biased, stale, or
drifting sensor from one frame. Product hardware should therefore supply
per-channel validity/CRC, stale timestamp, saturation and rate checks; the
model should consume that validity mask. A simple RGB--rope disagreement gate
would be unsafe because useful rope is expected to disagree with the visual
estimate precisely during occlusion.

## Frozen product perturbation matrix

Run root:
`/data/wentao/ropetrack/runs/direct_pose_product_validation_20260723`.
The protocol was frozen before scores (SHA-256
`3c0c2bab4867c2ed792bea2667d57a1e6d56391711f1c048c7845368ec4fd5d4`).
It evaluates existing fold-matched h128 heads only: 21 conditions, three fixed
folds/seeds, episode bootstrap with 2,000 replicates, no checkpoint selection.
HOT3D is reused participant-disjoint OOF validation, not an untouched test;
ARCTIC is the frozen external anchor.

### HOT3D

| Condition | PA mean / P95 mm | Delta versus WiLoR [95% CI] | Clean gain retained | Improved rows |
|---|---:|---:|---:|---:|
| WiLoR / all missing | 8.984 / 16.820 | 0 | 0% | 0% |
| clean paired rope | **5.732 / 10.017** | **-3.252 [-3.580,-2.918]** | 100% | 90.0% |
| shuffled rope | 12.357 / 21.573 | +3.373 | -103.7% | 26.5% |
| Gaussian noise 0.05 | 6.485 / 10.619 | -2.499 | 76.8% | 82.3% |
| dropout 0.10 | 6.278 / 10.910 | -2.706 | 83.2% | 88.4% |
| noise 0.05 + dropout 0.10 | 6.910 / 11.335 | -2.074 | 63.8% | 79.8% |

All fixed bias/gain, single-finger missing, dropout and combined conditions
remain better than WiLoR. Single missing middle/ring/pinky retain about
59--60% of the clean gain; thumb/index retain 77%/71%.

The occlusion result is the strongest product signal:

| HOT3D fixed phase | WiLoR PA | Clean DirectPose PA | Gain | Noise 0.05 retained | Dropout 0.10 retained |
|---|---:|---:|---:|---:|---:|
| context | 7.643 | 5.297 | 2.346 | 68.9% | 82.9% |
| low visibility | 10.271 | 6.148 | **4.123** | 81.2% | 83.4% |

This supports the intended role: rope helps most when vision is weak. It is
still ideal-sensor evidence, not proof from physical rope hardware. Clean
motion error is essentially unchanged from WiLoR (`12.635` versus `12.629`),
while simulated noise/dropout worsen it, so temporal stability is not yet a
claimed improvement.

### ARCTIC

| Condition | PA mean / P95 mm | Delta versus WiLoR [95% CI] | Clean gain retained | Improved rows |
|---|---:|---:|---:|---:|
| WiLoR / all missing | 6.695 / 11.986 | 0 | 0% | 0% |
| clean paired rope | **5.979 / 9.901** | **-0.716 [-0.850,-0.587]** | 100% | 66.8% |
| shuffled rope | 10.908 / 18.967 | +4.213 | -588.1% | 8.8% |
| Gaussian noise 0.05 | 6.449 / 10.262 | **-0.246 [-0.390,-0.110]** | **34.4%** | 50.6% |
| dropout 0.10 | 6.142 / 10.188 | -0.553 | 77.3% | 63.5% |
| noise 0.05 + dropout 0.10 | 6.550 / 10.482 | -0.145 | 20.2% | 47.7% |

The ARCTIC `sigma=0.05` failure is relative robustness, not a catastrophic
regression: its mean still significantly beats WiLoR. However, the clean gain
is smaller and most of it is lost, while motion error worsens from `1.171` to
`1.187`. This is enough to reject the predeclared claim that the current model
retains most clean gain at that noise level.

### ARCTIC noise-dose ceiling

The follow-up protocol was independently written before its results (SHA-256
`ac8d6ea73a4fd0526adb675242a5db2b5a0a7193da408a1b5ec5805ed6ae39b0`).
It freezes five additional noise levels, three seeds/folds, the same
episode-bootstrap procedure and a diagnostic-only gate: retain at least 60%
of clean gain and remain no worse than WiLoR.

| Gaussian noise std | PA mean / P95 mm | Delta versus WiLoR [95% CI] | Clean gain retained | Improved rows | Motion error |
|---:|---:|---:|---:|---:|---:|
| 0.010 | 6.002 / 9.926 | -0.693 [-0.828,-0.564] | 96.7% | 66.0% | 1.125 |
| 0.020 | 6.068 / 9.960 | -0.627 [-0.763,-0.497] | 87.5% | 63.4% | 1.135 |
| 0.025 | 6.115 / 9.990 | -0.580 [-0.718,-0.449] | 81.0% | 62.1% | 1.141 |
| **0.030** | **6.170 / 10.015** | **-0.525 [-0.666,-0.394]** | **73.3%** | **60.3%** | **1.149** |
| 0.040 | 6.298 / 10.152 | -0.397 [-0.538,-0.262] | **55.4%** | 55.9% | 1.167 |

All 45 expected outputs and the raw-artifact hash verify. The largest tested
passing standard deviation is `0.03` in normalized rope units. This is a useful
simulation acceptance target for a hardware pilot, not a calibrated physical
error bar: correlated drift, hysteresis, slack and latency still require real
logs. The original `sigma=0.05` product gate remains failed, so this diagnostic
does not authorize a post-hoc robust-training recipe.

## PALM boundary

The official [PALM repository](https://github.com/facebookresearch/PALM) and
[paper](https://arxiv.org/abs/2511.05403) describe multi-view RGB, scans and
MANO registrations that may be valuable for a broad hand-geometry prior. Its
released training recipe is a large multi-view reconstruction setup, not a
drop-in pose-likelihood module for DirectPose. DirectPose currently freezes
betas and uses trusted 3D joint supervision for a bounded local residual, so
downloading PALM now would expand scope without answering the failed
sensor-noise or multi-domain-update gates.

If revisited, the minimum useful PALM experiment is separate, subject-disjoint
prior pretraining or a frozen plausibility regularizer, followed by the same
gradient/retention audit. Do not mix it equally into HOT3D/ARCTIC training or
describe its synthetic/registered geometry as rope evidence. License review is
required before any commercial product use.

## Jobs, verification and boundaries

- conflict attribution jobs: CPU protocol `192843`, 1-GPU smoke `192844`,
  formal 1-GPU audit `192847`; all completed;
- product apply jobs: `192874`, `192878_1`--`192878_5`; all completed with at
  most two GPUs active;
- product scorer: first `192879` failed before reading results because the
  run-local launcher lacked the project import path; corrected launcher
  `192920` completed and verifier passed;
- real-checkpoint invalid-value gate: first `192894` hit the same pre-model
  import issue; corrected `192896` completed and passed;
- product verifier reconstructs conditions and metrics from immutable inputs,
  checks the frozen protocol hash, and validates exact fallback: PASS;
- ARCTIC noise-dose jobs: 1-GPU smoke `192937`, two-fold GPU array `192940`,
  CPU scorer/verifier `192941`; all completed, 45/45 outputs verified;
- successful GPU jobs used an aggregate `00:59:53`, about `0.998` single-GPU
  hours; protocol through final noise verifier spanned about `01:10:47` wall
  time, with no more than two GPUs concurrently active;
- attribution raw/summary/verifier SHA-256:
  `773f1e6b0f4e27152bc4c2287fc9f05ecc9cf43b212d6bf27be4a22742f1a17f`,
  `931823878ae402292d8c24b6a539863e9ecd0364720606eea89a647b978be0e1`,
  `43e76eb3853e8e0bbc9ce35719aa8162716fbabce8625e1404d8acaee1159ffa`;
- product raw/summary/verifier SHA-256:
  `3db78856adf1d506fd86ebf7d25cabcc1ba8209c04c23df691d4db825aaa4f0f`,
  `e476cf27bd1b831a00bcfb911c1445efaf522b698b64f7a31e4635e59123967a`,
  `3dd693c9fbb6cf5a8917076f752eedc61f31bab473799060c25daf0b4c9c21fa`;
- noise-dose raw/summary/verifier SHA-256:
  `894e7281c05121b03279e767e410e2c7f48faf3405b1addab43b3d8ad5facd73`,
  `b9d80c4b7e976eb1ace203f849ae59fb6a2591740045bb9aa89eb163caf1c249`,
  `fa1ce9a7eb08452ef2b8b5fdbba148b809bf193ffee34a1e26b097cd31014741`;
- no data, predictions, checkpoints, metrics, caches, or large figures are
  committed.

The software path is therefore: hardware validity mask -> per-finger exact
fallback -> current DirectPose correction only on valid channels -> WiLoR local
pose otherwise. The model is a promising HOT3D-centered conditional aid, but a
physical-sensor pilot and the frozen ARCTIC noise ceiling are required before
calling it deployable.
