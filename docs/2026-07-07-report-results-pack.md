# Report Results Pack (2026-07-07)

Consolidated, report-ready tables for the teacher progress report. All deltas
are same-decoder PA-MPJPE in mm (negative = improvement) unless noted; sliced
values come from `score_sliced_predictions.py`, aggregated by
`summarize_runs.py` (no hand-copied numbers; sources:
`.local_checks/p2_student_tables_final`, `.local_checks/p2_queue_tables_final`,
experience 0013/0020/0032/0035/0036/0038/0042).

## Act 1 — Occlusion breaks the RGB baseline

| Quantity | Value | Source |
|---|---:|---|
| FreiHAND clean PA-MPJPE (WiLoR, GT bbox) | 4.99 mm | 0013 |
| FreiHAND mask70 hard PA (same-decoder base) | 10.07 mm | 0036 |
| Occlusion damage | **+5.07 mm** | |
| Diagnosed failure mode | occluded curled fingers predicted too open | 0023 |

## Act 2 — Test-time rope optimization repairs 1/3 of the damage

Dose-response (FreiHAND mask70, rope/mult5; the under-optimization diagnosis):

| Recipe | Rope closure | All-joint gain (mm) |
|---|---:|---:|
| lr=2, steps=120 (0032 default) | 3.8% | 0.16 |
| lr=8, steps=400 | 27.8% | 1.00 |
| lr=32, steps=120 | 30.1% | 1.07 |
| lr=32, steps=400 | 41.8% | 1.40 |

Figure: `dose_response.png`.

Winner = `rope + flex15 + gate010 + steps=400/lr=32/alpha_l2=0.001` (0036):

| Split | PA (mm) | All-joint gain | Occluded-tip gain | Closure |
|---|---:|---:|---:|---:|
| FreiHAND mask70 (WiLoR) | 8.39 | **-1.68** | **-5.47** | 48.5% |
| FreiHAND finger_end80 (WiLoR) | 8.32 | -1.66 | -3.45 | 50.6% |
| FreiHAND mask70 (HaMeR) | 9.12 | -1.70 | -5.64 | 51.3% |
| HO3D v2 mask70 (WiLoR, no retuning) | 8.39 | -1.05 | -2.09 | 54.2% |
| HO3D v2 mask70 (HaMeR, no retuning) | 8.67 | -0.66 | -0.82 | 55.4% |

Mechanism evidence for gating: ungated flex15 closes MORE rope (49.7% vs
48.5%) but scores WORSE (-1.39 vs -1.68 mm) — harmful closure on low-residual
fingers is real; the gate trades closure for pose quality (0036).

Recovery framing: 1.68 mm of the 5.07 mm occlusion damage = **~33% repaired**
by a training-free correction from 5 scalars.

## Act 3 — Sensor realism (noise/dropout on the winner, mask70)

| Noise std (approx. mm) | Dropout | All-joint gain (mm) | Retained | Clean-tip gain (mm) |
|---:|---:|---:|---:|---:|
| 0 | 0 | -1.68 | 100% | -0.24 |
| 0.025 (~1.2mm) | 0 | -1.60 | 95% | -0.22 |
| 0.05 (~2.5mm) | 0 | -1.38 | 82% | -0.16 |
| 0.05 | 0.2 | -1.01 | 60% | - |
| 0.075 (~3.8mm) | 0 | -1.09 | 65% | -0.10 |
| 0.10 (~5mm) | 0 | -0.78 | 46% | - |
| 0.15 (~7.5mm) | 0 | -0.19 | 11% | **+0.10 (harmful)** |
| 0.20 (~10mm) | 0 | +0.38 (harmful) | fails | - |

Claim wording: moderate sensor noise (<=2.5 mm) keeps >80% of the gain; the
usefulness boundary is ~5-7.5 mm; beyond that the correction turns harmful.
Figure: `noise_curve.png`.

## Act 4 — Distillation: train-split student, eval-split verification

Student: 65-d input (base pose + rope reading + residual + valid) -> 15 alphas,
one forward pass. Trained ONLY on the FreiHAND mask70 TRAINING split teacher
(32,560 samples); every number below is the held-out EVALUATION split.

| Eval cell | Student gain (mm) | Teacher gain (mm) | Recovery |
|---|---:|---:|---:|
| FreiHAND mask70 | -1.64 | -1.68 | **97%** |
| FreiHAND finger_end80 (cross-perturbation) | -1.60 | -1.66 | 97% |
| HO3D v2 mask70 (cross-dataset) | -0.91 | -1.05 | 87% |
| HaMeR mask70 (cross-backend) | -1.65 | -1.70 | 97% |
| mask70 with sensor noise 0.05 | **-1.53** | -1.38 | **110% (beats teacher)** |

Controls and stability:

| Check | Result |
|---|---|
| Shuffled-rope control | gain collapses to -0.05 mm, closure 0.7%, alpha 0.0016 -> the gain provably comes from the rope signal |
| Seeds 0/1/2 | 8.432 / 8.436 / 8.438 mm PA (+-0.003 mm) |
| Clean split | student 4.979 vs baseline 4.956 mm = +0.023 mm, inside the +-0.05 mm neutrality band (gated teacher was +0.10 mm) |
| No-aug variant | -1.72 mm on clean input (slightly beats teacher: amortization smooths per-sample optimization jitter); noise-input cell pending to justify the augmented default |
| Occluded-tip | student -5.26 mm vs teacher -5.47 mm |

Latency (pending exact numbers from job logs): teacher = 400 MANO
forward/backward per sample; student = one 65->256->256->15 forward pass
(sub-ms), decode shared.

## Act 5 — Ceilings and remaining headroom (strong-recipe oracle_tip/flex15)

| Split | Oracle gain (mm) | Rope winner (mm) | Net headroom (mm) | Rope reaches |
|---|---:|---:|---:|---:|
| FreiHAND mask70 | -1.97 | -1.68 | 0.29 | 85% |
| FreiHAND finger_end80 | -2.48 | -1.66 | 0.82 | 67% |
| HO3D v2 mask70 (WiLoR) | -1.16 | -1.05 | 0.11 | 90% |
| HO3D v2 mask70 (HaMeR) | -1.29 | -0.66 | 0.63 | 51% |

Wording caution: on HO3D/WiLoR the rope correction is already near its
action-space ceiling; the case for image-conditioned refinement (P3) rests on
finger_end80 and HaMeR headroom, not on a blanket "large gap".

## Must-state caveats (own them before the advisor asks)

1. **Simulated sensor**: rope readings are derived from GT joints; the noise
   ablation (Act 3) is the realism bound. The 5 scalars carry far less than
   the 63-dim answer (oracle gap in Act 5 proves no trivial leakage).
2. **Recipe provenance**: optimizer hyperparameters were selected on the eval
   split; mitigations = zero-retuning transfer (HO3D, finger_end80, HaMeR)
   and untuned student hyperparameters with +-0.003 mm seed stability.
3. **Protocol boundaries**: all deltas are same-decoder; do not splice with
   the 0020 hard-baseline table at joint level (~0.3-0.5 mm protocol offset);
   cross-table comparisons use mesh/F metrics.
4. **Clean images**: the pipeline is a hard/occluded-sample corrector; on
   clean images the student is neutral (+0.02 mm), not an improvement.
5. **Effective teacher diversity**: the three FreiHAND-based train teachers
   share the same 32,560 underlying images (different perturbation/backend);
   dataset diversity is 2 corpora, not 4.

## Act 4b - Multi-teacher student (final model; 0044)

Four train teachers (FreiHAND mask70 / finger_end80, HaMeR mask70, HO3D v3
mask70 stride4 = 118,512 samples), same student architecture, augmented:

| Eval cell | Multi student (mm) | Single student (mm) | Verdict |
|---|---:|---:|---|
| FreiHAND mask70 | -1.636 | -1.636 | tie - FreiHAND unharmed |
| FreiHAND finger_end80 | -1.623 | -1.603 | slightly better |
| HO3D v2 mask70 | **-0.972** | -0.911 | +0.06 mm from HO3D v3 train data (recovery 87% -> 93%) |
| HaMeR mask70 | **-1.697** | -1.655 | reaches 100% of the HaMeR teacher |
| mask70 + noise 0.05 | -1.539 | -1.527 | slightly better |
| Shuffled control | -0.056 | - | still collapses |
| Clean split | +0.041 mm vs baseline | +0.023 mm | inside the +-0.05 band, closer to its edge - state it |

Deployment-default decision (noaug @ noise0.05 cell): no-aug retains 82% of
its clean gain under sensor noise (-1.414), the augmented student retains 93%
(-1.527/-1.539) -> **augmentation trades 0.08 mm clean accuracy for noise
robustness; the augmented multi student is the release model.**

HO3D v3 s4 train teacher sanity: closure 0.582, gated 0.282, alpha 0.043 -
same regime as the FreiHAND train teacher (0.504/0.288/0.053).

Qualitative panels (`.local_checks/p2_student_multi_report_panels`): on the
four most-improved mask70 samples, base PA 28.4-31.5 mm drops to 12.8-18.0 mm
(halved); the one-pass student matches or slightly beats the 400-step teacher
on 3 of 4.

## Act 4c - Multi-v2 ablation (5 teachers; 0047)

Adds HO3D v3 finger_end80 stride4 teacher, for 139,344 teacher samples total.
Conclusion: useful ablation, **not** the release model; keep the 0044
four-teacher augmented multi student as default.

| Eval cell | Multi-v2 gain (mm) | Four-teacher gain (mm) | Verdict |
|---|---:|---:|---|
| FreiHAND mask70 | -1.636 | -1.636 | tie |
| FreiHAND finger_end80 | -1.619 | -1.623 | tie/slightly worse |
| HO3D v2 mask70 | -0.834 | -0.972 | worse; adding HO3D finger_end teacher did not help this axis |
| HO3D v2 finger_end80 | -0.524 | - | new axis; positive but modest |
| HaMeR mask70 | -1.685 | -1.697 | tie/slightly worse |
| mask70 + noise 0.05 | -1.536 | -1.539 | tie |
| Shuffled control | -0.059 | -0.056 | still collapses |

New HO3D finger_end80 slice: occluded-tip gain -0.735 mm, closure 0.343,
gated fraction 0.499. This says the added teacher produces some transfer to
the matching HO3D finger-end disturbance, but not enough to justify replacing
the four-teacher release model.

P3 asset note: frozen WiLoR feature caches are ready for FreiHAND mask70
eval/train: `(3960, 1280)` and `(32560, 1280)`.

## Pending slots

- [ ] Latency numbers from job logs + timed student forward.
