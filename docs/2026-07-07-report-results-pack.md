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

## Pending slots (fill when Codex's final batch lands)

- [ ] Multi-teacher student (4 teachers incl. HO3D v3 s4): same 6-cell eval
      table as Act 4; key question = does HO3D transfer improve without
      hurting FreiHAND cells.
- [ ] Multi-teacher shuffle control.
- [ ] noaug @ noise0.05 cell -> pick deployment default (main vs noaug).
- [ ] Qualitative figure: base/teacher/student mesh triptychs, top-3 mask70.
- [ ] Latency numbers from job logs + timed student forward.
- [ ] HO3D v3 s4 train teacher sanity row (closure/gated/alpha vs FreiHAND
      train teacher).
