# RELEASE.md — RopeTrack P2 Release Model

Pins the identity, provenance, and reproduction path of the released model.
Numbers cite `docs/2026-07-07-report-results-pack.md` (the only number
source); details cite `experience/` notes.

## Model Identity

**The 0044 four-teacher augmented multi alpha-student.**

- Checkpoint (remote release copy):
  `/data/wentao/ropetrack/releases/p2_four_teacher_student/student.pt`
- Original checkpoint:
  `/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104/students/student_multi/student.pt`
- Architecture: `RopeAlphaStudent` MLP 65 -> 256 -> 256 -> 15
  (`ropetrack/refine/alpha_student.py`); input = base MANO hand_pose (45) |
  base rope (5) | rope reading (5) | residual (5) | valid (5); output = 15
  flex15 alphas, tanh-bounded at max_alpha 0.5, zero-init final layer.
- The residual gate (threshold 0.1, normalized rope units) is a hard rule
  applied at inference — never learned.
- Trained with sensor-noise augmentation (std 0.05, dropout 0.1), val split
  0.1 + early stopping, seed 0; `beats_zero_baseline=True`.
- Apply with:
  `scripts/rope_refiner/apply_rope_refinement.py --mode student --checkpoint <student.pt> ...`

## Frozen Teacher Recipe (produces the training targets)

```
apply_rope_refinement.py --mode optimize --objective rope --action-space flex15 \
  --gate-residual-threshold 0.1 --opt-steps 400 --opt-lr 32 --opt-alpha-l2 0.001 \
  --batch-size 512
```

Selected in P1 Batch B (0035/0036). The optimize loss is a batch mean, so
the effective lr couples to batch size — keep batch 512 when reproducing.

## The Four Train Teachers (0037/0039/0041/0043)

| Teacher | Samples | Closure | Remote dir |
|---|---:|---:|---|
| FreiHAND mask70 (WiLoR) | 32,560 | 0.504 | `runs/rope_p2_train_teacher_20260707_121353/teacher/rope_flex15_gate010` |
| FreiHAND finger_end80 (WiLoR) | 32,560 | 0.525 | `runs/rope_p2_queue_20260707_135405/q1_finger_end80_train/teacher` |
| FreiHAND mask70 (HaMeR) | 32,560 | 0.508 | `runs/rope_p2_queue_20260707_135405/q2_hamer_mask70_train/teacher` |
| HO3D v3 mask70 stride4 (WiLoR) | 20,832 | 0.582 | `runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/teacher/rope_flex15_gate010` |

Diversity caveat: the three FreiHAND teachers share the same 32,560 images
(different perturbation/backend) — 2 image corpora, not 4. Teacher-quality
parity is a precondition: the under-converged HO3D v3 finger_end80 teacher
(closure 0.34) made the five-teacher variant WORSE on HO3D (0047/0049).

## Headline Numbers (held-out eval split; same-decoder deltas, mm)

| Eval axis | Student gain | vs teacher | Note |
|---|---:|---:|---|
| FreiHAND mask70 | -1.64 | 97% | occluded-tip -5.25 |
| FreiHAND finger_end80 | -1.62 | 97% | cross-perturbation |
| HO3D v2 mask70 | -0.97 | 93% | cross-dataset |
| HaMeR mask70 | -1.70 | 100% | cross-backend |
| mask70 + sensor noise 0.05 | -1.54 | 110% | beats the teacher under noise |
| Shuffled-rope control | -0.06 | — | gain collapses: signal is the rope |
| Clean split | +0.04 vs baseline | — | neutral (±0.05 band) |

## Reproduction From a Fresh Clone (per teacher, then student)

1. Hard train root + rope labels: `scripts/make_hard_images.py` +
   `scripts/make_rope_labels.py` (`--split training`, HO3D adds `--stride 4`;
   exact commands in 0037/0039/0041).
   New standalone labels go under `/data/wentao/ropetrack/rope_labels/<dataset>/`;
   training hard roots may also carry an aligned local `rope_labels.jsonl`.
2. Export with MANO cache: `scripts/eval.py --split training
   --save-mano-cache` (HO3D train root: `--dataset ho3d_v3_mask70_train
   --protocol-check-samples 0`).
3. Teacher: the frozen recipe above, per train root.
4. Student: `scripts/rope_refiner/train_alpha_student.py --teacher-dir <4 dirs>
   --action-space flex15` (+ `--shuffle-rope` for the mandatory control).
5. Eval: `apply_rope_refinement.py --mode student` per eval cell, then
   `scripts/score_predictions.py`, `scripts/score_sliced_predictions.py`,
   aggregated by `scripts/rope_refiner/summarize_runs.py`.

Requires HPC data roots (`/data/wentao/ropetrack/...`) and MANO files;
per-stage Slurm launchers are archived in untracked `.local_checks/`
(submit_p2_student_multi.sh and run_* scripts) — the tracked per-step
commands above are canonical.

## Must-Ship Caveats

The five report-pack caveats (simulated GT-derived sensor with the noise
curve as realism bound; eval-selected optimizer recipe mitigated by
zero-retuning transfer; same-decoder protocol boundaries; clean-image
neutrality, not improvement; effective teacher diversity = 2 corpora) plus:
lr/batch coupling above; the 0028 HO3D rope rows predate the joint-order fix
(0029) and are superseded by 0038; legacy 45-dim MLP refiner
(`--mode checkpoint`) was removed in the consolidation pass — its negative
result stays documented in experience/0026.

## Stage Commits

P1 flex/gating `9040b29`; Batch B results `0b2498a`; noise flags `8e1ecc7`;
student tooling `4d84d92`; multi-teacher `2964618`; release results recorded
`8d876ea` (0044) and `dac2633` (0049).
