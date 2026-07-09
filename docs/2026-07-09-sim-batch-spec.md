# Sim Batch Spec: Calibration Tolerance + Action-Space Scissors (2026-07-09)

Audience: Codex (adversarial review + HPC submission + result notes), Claude
(code prep before handoff). User approved E1 and E2; E3 is OPTIONAL and runs
only after an explicit user go (hardware-v2 pricing).

House rules that apply to everything below: `python -m unittest discover -s
tests` green before handoff; adversarial review before submission; every new
cell flows through `scripts/rope_refiner/summarize_runs.py` (numbers enter the
report pack from generated tables, never hand-copied); `json_sanitize` on every
dump; `sample_id` alignment with loud failure; local tests use the injectable
`FakeMano` pattern.

Shared references:

- Winner teacher recipe: `rope + flex15 + gate010 + steps=400 lr=32
  alpha_l2=0.001 max_alpha=0.5 batch=512` (0036).
- Release student: 0044 four-teacher augmented multi student; checkpoint
  mirrored in `.local_checks/releases/` and
  `/data/wentao/ropetrack/releases/p2_four_teacher_student/`.
- Baseline noise ablation for comparison: gaussian std 0.05 retains 82%
  (report pack Act 3).

---

## E1 — Calibration-error (bias/scale) tolerance curve

### Why

The physical rope device exists. Its dominant real error is NOT zero-mean
jitter but calibration error: per-string zero offset (bias) and hand-size /
gain mismatch (scale). The published ablation only covers gaussian noise +
dropout. `gate010` is specifically vulnerable to DC bias: a constant offset
manufactures residual on every finger, the gate opens everywhere, and harmful
closure hits fingers that were correct. Output of E1 = the tolerance curve
that (a) sets the calibration spec for the real-hardware protocol and (b)
decides whether bias-augmented student retraining is required.

### Code changes (Claude, before submission)

1. `ropetrack/refine/analysis.py::perturb_rope_reading` — extend the single
   injection point with three new, default-off parameters:
   - `bias_std`: per-sample, per-finger offset `b_f ~ N(0, bias_std)`,
     independent across fingers (each string is calibrated separately).
   - `bias_fixed`: one scalar added to ALL fingers and samples (worst-case
     coherent DC miscalibration).
   - `scale_range r`: per-sample global gain `s ~ U[1-r, 1+r]` applied to all
     fingers of that sample (hand-size error is common-mode).
   Perturbation order: `rope' = clip(s * rope + b + bias_fixed + gaussian, 0, 1)`.
   Seeded through the existing rng path; existing callers unchanged.
   Semantics note: a real session has ONE fixed bias vector; the per-sample
   random draw estimates expected damage over possible miscalibrations, the
   `bias_fixed` mode covers the coherent worst case. Report both.
2. `scripts/rope_refiner/apply_rope_refinement.py` — plumb
   `--rope-noise-bias-std`, `--rope-noise-bias-fixed`,
   `--rope-noise-scale-range` for BOTH `optimize` and `student` modes. The
   student's residual/valid input features must be computed from the perturbed
   reading, exactly as the gaussian path does today.
3. `scripts/rope_refiner/train_alpha_student.py` — same three knobs as
   training augmentation (used only in Phase 2).
4. Tests: perturb semantics (clip, seeding, per-finger independence vs
   common-mode scale), CLI plumb-through, FakeMano end-to-end smoke for both
   modes.

### Cells (FreiHAND mask70 eval; teacher = winner recipe, student = release checkpoint)

| Sweep | Values | Paths | Cells |
|---|---|---|---:|
| A random bias | bias_std ∈ {0.025, 0.05, 0.075, 0.10} | teacher + student | 8 |
| B coherent DC | bias_fixed ∈ {+0.05, −0.05} | teacher + student | 4 |
| C scale | r ∈ {0.05, 0.10} | teacher + student | 4 |
| D combined | bias_std 0.05 + r 0.05 | teacher + student | 2 |

18 cells, each an apply+score run of the same shape as the published noise
cells. Record standard sliced scores + closure + gated fraction + alpha mean
abs (gated fraction is a named observable here, not incidental).

### Pre-registered predictions / gates

- P1: coherent DC `bias_fixed = 0.05` retains LESS of the all-joint gain than
  gaussian 0.05 did (82%) — DC defeats gating and pulls fingers coherently.
- P2: teacher gated fraction decreases monotonically with bias magnitude
  (gate opening on manufactured residual).
- P3: the release student (gaussian-only augmentation) degrades under bias at
  least as fast as the teacher — it has never seen DC errors.
- Decision gate: if student retention at `bias_std = 0.05` < 60%, Phase 2 is
  mandatory before any real-hardware eval.

### Phase 2 (conditional on the gate)

Retrain the multi student with bias/scale augmentation (suggested start:
bias_std 0.05, r 0.05) with the full mandatory control set (shuffle, seeds,
clean split, no-aug comparison). Gate: bias-augmented student retains ≥80% at
bias_std 0.05 while its clean-calibration mask70 gain stays within 0.05 mm of
the release student.

### Deliverables

`bias_curve.png` (x = bias_std; teacher and student lines; DC cells as
markers), summarize_runs tables, report-pack extension section, experience
note.

---

## E2 — Action-space scissors plot (oracle vs rope as DOF grows)

### What this is (plain words)

One figure, x-axis = action-space dimension (5 → 15 → 45). For each action
space, run the SAME strong-recipe test-time optimization twice: once
rope-driven, once oracle-driven (GT targets). Plot all-joint gain vs
dimension. Expected shape: the oracle line keeps rising with dimension (a more
expressive space can fix more), while the rope line flattens after 15 (five
scalars cannot determine more DOF). The two lines opening like scissors is the
point: it proves the bottleneck is sensor information, not the correction
module — the definitive answer to "just make the module bigger" — and the gap
between the lines at dim 45 prices what richer sensing could buy (input to any
hardware-v2 discussion, see E3).

### Code changes (Claude)

1. `ropetrack/refine/actions.py` — new `pose45` action space: alpha `[N, 45]`
   additive axis-angle delta on `hand_pose` (15 joints × 3 axes), no
   per-sample directions needed. Touch `ACTION_SPACES`, `alpha_dim` (45),
   `_check_alpha`, `apply_action_np`/`apply_action_torch`
   (`base + alpha.reshape(N,45)` after the optimizer's tanh/max_alpha bound,
   same as flex paths), `per_finger_alpha_abs` (aggregate 9 dims/finger via
   `FINGER_POSE_GROUPS`). Per-finger gating works through `JOINT_TO_FINGER`
   unchanged (gate zeroes that finger's 9 dims). np/torch parity tests +
   FakeMano smoke.
2. `scripts/rope_refiner/apply_rope_refinement.py` — accept `pose45`
   everywhere action spaces are enumerated (CLI choices, alpha summary).
3. No oracle code: `oracle_chain` (all 20 non-wrist joints as targets) is
   already implemented in `ropetrack/refine/oracle.py` — it has simply never
   been run at the strong recipe.

### Cells (FreiHAND mask70; strong recipe fixed)

Reused, do NOT rerun (0036/0038): rope mult5_gate010, rope flex15_gate010,
oracle_tip mult5, oracle_tip flex15.

New cells:

| Objective | Action space | Note |
|---|---|---|
| oracle_chain | mult5 | fills the 5-dim chain point |
| oracle_chain | flex15 | quantifies non-tip target info in the fixed space |
| oracle_chain | pose45 | top-right of the scissors |
| oracle_tip | pose45 | |
| rope (gate010) | pose45 | does rope exploit extra DOF? |
| oracle_chain (alpha_l2=0.004) | pose45 | single regularization-sensitivity sanity cell |

Convention (same as published Act 5): rope cells gated (deployment config),
oracle cells ungated (pure ceiling). Optional if budget allows: repeat the new
cells on finger_end80 (the largest-headroom split).

### Pre-registered predictions / gates

- P1: `oracle_chain/pose45` all-joint gain exceeds `oracle_tip/flex15`
  (−1.97 mm) by ≥0.3 mm — the ceiling rises with space + richer targets.
- P2: `rope/pose45` lands within ±0.15 mm of `rope/flex15` (−1.68 mm) or
  worse — rope cannot exploit the extra DOF; the scissors open.
- P3 (descriptive, no gate): `oracle_chain/flex15` vs `oracle_tip/flex15`
  isolates the value of non-tip targets inside the same space.
- If P2 is REFUTED (rope/pose45 wins by >0.15 mm): that is a major positive
  surprise (minimum-norm optimization extracts more than observability
  suggests) — stop and revisit the student's action space before any hardware
  decision.

### Deliverables

`scissors.png` (x = dim {5, 15, 45}; lines: rope, oracle_tip, oracle_chain),
tables, report-pack extension, experience note.

---

## E3 — OPTIONAL: rope-configuration pricing for hardware v2 (needs user go)

### Why

Colleague proposal: two strings per finger, because PIP/DIP flexion decouples
in real grasps (hook grip, pressing) and one tip→wrist length only measures
aggregate curl (1 constraint vs 3 DOF per finger in flex15). Abduction
(侧摆) is nearly invisible to tip→wrist lengths. Since rope labels derive from
GT joints, candidate sensor layouts can be priced in sim by the label
generator BEFORE any hardware is built — including where the second string
should attach.

### Configs

- `R5` — baseline tip→wrist (existing labels).
- `R10-PIP` — R5 + one PIP-joint→wrist distance per finger (the colleague's
  2-per-finger layout; tells the optimizer WHERE along the finger the curl
  happens).
- `R9-SPREAD` — R5 + 4 adjacent fingertip-to-fingertip distances
  (thumb-index, index-middle, middle-ring, ring-pinky). Information probe for
  abduction; mechanical realizability is TBD and explicitly not claimed.

### Code (larger; only on user confirmation)

`scripts/make_rope_labels.py` generalized to arbitrary joint-id endpoint pairs
(schema version bump); per-finger residual/gating aggregates that finger's
ropes by mean; `apply_rope_refinement.py` target dim follows the label schema.
First pass is TEACHER-ONLY (test-time optimization on eval splits — no student
retraining, no new training data), which keeps E3 cheap.

### Cells (first pass)

{R10-PIP, R9-SPREAD} × {flex15_gate010, pose45_gate010} on mask70 +
finger_end80 eval. pose45 is included because 2 constraints/finger may start
to justify more DOF — the "sensors unlock action space" interaction is the
headline question.

### Pre-registered predictions (draft gates; tighten after E2 lands)

- P1: R10-PIP/flex15 beats R5/flex15 (−1.68 mm) by ≥0.10 mm on mask70 and
  closes ≥50% of the gap to oracle_tip/flex15 (−1.97 mm).
- P2: R9-SPREAD helps finger_end80 more than mask70 (spread errors weigh more
  when all five fingers are disturbed).
- P3: R10-PIP/pose45 > R10-PIP/flex15 (with more constraints, extra DOF stops
  being pure hallucination room).

---

## Explicitly NOT in this batch

Occlusion-augmented fine-tune baseline and rope-LoRA conditioning: both
require a WiLoR training harness (hard-root dataloader with MANO GT,
keypoint/param losses, LoRA / head-only options, clean+masked multi-task mix
to protect clean performance). They share ~90% of that infrastructure, so it
gets built once, later, as its own spec — the fine-tune baseline is the
control arm of the LoRA experiment and they should land together. Interim
reviewer ammo on "why not just use image features": the P3 v0 image-only
control (frozen-feature form of the question) is already recorded (0049).

## Run mechanics (Codex)

Run roots `rope_e1_bias_<ts>` / `rope_e2_scissors_<ts>` (+ `rope_e3_cfg_<ts>`
if it runs); archive launchers per `.local_checks` convention; pull tables
into `.local_checks/`; extend the report pack only via `summarize_runs.py`
output; write the experience note after results land. E1 and E2 are
independent — submit in parallel. E1 Phase 2 waits for the E1 curve. E3 waits
for the user's explicit go.
