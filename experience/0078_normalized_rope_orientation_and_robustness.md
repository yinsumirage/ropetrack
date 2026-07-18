# Normalized-rope global orientation and calibration robustness

Date: 2026-07-19

## Questions and frozen protocol

This batch keeps the existing five rope readings normalized to `[0, 1]`. It
does not test centimetre-valued inputs or change calibration semantics. It
asks two narrower questions:

1. Can frozen localized WiLoR tokens plus normalized rope learn a residual
   MANO global orientation, beyond the existing 45D hand-pose correction?
2. How much additive noise, coherent offset/gain error, and channel dropout
   can the participant-CV direct pose head tolerate at inference?

The HOT3D test protocol is the frozen 0077 split: 8,147 natural-visibility
hands from nine participants, with each participant scored only by the fold
that excluded all of that participant's frames. Every orientation head uses
the corresponding expanded h128 pose head and fold bundle. Model selection
uses only an internal episode split of the training bundle. Official training
GT supplies the joint loss; test GT is used only by scorers and the diagnostic
oracle. No example is selected by test error.

Run root:
`/data/wentao/ropetrack/runs/direct_pose_ab_20260719`.

## Implementation

`scripts/rope_refiner/direct_pose_head.py` now has default-off apply controls
for normalized rope noise, dropout, fixed/random bias, fixed/random gain, and
joint-only output. The existing clean path is unchanged at all-zero/default
settings.

`scripts/rope_refiner/direct_global_orient_head.py` implements the bounded
axis-angle residual screen. Its state is the base global orientation and hand
pose; optional inputs are a single cross-attention over the frozen 4x3 WiLoR
tokens and the same base/input/residual/valid rope features as the pose head.
The loss is root-relative 3D joint L1 after the frozen pose correction. A
deterministic control shuffles measured rope and validity together within each
participant while leaving the pose head's rope input correct.

This is an experiment module, not a release-path replacement.

## A0: rotation headroom exists on HOT3D

The per-frame SVD oracle removes only a rigid rotation after wrist centering.

| Input | Camera | Root-relative | Rotation oracle | Rotation headroom |
|---|---:|---:|---:|---:|
| WiLoR base | 138.581 | 66.574 | 12.289 | 54.285 |
| expanded pose head | 139.539 | 65.342 | 8.030 | 57.311 |

The expanded-head rotation-headroom episode-bootstrap 95% CI is
`[53.061, 61.349]` mm, and it is positive for all 9/9 participants. This
justifies a learned screen, but the oracle is not an inference method.

## A1: HOT3D global-orientation screen

All values are joint error in mm. PA is invariant to a rigid global rotation
up to numerical precision.

| Method | Camera | Root-relative | PA |
|---|---:|---:|---:|
| WiLoR base | **138.581** | 66.574 | 8.984 |
| expanded pose head | 139.539 | 65.342 | 5.732 |
| pose + RGB orientation | 171.350 | 24.578 | 5.732 |
| pose + rope orientation | 169.860 | 33.848 | 5.732 |
| pose + RGB+rope orientation | 170.748 | **23.571** | 5.732 |
| RGB+rope, measured rope shuffled within participant | 170.886 | 25.576 | 5.732 |

RGB+rope improves root-relative error by 1.007 mm over RGB alone, with a
paired-episode 95% CI of `[-1.752, -0.249]` mm. Shuffling only the orientation
branch's measured rope worsens it by 2.005 mm, CI `[1.553, 2.465]` mm. Thus
paired per-frame rope contributes to the learned HOT3D orientation result; it
is not merely an image branch or participant-range effect.

The camera result is a real failure, not a missing wrist-centre patch. MANO's
wrist is already the local origin, so rotating about MANO origin and rotating
about wrist are equivalent here. The large base camera translation error had
been partly cancelled by the old rotation. Correcting root-relative rotation
removes that accidental cancellation, and no GT-free translation adjustment
is available in this module.

## A2: the orientation gain does not transfer to ARCTIC

The same three fold heads were frozen and applied to the independent ARCTIC
P2-val stride-10 anchor. The table gives the three-model mean; all three folds
regress root-relative error.

| Method | Camera | Root-relative | PA |
|---|---:|---:|---:|
| expanded pose head | **51.577** | **14.893** | 5.979 |
| pose + RGB orientation | 52.964 | 23.851 | 5.979 |
| pose + RGB+rope orientation | 53.302 | 23.673 | 5.979 |

RGB+rope fold root errors are 19.947, 22.325, and 28.747 mm, versus clean
15.001, 14.857, and 14.822 mm. The HOT3D result therefore contains a strong
dataset coordinate/camera-convention component. Do not claim a general global
orientation solution and do not add this head to the release model. A future
retry needs dataset-balanced validation and an explicit translation/global
frame design, not another larger residual head.

## B: normalized-rope perturbation matrix

The expanded h128 pose head is frozen. The clean reference is PA 5.732 mm;
WiLoR base is 8.984 mm, so clean gain is 3.252 mm. `Retained` is
`(base - perturbed) / (base - clean)`. Gaussian noise and dropout cells use
three seeds; the table reports their mean and seed range. Fixed bias is one
coherent offset on every normalized reading, and fixed gain multiplies every
reading before clamping to `[0,1]`.

### Noise and coherent bias

| Condition | PA all | Seed range | PA low visibility | Delta vs clean | Retained |
|---|---:|---:|---:|---:|---:|
| clean | 5.732 | - | 6.148 | +0.000 | 100.0% |
| noise 0.025 | 5.958 | 5.956-5.960 | 6.380 | +0.226 | 93.0% |
| noise 0.050 | 6.485 | 6.481-6.491 | 6.925 | +0.753 | 76.8% |
| noise 0.100 | 7.851 | 7.841-7.866 | 8.322 | +2.119 | 34.8% |
| noise 0.200 | 10.598 | 10.557-10.627 | 11.112 | +4.867 | -49.6% |
| bias -0.100 | 6.442 | - | 6.934 | +0.710 | 78.2% |
| bias -0.050 | 5.775 | - | 6.234 | +0.043 | 98.7% |
| bias -0.025 | 5.653 | - | 6.088 | -0.079 | 102.4% |
| bias +0.025 | 6.072 | - | 6.483 | +0.340 | 89.5% |
| bias +0.050 | 6.641 | - | 7.059 | +0.910 | 72.0% |
| bias +0.100 | 8.080 | - | 8.523 | +2.349 | 27.8% |

The direction asymmetry is model/dataset specific: this direct HOT3D head is
more sensitive to positive bias, unlike the older FreiHAND release-student E1
screen. At `+0.05`, the paired PA degradation CI is `[0.783, 1.035]` mm;
at `-0.05`, it is `[-0.085, 0.175]` mm.

### Coherent gain

| Gain | PA all | PA low visibility | Delta vs clean | Retained |
|---:|---:|---:|---:|---:|
| 0.80 | 7.586 | 8.072 | +1.854 | 43.0% |
| 0.90 | 6.171 | 6.661 | +0.439 | 86.5% |
| 0.95 | 5.747 | 6.199 | +0.015 | 99.5% |
| 1.05 | 6.340 | 6.762 | +0.609 | 81.3% |
| 1.10 | 7.335 | 7.787 | +1.604 | 50.7% |
| 1.20 | 9.045 | 9.457 | +3.313 | -1.9% |

Roughly 5% gain error is acceptable in this simulation. Ten percent is
already asymmetric and a 20% positive gain erases the clean benefit.

### Missing channels and combined errors

| Condition | PA all | Seed range | PA low visibility | Delta vs clean | Retained |
|---|---:|---:|---:|---:|---:|
| dropout 0.10 | 7.952 | 7.937-7.966 | 8.260 | +2.220 | 31.7% |
| dropout 0.20 | 9.979 | 9.935-10.013 | 10.162 | +4.247 | -30.6% |
| dropout 0.40 | 13.453 | 13.400-13.489 | 13.436 | +7.722 | -137.4% |
| noise 0.05 + bias -0.05 | 6.375 | 6.370-6.383 | 6.836 | +0.644 | 80.2% |
| noise 0.05 + gain 0.90 | 6.669 | 6.662-6.677 | 7.145 | +0.937 | 71.2% |
| noise 0.05 + dropout 0.20 | 10.477 | 10.465-10.491 | 10.658 | +4.745 | -45.9% |

Dropout is the dominant deployment failure: even 10% missing readings retain
only 31.7% of the clean gain, and 20% is worse than using the base model. The
validity mask alone does not make the direct head robust to missing channels.

## Operational record

- Job `186254` failed because the isolated Git worktree already contained an
  empty gitlink directory, so the WiLoR symlink was accidentally created one
  level too deep. Removing only that empty directory and linking
  `third_party/wilor` directly fixed the import; smoke `186255` completed.
  Worktree isolation itself was not the model failure.
- Jobs `186256`, `186311`, `186313`, and `186314` produced and scored all 42
  perturbation conditions over three folds. The tail job skips already-valid
  summaries, avoiding duplicate GPU work.
- Oracle/orientation jobs `186253`, `186266`, `186275`, `186350`, `186354` to
  `186356`, shuffle jobs `186368`, `186371`, `186380`, and ARCTIC jobs
  `186382` to `186383` completed with exit code 0.
- Score job `186373` used stale argument names and failed before reading model
  outputs; corrected job `186380` completed. Job `186372` was immediately
  cancelled because it was attached to a guessed global Slurm ID rather than
  the returned stitch ID. Do not infer dependent IDs between submissions.

## Decision

Keep the original normalized `[0,1]` input for the next matched experiment;
this batch provides no evidence that centimetre-valued rope would improve the
model. It does not prove that per-session min/max calibration is drift-free.
Instead it establishes a practical simulated tolerance region: approximately
noise `<=0.05`, coherent gain near `0.95-1.05`, and modest offset, with clear
direction asymmetry.

Do not promote the current global-orientation head: it learns a strong HOT3D
root-relative correction with real rope contribution, but worsens camera error
and fails the ARCTIC transfer gate.

The next robustness training experiment should be small and targeted: clean
control versus moderate continuous calibration augmentation versus explicit
rope-channel dropout augmentation, with clean PA and the frozen matrix above
as joint gates. Do not launch another broad head-size, LoRA, or input-unit
sweep before that matched test. Hardware logs should record raw values,
calibrated `[0,1]` values, and validity independently so this simulated
tolerance curve can later be checked against real session drift.
