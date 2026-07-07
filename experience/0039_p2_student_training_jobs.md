# 0039 P2 Student Training And Queue Jobs

Date: 2026-07-07

## Code

Committed P2 alpha-student tooling:

```text
4d84d92 Add alpha student distillation tools
```

Added:

- `ropetrack/refine/alpha_student.py`
- `scripts/rope_refiner/train_alpha_student.py`
- `scripts/rope_refiner/summarize_runs.py`
- `scripts/rope_refiner/plot_report_figures.py`
- `apply_rope_refinement.py --mode student`
- tests for student training/apply, aggregation, and report figures.

Local verification:

```text
python -m unittest discover -s tests
182 tests OK

python -m pytest tests -q
182 passed, 4 warnings
```

HPC repo was fast-forwarded to `4d84d92`.

Note: importing torch on the login node failed with missing torch package
submodules. Per user instruction, jobs were submitted anyway; if Slurm jobs
fail, debug from job logs rather than blocking on login-node torch import.

## Batch T/E: Student Training And Evaluation

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_student_20260707_135405
```

Teacher:

```text
/data/wentao/ropetrack/runs/rope_p2_train_teacher_20260707_121353/teacher/rope_flex15_gate010
```

Training variants in one GPU job:

- `student_main`: default aug `noise=0.05/dropout=0.1`
- `student_shuffled`: `--shuffle-rope`
- `student_noaug`: `--aug-noise-std 0 --aug-dropout 0`
- `student_seed1`
- `student_seed2`
- `student_rope_loss`: `--rope-loss-weight 0.5`

Evaluation cells:

- `mask70_main`
- `mask70_shuffled`
- `mask70_noaug`
- `mask70_main_noise0p05`
- `finger_end80_main`
- `ho3d_wilor_main`
- `hamer_mask70_main`
- `clean_main`
- `mask70_seed1`
- `mask70_seed2`

Jobs:

| Group | Kind | Job | Dependency |
|---|---|---:|---:|
| T | train GPU | 169878 | - |
| E | apply GPU | 169879 | 169878 |
| E | score CPU | 169880 | 169879 |

## Batch Q: Queue Fillers

Queue root:

```text
/data/wentao/ropetrack/runs/rope_p2_queue_20260707_135405
```

Jobs:

| Group | Content | Job | Dependency |
|---|---|---:|---:|
| Q1 | finger_end80 training hard root + rope labels | 169881 | - |
| Q1 | WiLoR export + winner teacher on finger_end80 training | 169882 | 169881 |
| Q2 | HaMeR mask70 train export + winner teacher | 169883 | - |
| Q3 | noise curve补点 `sigma={0.025,0.075,0.15}` | 169884 | - |
| Q3 | score noise补点 | 169885 | 169884 |
| Q4 | HO3D strong oracle `oracle_tip/flex15` for WiLoR and HaMeR | 169886 | - |
| Q4 | score HO3D strong oracle | 169887 | 169886 |

Q5 qualitative figures were not submitted yet because they depend on E1
student outputs and there is no existing top-improved student/teacher figure
entrypoint. Generate them after `mask70_main` is scored.

## Readout Gates

Student acceptance:

- `mask70_main` should retain at least 80% of the Batch B teacher gain
  (target all-joint delta at least about `-1.34 mm`).
- `mask70_shuffled` should be near zero gain.
- `clean_main` should be close to neutral (`|delta| <= 0.05 mm` target).
- Compare `mask70_main_noise0p05` against the teacher noise cell
  (`noise_std=0.05`, all-joint delta `-1.384 mm`) for possible denoising gain.
