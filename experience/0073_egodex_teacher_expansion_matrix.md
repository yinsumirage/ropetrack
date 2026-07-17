# EgoDex all-task teacher expansion matrix

Date: 2026-07-18

## Question and frozen scope

This experiment tests whether EgoDex can contribute useful pseudo-teacher data
to the released four-teacher alpha student without replacing trusted MANO
benchmarks. Raw EgoDex remained read-only. The run selected 5,227 episodes
covering all 111 task categories, decoded 87,878 frames at stride 30, and
exported 169,455 aligned hand samples and rope labels. WiLoR inference had zero
failures. The flex15 teacher retained 158,078 rows with native confidence at
least 0.25; a deterministic balanced variant retained 32,560 rows.

This is a bounded expansion, not full-dataset training: it uses 5,227 episodes
from the audited 318,082 clips and does not treat EgoDex joints as MANO ground
truth.

Run root:

```text
/data/wentao/ropetrack/runs/egodex_teacher_matrix_20260717
```

All preparation, base, teacher, train, apply, score, and final-verification jobs
completed with exit `0:0`. The final `SUCCESS` and `HO_WEIGHT_SUCCESS` markers
exist.

## Student matrix

PA joint error is in mm; lower is better. Every row uses the same frozen WiLoR
base and evaluation samples: EgoDex test 162,191, FreiHAND clean 3,960,
FreiHAND mask70 3,960, and HO3D v2 mask70 11,524.

All five train logs passed the zero-alpha validation baseline. The full and
capped mixtures used 248,931/27,659 and 135,965/15,107 train/validation rows;
the HO3D 2x and 4x follow-ups used 267,680/29,742 and 305,177/33,909.

| Student | EgoDex test | FreiHAND clean | FreiHAND mask70 | HO3D v2 mask70 |
|---|---:|---:|---:|---:|
| WiLoR base | 11.168 | 4.956 | 10.068 | 9.436 |
| EgoDex-only, 111 tasks | 10.868 | **4.868** | 8.619 | 8.565 |
| Four teachers + full EgoDex | **10.863** | 4.939 | **8.411** | **8.522** |
| Four teachers + capped EgoDex | 10.877 | 4.983 | 8.422 | 8.524 |
| Four-teacher release | 11.552 | 4.997 | 8.432 | **8.465** |

The full EgoDex mixture is the only new mixed student that improves all four
protocols versus the matched WiLoR bases. Its signed deltas are -0.305 mm on
EgoDex, -0.017 mm on FreiHAND clean, -1.656 mm on FreiHAND mask70, and
-0.914 mm on HO3D mask70. It also beats the release student on EgoDex,
FreiHAND clean, and FreiHAND mask70, but remains 0.057 mm worse on HO3D.

Using all 158,078 filtered EgoDex rows is consistently better than capping them
at 32,560. In this bounded selection, EgoDex teacher volume does not drown the
older teachers. The EgoDex-only student also transfers positively, confirming
that the signal is not solely EgoDex annotation bias, but it is weaker than the
mixed student on both masked MANO benchmarks.

## HO3D teacher-weight follow-up

The smallest follow-up duplicated the existing HO3D teacher rows while keeping
every other input fixed. This tested whether the remaining 0.057 mm release gap
was merely a sampling-weight issue.

| Student | EgoDex test | FreiHAND clean | FreiHAND mask70 | HO3D v2 mask70 |
|---|---:|---:|---:|---:|
| Full EgoDex mixture | **10.863** | **4.939** | **8.411** | **8.522** |
| HO3D teacher 2x | 10.872 | 4.967 | 8.416 | 8.541 |
| HO3D teacher 4x | 10.899 | 4.974 | 8.413 | 8.688 |

Both weighted variants are worse than the unweighted full mixture on all four
protocols. Relative to it, HO3D 2x regresses by +0.019 mm on HO3D and HO3D 4x
regresses by +0.166 mm. The clean FreiHAND preservation gate also flips from a
-0.017 mm improvement over base to +0.011/+0.017 mm regressions. More HO3D
teacher repetition is therefore rejected.

The masked sliced results agree with the aggregate scores. For HO3D, the
all-joint deltas versus WiLoR weaken from -0.914 mm to -0.895 and -0.748 mm;
the occluded-tip deltas remain helpful but do not rescue the aggregate result.

## Decision

- **Keep as an experimental candidate:** four teachers plus the full 158,078-row
  confidence-filtered, all-task EgoDex teacher. It improves three of four
  protocols relative to the release and all four relative to WiLoR.
- **Do not replace the release checkpoint:** the gain over release is only
  0.021 mm on FreiHAND mask70, while HO3D is 0.057 mm worse. This is a useful
  data-expansion result, not a robust new release win.
- **Stop teacher repetition and full 318k-clip scaling:** simple HO3D weighting
  moves in the wrong direction, and the bounded 111-task selection already
  answers whether EgoDex can add transferable signal.

The smallest justified next experiment is no additional EgoDex scaling. If a
new model is needed, use validation-based teacher sampling or loss weighting
rather than duplicate rows, and require improvement over the release on both
FreiHAND and HO3D before promotion.
