# 0043 P2 Multi-Teacher Student Jobs

Date: 2026-07-07

## Purpose

After the HO3D v3 stride4 train teacher finished, this starts the multi-teacher
student experiment. The goal is to test whether a single student trained from
four teachers improves cross-dataset/backend robustness without losing the
strong FreiHAND mask70 result.

## HO3D v3 Teacher Health Check

Job `169973` completed successfully:

```text
169973 h3v3_s4_teach COMPLETED 0:0 00:38:46
```

Teacher:

```text
/data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/teacher/rope_flex15_gate010
```

Health check against the FreiHAND train teacher baseline
(`closure=0.504`, `gated=0.288`, `alpha=0.053`):

| Teacher | Samples | Closure | Gated fingers | Alpha mean abs |
|---|---:|---:|---:|---:|
| FreiHAND mask70 train | 32560 | 0.503591 | 0.287918 | 0.052586 |
| HO3D v3 stride4 train | 20832 | 0.581811 | 0.281740 | 0.042844 |

This passes the planned gate: closure is well above 0.3 and gated fraction is
far below 0.7. No bbox visualization stop was needed.

## Submitted Jobs

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104
```

Teachers:

```text
/data/wentao/ropetrack/runs/rope_p2_train_teacher_20260707_121353/teacher/rope_flex15_gate010
/data/wentao/ropetrack/runs/rope_p2_queue_20260707_135405/q1_finger_end80_train/teacher
/data/wentao/ropetrack/runs/rope_p2_queue_20260707_135405/q2_hamer_mask70_train/teacher
/data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/teacher/rope_flex15_gate010
```

| Kind | Job | Dependency | Output |
|---|---:|---:|---|
| train GPU | 170175 | - | `students/student_multi`, `students/student_multi_shuffled` |
| apply GPU | 170176 | 170175 | 8 eval cells |
| score CPU | 170177 | 170176 | `tables/runs_summary.*` |

Evaluation cells:

- `multi_mask70`
- `multi_finger_end80`
- `multi_ho3d_wilor`
- `multi_hamer_mask70`
- `multi_clean`
- `multi_mask70_noise0p05`
- `multi_shuffled_mask70`
- `noaug_mask70_noise0p05`

Interpretation gate once results land:

- if `multi_ho3d_wilor` keeps at least about -1.0 mm gain and FreiHAND mask70
  does not drop by more than 0.05 mm versus `student_main`, multi-teacher can
  enter the report as the stronger deployment student;
- otherwise keep the single-teacher main student as the headline and use
  multi-teacher as next-plan material.
