# 0045 P2 Multi-v2 Jobs

Date: 2026-07-07

## Purpose

Submit the follow-up multi-teacher experiment requested after `0044`: add a
HO3D v3 stride4 `finger_end80` training teacher, add a HO3D v2 `finger_end80`
eval/export/teacher cell, then train a five-teacher student and evaluate the
full matrix including HO3D finger-end transfer.

## Code

Committed the training dataset config:

```text
452510f Add HO3D v3 finger-end train config
```

Added:

```text
configs/datasets/ho3d_v3_finger_end80_train.yaml
```

Light local check:

```text
python -c "from ropetrack.eval.config import build_run_args; ..."
ho3d_v3_finger_end80_train ho3d ... training gt_bbox mano_vertices
```

## Jobs

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_multi_v2_20260707_190953
```

Jobs:

| Stage | Job | Dependency | Output |
|---|---:|---:|---|
| R1 data CPU | 170342 | - | `/data/wentao/ropetrack/hard/ho3d_v3/finger_end80_train` |
| R1 teacher GPU | 170343 | 170342 | `r1_ho3d_v3_finger_train/teacher/rope_flex15_gate010` |
| R2 eval+teacher GPU | 170344 | - | `r2_ho3d_v2_finger_eval` |
| R3 train GPU | 170345 | 170343 | `students/student_multi_v2`, `students/student_multi_v2_shuffled` |
| R3 eval GPU | 170346 | 170345, 170344 | eight student eval cells |
| R3 score CPU | 170347 | 170346 | `tables/runs_summary.*` |

At submission:

```text
170342 PD (Priority)
170344 PD (Priority)
170343/170345/170346/170347 PD (Dependency)
```

## Matrix

R1:

- Generate HO3D v3 train `finger_end80` hard root with `--stride 4`.
- Reuse `/data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl`
  because rope labels are independent of the hard-image perturbation.
- Export WiLoR with MANO cache.
- Generate winner teacher: `rope/flex15/gate010`, 400 steps, lr 32.

R2:

- Re-export HO3D v2 `finger_end80` eval with MANO cache.
- Generate winner teacher for the eval cell, so the student can be compared
  against the test-time teacher on the same split.

R3 five-teacher student:

- FreiHAND mask70 train teacher.
- FreiHAND finger_end80 train teacher.
- HaMeR/FreiHAND mask70 train teacher.
- HO3D v3 mask70 stride4 train teacher.
- HO3D v3 finger_end80 stride4 train teacher.

R3 eval cells:

- `v2_mask70`
- `v2_freihand_finger_end80`
- `v2_ho3d_mask70`
- `v2_ho3d_finger_end80`
- `v2_hamer_mask70`
- `v2_clean`
- `v2_mask70_noise0p05`
- `v2_shuffled_mask70`

## Readout

Expected use:

- If `v2_ho3d_finger_end80` improves clearly over the current multi student
  on the new HO3D finger-end eval cell and FreiHAND mask70 does not drop by
  more than 0.05 mm, multi-v2 can be used as the strongest generalist student.
- Otherwise, keep `0044` multi-teacher as the generalization result and leave
  HO3D finger-end training as next-plan evidence.
