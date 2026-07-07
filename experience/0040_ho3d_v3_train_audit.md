# 0040 HO3D v3 Train Split Audit

Date: 2026-07-07

## Purpose

GPU P2 jobs were queued, so this used the CPU queue to audit HO3D v3 train
format assumptions before adding the training-split pipeline. The main question
was whether the train metadata can support hard-image generation, rope labels,
GT-bbox export, and later multi-dataset student training without guessing.

## Code

Committed multi-teacher student training and the audit script:

```text
2964618 Add multi-teacher student training and HO3D audit
```

Local verification before submission:

```text
python -m unittest discover -s tests
189 tests OK

python -m pytest tests -q
189 passed, 4 warnings
```

HPC repo was fast-forwarded to `2964618`.

## CPU Audit Job

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_audit_20260707_142013
```

Command:

```bash
python scripts/audit_ho3d_train_split.py \
  --input-root /data/wentao/ropetrack/HO3D_v3 \
  --sample-count 300 \
  --output /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_audit_20260707_142013/ho3d_v3_train_audit.json
```

Job:

| Kind | Job | State | Elapsed |
|---|---:|---|---:|
| CPU audit | 169915 | COMPLETED `0:0` | 00:00:25 |

## Findings

Output:

```text
/data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_audit_20260707_142013/ho3d_v3_train_audit.json
```

Key fields:

| Field | Value |
|---|---:|
| `num_split_ids` | 83,325 |
| `num_sequences` | 55 |
| `frames_per_sequence_min_max` | 764 / 2509 |
| `num_sampled` | 300 |
| `num_metas_audited` | 300 |
| image extensions | `.jpg`: 300 |
| `frac_with_handBoundingBox` | 0.0 |
| `shape_failures` | none |
| `nan_annotation_counts` | none |

The single observed meta schema contains:

```text
camIDList, camMat, handBeta, handJoints3D, handPose, handTrans,
handVertContact, handVertDist, handVertIntersec, handVertObjSurfProj,
objCorners3D, objCorners3DRest, objLabel, objName, objRot, objTrans
```

## Pipeline Consequence

HO3D v3 train is usable for the planned P2.1 pipeline, but it does not contain
`handBoundingBox`. The training adapter should therefore:

- read `train.txt` entries as `SEQ/frame`;
- resolve RGB images as `.jpg`;
- use `handJoints3D [21,3]`, `handPose [48]`, and `camMat [3,3]`;
- generate GT bboxes by projecting valid 3D joints when no `handBoundingBox`
  exists;
- support a stride sampler, likely stride 3, to reduce the 83,325 video frames
  to roughly the FreiHAND train-teacher scale.

This makes the next implementation branch clear: extend the dataset adapter and
hard/rope/export scripts around a projected-joint bbox fallback rather than a
train-meta bbox path.
