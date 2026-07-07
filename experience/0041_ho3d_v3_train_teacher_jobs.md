# 0041 HO3D v3 Train Teacher Jobs

Date: 2026-07-07

## Purpose

After the train-split audit in `0040`, this submits the first HO3D v3 training
teacher pipeline for P2.1 multi-dataset student training. The pipeline uses
stride 4 video subsampling so the HO3D train teacher is roughly balanced with
the FreiHAND train teacher instead of duplicating near-identical adjacent
video frames.

## Code

Committed HO3D v3 train pipeline:

```text
1691a4e Add HO3D v3 train pipeline
```

Included:

- HO3D train sample iterator and projected-joint bbox fallback in
  `ropetrack/datasets/hand_pose.py`;
- HO3D `--split training --stride` support in `scripts/make_hard_images.py`;
- HO3D training rope-label generation in `scripts/make_rope_labels.py`;
- `configs/datasets/ho3d_v3_mask70_train.yaml`;
- regression tests in `tests/test_ho3d_train_split.py`.

Local verification:

```text
python -m unittest discover -s tests
Ran 201 tests
OK
```

HPC repo was fast-forwarded to `1691a4e` via a git bundle because local
`git push` to GitHub did not complete.

## Jobs

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307
```

Data products:

```text
/data/wentao/ropetrack/hard/ho3d_v3/mask70_train
/data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl
```

Teacher output:

```text
/data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/teacher/rope_flex15_gate010
```

| Kind | Job | Dependency | State at submit | Output |
|---|---:|---:|---|---|
| data CPU | 169972 | - | `PD (Priority)` | hard root + stride4 rope labels |
| teacher GPU | 169973 | 169972 | `PD (Dependency)` | WiLoR export + `rope/flex15/gate010` teacher |

CPU job command:

```bash
python scripts/make_hard_images.py --dataset ho3d --split training --stride 4 \
  --input-root /data/wentao/ropetrack/HO3D_v3 \
  --output-root /data/wentao/ropetrack/hard/ho3d_v3/mask70_train \
  --effect mask --severity 0.70 --limit 0

python scripts/make_rope_labels.py --dataset ho3d --split training --stride 4 \
  --input-root /data/wentao/ropetrack/HO3D_v3 \
  --output /data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl \
  --limit 0
```

GPU job command:

```bash
python scripts/eval.py \
  --dataset ho3d_v3_mask70_train \
  --method wilor_original \
  --split training \
  --protocol-check-samples 0 \
  --save-mano-cache \
  --batch-size 64 \
  --num-workers 0 \
  --out-dir /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor

python scripts/rope_refiner/apply_rope_refinement.py \
  --mode optimize \
  --objective rope \
  --action-space flex15 \
  --gate-residual-threshold 0.1 \
  --opt-steps 400 \
  --opt-lr 32 \
  --opt-alpha-l2 0.001 \
  --dataset ho3d \
  --rope-labels /data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl \
  --pred-dir /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor/eval_input \
  --run-meta /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor/run_meta.json \
  --mano-cache /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor/mano_cache.npz \
  --out-dir /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/teacher/rope_flex15_gate010 \
  --device cuda \
  --batch-size 512
```

## Next Check

When `169972` completes, verify:

- hard root has a strided `train.txt`;
- rope label line count matches the strided train list;
- `training_xyz.json` and `hard_manifest.jsonl` line/sample counts match.

When `169973` completes, the teacher can be combined with the FreiHAND train
teacher using:

```bash
python scripts/rope_refiner/train_alpha_student.py \
  --teacher-dir \
  /data/wentao/ropetrack/runs/rope_p2_train_teacher_20260707_121353/teacher/rope_flex15_gate010 \
  /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/teacher/rope_flex15_gate010 \
  --action-space flex15 \
  --out-dir <student_multi>
```
