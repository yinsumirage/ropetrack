# 0011 HO3D Generic WiLoR Original GT-BBox Full Run

Date: 2026-07-03

## Purpose

Verify that the new generic `scripts/bench_ho3d.py` reproduces a full HO3D v2
GT-bbox benchmark with the original WiLoR checkpoint.

## Jobs

Run directory:

```text
/data/wentao/ropetrack/runs/ho3d_v2_wilor_original_gtbbox_bench_ho3d_20260703_011850
```

Submitted jobs:

```text
GPU prediction/export: 162206
CPU parallel eval:     162207
```

GPU command used `scripts/bench_ho3d.py` with:

```text
--ho3d-root /data/wentao/ropetrack/HO3D_v2_eval
--mode gt_bbox
--backend wilor
--wilor-ckpt /data/wentao/ropetrack/pretrained_models/wilor_final.ckpt
--batch-size 128
--num-workers 4
--limit 0
```

CPU eval used:

```text
python scripts/eval_ho3d_parallel.py \
  <run>/eval_input \
  <run>/eval_results_parallel_16cpu \
  --version v2 \
  --num-workers 16 \
  --chunksize 16
```

## Result

GPU export completed:

```text
xyz 11524
verts 11524
failures 0
candidates 11524
prediction_path gt_bbox_cross_image_batch
checkpoint /data/wentao/ropetrack/pretrained_models/wilor_final.ckpt
real 957.20s
```

CPU eval completed:

```text
Evaluated 11524 samples with 16 worker(s).
real 253.41s
```

Scores:

```text
xyz_procrustes_al_mean3d: 0.753656
xyz_procrustes_al_auc3d: 0.849367
xyz_scale_trans_al_mean3d: 1.488246
xyz_scale_trans_al_auc3d: 0.706184
mesh_al_mean3d: 0.777823
mesh_al_auc3d: 0.844507
f_al_score_5: 0.640977
f_al_score_15: 0.982330
```

These match the earlier `wilor_final` GT-bbox retry results from
`experience/0004_ho3d_v2_wilor_baseline.md` up to rounding. This confirms the
new generic HO3D script preserves the full GT-bbox benchmark result while using
the batched reconstruction path.

## Notes

- `scripts/bench_ho3d.py` default mode was changed to `gt_bbox` locally before
  this run, matching the official benchmark protocol.
- The initial few minutes showed low GPU usage while HO3D v2 sample order was
  inferred and the Lightning checkpoint was loaded.
- GPU utilization then showed real WiLoR work; memory stayed around 5.8 GiB.
