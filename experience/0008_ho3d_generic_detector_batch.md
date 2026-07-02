# 0008 HO3D Generic Detector Batch

Date: 2026-07-02

## Purpose

Create `scripts/bench_ho3d.py` as the generic successor to
`scripts/bench_ho3d_v2.py`, while keeping the v2 script untouched as the
verified baseline.

The new script supports a generic candidate flow:

```text
sample -> 0/N BBoxItem -> cross-image crop batch -> per-sample selection
```

This handles HO3D `gt_bbox`, detector boxes, and future datasets where one
sample may have multiple hand boxes.

## Local Design

- `BBoxItem` stores `sample_index`, `bbox_index`, box coordinates, hand side,
  score, and source.
- `gt_bbox` mode creates one or more `BBoxItem`s from each sample meta file.
- `detector` mode batches YOLO over images, flattens variable boxes into
  `BBoxItem`s, then reuses the same crop/model batch path.
- Export still writes one HO3D prediction per sample, selecting the highest
  score candidate and zero-filling missing detections.

## Local Verification

```text
$env:PYTHONPATH='src'; python -m unittest discover -s tests
21 tests OK

python -m py_compile scripts/bench_ho3d.py scripts/bench_ho3d_v2.py scripts/eval_ho3d_parallel.py
OK

git diff --check
OK
```

## HPC Smoke

Remote files were copied to `~/project/ropetrack` for smoke tests. The remote
worktree was not committed.

Cancelled job:

```text
162019 rt_ho3d_generic_smoke
```

Reason: it used HO3D v2 with `--limit 64`, but HO3D v2 has no
`evaluation.txt`, so the script first triggered the expensive full
GT-root order inference over all 11524 meta files. The GPU was holding memory
with near-zero utilization, so the job was cancelled.

Successful HO3D v3 smoke:

```text
job: 162041
run: /data/wentao/ropetrack/runs/ho3d_generic_v3_smoke_20260702_225211_162041
limit: 32
gt_bbox: 32 xyz, 32 verts, 0 failures, 32 candidates
detector: 32 xyz, 32 verts, 0 failures, 32 candidates
```

Detector speed comparison on HO3D v3:

```text
job: 162051
run: /data/wentao/ropetrack/runs/ho3d_detector_batch_compare_20260702_225712_162051
limit: 128
old detector serial: 220.26 s, 0 failures
new detector batch_size=16: 54.78 s, 0 failures, 128 candidates
```

The fast detector batch path was about 4x faster on this 128-frame smoke.

Output comparison:

```text
old serial vs new detector_batch_size=16:
xyz max abs diff: 0.004491
verts max abs diff: 0.004496
allclose(atol=1e-5): False
```

Isolation run:

```text
job: 162055
run: /data/wentao/ropetrack/runs/ho3d_detector_batch_isolate_20260702_230611_162055
new detector_batch_size=1: 288.75 s, 0 failures, 128 candidates
old serial vs new detector_batch_size=1 allclose(atol=1e-5): True
detector_batch_size=16 vs 1 allclose(atol=1e-5): False
```

Takeaway: output differences come from YOLO batched detection, not from the
cross-image crop/model batch. The default `--detector-batch-size` is therefore
`1` for semantic compatibility. Use `--detector-batch-size 16` when speed is
more important than exact detector-wrapper parity.
