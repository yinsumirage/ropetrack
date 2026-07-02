# 0007 HO3D GT BBox Batch Export

Date: 2026-07-02

## Purpose

Speed up HO3D `gt_bbox` prediction export by matching the upstream HaMeR eval
shape: build a cross-image dataset, batch hand crops, and call the model once
per batch instead of once per sample.

## Local Change

`scripts/bench_ho3d_v2.py` now routes `--mode gt_bbox` through a cross-image
batch path:

```text
samples -> meta handBoundingBox -> CrossImageGtBBoxDataset -> DataLoader
-> predictor._wilor_model / predictor._hamer_model -> HO3D pred.json
```

The detector path is unchanged and still uses `predictor.predict(image_path)`.

Use `--batch-size 128` for the real H200 GPU run unless a later full-run sweep
shows otherwise:

```bash
python scripts/bench_ho3d_v2.py \
  --ho3d-root /data/wentao/ropetrack/HO3D_v2_eval \
  --out-dir /data/wentao/ropetrack/runs/<run> \
  --mode gt_bbox \
  --backend wilor \
  --batch-size 128 \
  --num-workers 4 \
  --limit 0
```

## Data Check

Remote read-only sampling confirmed that HO3D meta files expose
`handBoundingBox`:

```text
HO3D_v2_eval evaluation meta count: 11524
sampled: AP10/0000, MPM12/0560, SM1/0889 -> has_bbox=True

HO3D_v3 evaluation.txt count: 20137
sampled: SM1/0000, SB11/1322, AP12/1615 -> has_bbox=True
```

Note: `/data/wentao/ropetrack/HO3D_v3/evaluation/*/meta/*.pkl` contains 20428
meta files, but `evaluation.txt` contains 20137 samples. The benchmark export
should keep using `evaluation.txt` order when present.

## Verification

Local checks:

```text
$env:PYTHONPATH='src'; python -m unittest discover -s tests
python -m unittest tests.test_ho3d_v2_bench
python -m py_compile scripts/bench_ho3d_v2.py
```

The real proof is an HPC GPU run comparing a small `gt_bbox --batch-size 1`
export against `--batch-size 32` for identical `pred.json` lengths and close
scores.

## Batch Sweep

Short Slurm sweeps on `server53` used AnyHand WiLoR, HO3D v2 `gt_bbox`, and no
CPU eval.

Job `161951`, 512 samples per batch size:

| Batch size | Wall time | Samples/sec | Failures |
|---:|---:|---:|---:|
| 1 | 202.73 s | 2.53 | 0 |
| 16 | 46.46 s | 11.02 | 0 |
| 32 | 42.39 s | 12.08 | 0 |
| 64 | 40.40 s | 12.67 | 0 |

Job `161959`, 1024 samples per batch size:

| Batch size | Wall time | Samples/sec | Failures |
|---:|---:|---:|---:|
| 64 | 62.26 s | 16.45 | 0 |
| 128 | 47.66 s | 21.49 | 0 |
| 256 | 48.53 s | 21.10 | 0 |

The best tested setting is `--batch-size 128`. Batch 256 did not improve
throughput, so the current bottleneck is likely crop/IO/CPU scheduling rather
than H200 memory. GPU sampling peaked at 100% utilization, about 10.9 GiB memory
used, and about 697 W during the large-batch sweep.

Output consistency:

```text
batch 1 vs 64, 512 samples:
xyz max abs diff ~= 8.26e-06
verts max abs diff ~= 8.26e-06
allclose(atol=1e-5, rtol=1e-5): True

batch 64/128/256, 1024 samples:
xyz max abs diff <= 1.08e-06
verts max abs diff <= 7.16e-07
allclose(atol=1e-5, rtol=1e-5): True
```
