# 0005 HO3D v3 WiLoR Jobs

Date: 2026-07-02

## Purpose

Run HO3D v3 evaluation predictions for:

- AnyHand WiLoR checkpoint, detector bbox.
- AnyHand WiLoR checkpoint, GT bbox.
- Original WiLoR checkpoint, detector bbox.
- Original WiLoR checkpoint, GT bbox.

## Data Notes

HO3D v3 root:

```text
/data/wentao/ropetrack/HO3D_v3
```

The evaluation split has 20137 samples and uses `.jpg` images. The benchmark
export script was updated to resolve `.png`, `.jpg`, or `.jpeg`.

The GT JSON files must be complete before dependent CPU eval jobs start:

```text
/data/wentao/ropetrack/HO3D_v3/evaluation_xyz.json
/data/wentao/ropetrack/HO3D_v3/evaluation_verts.json
```

The CPU eval scripts validate both JSON lengths against 20137 and fail fast if
the upload is incomplete.

## Jobs

Submitted with GPU time limit shortened to 2 hours.

| Experiment | Run dir | GPU job | CPU eval job |
|---|---|---:|---:|
| AnyHand WiLoR detector | `/data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_detector_20260702_161953` | 161779 | 161780 |
| AnyHand WiLoR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_anyhand_wilor_gtbbox_20260702_161953` | 161781 | 161782 |
| WiLoR final detector | `/data/wentao/ropetrack/runs/ho3d_v3_wilor_final_detector_20260702_161953` | 161783 | 161784 |
| WiLoR final gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_wilor_final_gtbbox_20260702_161953` | 161785 | 161786 |

Each CPU eval job has an `afterok` dependency on its corresponding GPU job and
runs:

```bash
python scripts/eval_ho3d_parallel.py \
  <run>/eval_input \
  <run>/eval_results_parallel_16cpu \
  --version v3 \
  --num-workers 16 \
  --chunksize 16
```
