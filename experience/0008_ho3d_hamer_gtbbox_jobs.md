# 0008 HO3D HaMeR GT BBox Jobs

Date: 2026-07-02

## Purpose

Run batched `gt_bbox` HO3D predictions and parallel eval for:

- Original HaMeR checkpoint.
- AnyHand fine-tuned HaMeR checkpoint.
- HO3D v2 and HO3D v3 evaluation splits.

Detector-mode HaMeR is intentionally deferred until the detector path is
parallelized.

## Command Shape

The jobs used `scripts/bench_ho3d_v2.py` with:

```text
--mode gt_bbox
--backend hamer
--batch-size 128
--num-workers 4
--joint-source mano_vertices
```

Prediction jobs ran on GPU; eval jobs ran with `scripts/eval_ho3d_parallel.py`
on CPU using 16 workers.

## Jobs

| Experiment | Run dir | GPU job | CPU eval job |
|---|---|---:|---:|
| HO3D v2 original HaMeR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v2_hamer_orig_gtbbox_20260702_222653` | 161995 | 161996 |
| HO3D v2 AnyHand HaMeR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v2_anyhand_hamer_gtbbox_20260702_222653` | 161997 | 161998 |
| HO3D v3 original HaMeR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_hamer_orig_gtbbox_20260702_222653` | 161999 | 162000 |
| HO3D v3 AnyHand HaMeR gt_bbox | `/data/wentao/ropetrack/runs/ho3d_v3_anyhand_hamer_gtbbox_20260702_222653` | 162001 | 162002 |

All four prediction jobs completed with `num_failures: 0`.

Runtime:

| Experiment | GPU pred | CPU eval |
|---|---:|---:|
| HO3D v2 original HaMeR | 00:11:24 | 00:04:22 |
| HO3D v2 AnyHand HaMeR | 00:11:24 | 00:04:30 |
| HO3D v3 original HaMeR | 00:14:51 | 00:09:49 |
| HO3D v3 AnyHand HaMeR | 00:13:29 | 00:07:35 |

## Scores

HO3D eval writes mean errors in centimetres.

| Dataset | Checkpoint | PA joint mean | PA joint AUC | ST joint mean | ST joint AUC | PA mesh mean | PA mesh AUC | aligned F@5 | aligned F@15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HO3D v2 | original HaMeR | 0.773478 | 0.845385 | 1.503033 | 0.703901 | 0.793034 | 0.841445 | 0.634948 | 0.979961 |
| HO3D v2 | AnyHand HaMeR | 0.744107 | 0.851269 | 1.503280 | 0.703502 | 0.768454 | 0.846367 | 0.643820 | 0.984041 |
| HO3D v3 | original HaMeR | 0.742375 | 0.851546 | 1.443277 | 0.715428 | 0.692216 | 0.861566 | 0.673497 | 0.979792 |
| HO3D v3 | AnyHand HaMeR | 0.699943 | 0.860021 | 1.421043 | 0.720195 | 0.656574 | 0.868693 | 0.696439 | 0.984741 |

In paper-style millimetres:

```text
HO3D v2 original HaMeR: PA-MPJPE 7.73 mm, PA-MPVPE 7.93 mm
HO3D v2 AnyHand HaMeR: PA-MPJPE 7.44 mm, PA-MPVPE 7.68 mm
HO3D v3 original HaMeR: PA-MPJPE 7.42 mm, PA-MPVPE 6.92 mm
HO3D v3 AnyHand HaMeR: PA-MPJPE 7.00 mm, PA-MPVPE 6.57 mm
```

## Interpretation

AnyHand fine-tuning improves HaMeR on both HO3D v2 and v3 under GT bbox
evaluation. The gain is modest on HO3D v2 and clearer on HO3D v3.

Raw `xyz_mean3d` and `mesh_mean3d` are very large because this export/eval path
is still interpreted through aligned metrics, matching the earlier WiLoR
records. Use PA/ST metrics and aligned F-scores for baseline comparison.
