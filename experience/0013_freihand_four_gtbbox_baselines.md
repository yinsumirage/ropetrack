# 0013 FreiHAND Four GT-BBox Baselines

Date: 2026-07-03

## Purpose

Run the full FreiHAND evaluation split with oracle GT bboxes for four
checkpoints:

- HaMeR
- HaMeR w/ AnyHand
- WiLoR
- WiLoR w/ AnyHand

Use the same aligned FreiHAND protocol as the paper table:

```text
PA-MPJPE = xyz_procrustes_al_mean3d * 10
PA-MPVPE = mesh_al_mean3d * 10
F@5      = f_al_score_5
F@15     = f_al_score_15
```

The evaluator stores distances in centimetres, so PA-MPJPE and PA-MPVPE are
multiplied by 10 to report millimetres.

## Jobs

All jobs completed with `ExitCode 0:0`.

| Method | GPU job | GPU time | CPU eval job | CPU time | Run root |
|---|---:|---:|---:|---:|---|
| HaMeR | 162304 | 00:05:42 | 162305 | 00:02:49 | `/data/wentao/ropetrack/runs/freihand_hamer_gtbbox_20260703_024700` |
| HaMeR w/ AnyHand | 162306 | 00:05:40 | 162307 | 00:02:34 | `/data/wentao/ropetrack/runs/freihand_hamer_anyhand_gtbbox_20260703_024700` |
| WiLoR | 162308 | 00:05:53 | 162309 | 00:02:56 | `/data/wentao/ropetrack/runs/freihand_wilor_gtbbox_20260703_024700` |
| WiLoR w/ AnyHand | 162310 | 00:05:48 | 162311 | 00:02:53 | `/data/wentao/ropetrack/runs/freihand_wilor_anyhand_gtbbox_20260703_024700` |

All four runs used 3960 samples and had 0 prediction failures.

## Scores

| Method | PA-MPJPE mm | PA-MPVPE mm | F@5 | F@15 | raw xyz cm | raw mesh cm |
|---|---:|---:|---:|---:|---:|---:|
| HaMeR | 5.577 | 5.778 | 0.779918 | 0.989704 | 558.485 | 558.481 |
| HaMeR w/ AnyHand | 5.178 | 5.367 | 0.806136 | 0.992011 | 567.678 | 567.675 |
| WiLoR | 4.994 | 5.181 | 0.819659 | 0.992201 | 57.310 | 57.310 |
| WiLoR w/ AnyHand | 4.956 | 5.181 | 0.817835 | 0.993083 | 56.109 | 56.111 |

## Paper Table Comparison

Screenshot reference values:

| Method | Paper PA-MPJPE | Ours PA-MPJPE | Paper PA-MPVPE | Ours PA-MPVPE | Paper F@5 | Ours F@5 | Paper F@15 | Ours F@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HaMeR | 6.000 | 5.577 | 5.700 | 5.778 | 0.785 | 0.779918 | 0.990 | 0.989704 |
| HaMeR w/ AnyHand | 5.545 | 5.178 | 5.246 | 5.367 | 0.811 | 0.806136 | 0.993 | 0.992011 |
| WiLoR | 5.500 | 4.994 | 5.100 | 5.181 | 0.825 | 0.819659 | 0.993 | 0.992201 |
| WiLoR w/ AnyHand | 5.394 | 4.956 | 5.046 | 5.181 | 0.827 | 0.817835 | 0.994 | 0.993083 |

Interpretation:

- The reproduced aligned metrics are in the expected range.
- PA-MPJPE is slightly better than the screenshot rows for all four methods.
- PA-MPVPE is close but slightly worse, especially for the AnyHand rows.
- F@5 is lower by about 0.005 to 0.009 absolute; F@15 is essentially matched.
- The ordering is sensible: AnyHand improves HaMeR clearly; WiLoR and
  AnyHand-WiLoR are very close under this oracle-bbox run.

Do not over-interpret small differences. The run uses our `scripts/bench_freihand.py`
oracle-bbox crop/export path and the shared HO3D-style evaluator wrapper. Crop
details, checkpoint config, and exact preprocessing can move FreiHAND F@5 by a
few thousandths.

## Raw Metric Caveat

Raw `xyz_mean3d`, `mesh_mean3d`, `f_score_5`, and `f_score_15` remain diagnostic
only. They are not the values used in the screenshot table.

The WiLoR runs still show the earlier absolute-depth issue around 56 to 57 cm.
The HaMeR runs show an even larger raw camera translation mismatch around 5.6 m.
This does not contradict the aligned table: PA metrics remove global similarity
alignment and are the protocol used for the paper comparison.

Keep FreiHAND on the no-HO3D-flip path. There is still no evidence that a fixed
axis rotation is the right fix for raw absolute metrics.
