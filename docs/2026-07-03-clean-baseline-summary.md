# Clean Baseline Summary

Date: 2026-07-03

## Scope

This is the Stage 1 clean baseline report for `ropetrack`.

Included:

- FreiHAND evaluation split, 3960 samples.
- HO3D v2 evaluation split, 11524 samples.
- HO3D v3 evaluation split, 20137 samples.
- GT-bbox runs for HaMeR, HaMeR w/ AnyHand, WiLoR, and WiLoR w/ AnyHand.

Not included yet:

- Hard mask/blur/crop/appearance splits.
- Rope-distance labels or post-optimization.
- Training or fine-tuning.

## Result Tables

FreiHAND uses the paper-style aligned metrics: PA-MPJPE, PA-MPVPE, F@5, and
F@15. Distances are reported in millimetres.

| Method | PA-MPJPE | PA-MPVPE | F@5 | F@15 |
|---|---:|---:|---:|---:|
| HaMeR | 5.577 | 5.778 | 0.779918 | 0.989704 |
| HaMeR w/ AnyHand | 5.178 | 5.367 | 0.806136 | 0.992011 |
| WiLoR | 4.994 | 5.181 | 0.819659 | 0.992201 |
| WiLoR w/ AnyHand | 4.956 | 5.181 | 0.817835 | 0.993083 |

HO3D v2 uses the HO3D paper-style metrics. The evaluator writes mean distances
in centimetres; PA-MPJPE and PA-MPVPE below are converted to millimetres.

| Method | AUCj | PA-MPJPE | AUCv | PA-MPVPE | F@5 | F@15 |
|---|---:|---:|---:|---:|---:|---:|
| HaMeR | 0.845385 | 7.735 | 0.841445 | 7.930 | 0.634948 | 0.979961 |
| HaMeR w/ AnyHand | 0.851269 | 7.441 | 0.846367 | 7.685 | 0.643820 | 0.984041 |
| WiLoR | 0.849367 | 7.537 | 0.844507 | 7.778 | 0.640977 | 0.982330 |
| WiLoR w/ AnyHand | 0.851971 | 7.406 | 0.846304 | 7.688 | 0.645197 | 0.983529 |

HO3D v3 is now a clean reproduced benchmark, not a replacement for HO3D v2.
It is reported separately because split size, images, and protocol files differ.

| Method | AUCj | PA-MPJPE | AUCv | PA-MPVPE | F@5 | F@15 |
|---|---:|---:|---:|---:|---:|---:|
| HaMeR | 0.851546 | 7.424 | 0.861566 | 6.922 | 0.673497 | 0.979792 |
| HaMeR w/ AnyHand | 0.860021 | 6.999 | 0.868693 | 6.566 | 0.696439 | 0.984741 |
| WiLoR | 0.858267 | 7.088 | 0.868113 | 6.596 | 0.691721 | 0.983520 |
| WiLoR w/ AnyHand | 0.861273 | 6.938 | 0.870803 | 6.461 | 0.700943 | 0.983929 |

## What Is Settled

- The basic HPC data/model layout is usable.
- FreiHAND evaluation GT is present and merged.
- HO3D v2 and v3 GT-bbox benchmark runs are reproducible with the current
  scripts.
- The HO3D joint protocol bug was fixed by deriving joints from MANO vertices.
- FreiHAND has its own MANO joint/tip adapter and passes the protocol check.
- AnyHand fine-tuning improves HaMeR consistently across FreiHAND, HO3D v2, and
  HO3D v3.
- WiLoR and WiLoR w/ AnyHand are close under oracle GT-bbox evaluation.

## Caveats

- These are GT-bbox clean baselines. Detector-mode results are useful but not
  the main paper-style table.
- FreiHAND raw camera-space metrics are diagnostic only here; the paper table
  uses aligned metrics.
- HO3D v2 and HO3D v3 should stay separate in reports.
- The current repo has benchmark entrypoints, not a full manifest/prediction
  schema pipeline.

## Source Notes

- `experience/0004_ho3d_v2_wilor_baseline.md`
- `experience/0005_ho3d_v3_wilor_jobs.md`
- `experience/0009_ho3d_hamer_gtbbox_jobs.md`
- `experience/0011_freihand_wilor_gtbbox_eval.md`
- `experience/0012_ho3d_generic_wilor_original_gtbbox_full.md`
- `experience/0013_freihand_four_gtbbox_baselines.md`

## Next Stage

Stage 2 should start with one minimal hard split, one stable backend, and the
same evaluator. Do not add rope labels until the hard split produces a clear,
repeatable drop.
