# Experience Index

## Stage 1: Clean Baseline Reproduction

Scope: repo/HPC setup, data extraction, backend smoke tests, and clean
GT-bbox baseline reproduction for FreiHAND, HO3D v2, and HO3D v3. This stage
ends with the clean baseline report before hard splits or rope labels.

### Setup And Data

| Date | Note | Why read it |
|---|---|---|
| 2026-07-01 | [0000_repo_bootstrap.md](0000_repo_bootstrap.md) | Initial repo boundaries and skipped work. |
| 2026-07-01 | [0001_hpc_env_and_submodules.md](0001_hpc_env_and_submodules.md) | Current HPC paths, environment split, and AnyHand submodule cleanup lessons. |
| 2026-07-02 | [0002_data_layout_and_wilor_smoke.md](0002_data_layout_and_wilor_smoke.md) | Remote data layout, HO3D v2/v3 distinction, and AnyHand-WiLoR smoke result. |
| 2026-07-02 | [0003_data_storage_and_extraction.md](0003_data_storage_and_extraction.md) | Shared model storage, extracted FreiHAND/HO3D_v3 paths, and remaining asset caveat. |
| 2026-07-02 | [0006_hamer_demo_data_download.md](0006_hamer_demo_data_download.md) | Fresh HaMeR demo data download path, contents, and the shared-storage sbatch lesson. |
| 2026-07-03 | [0010_freihand_eval_gt_upload.md](0010_freihand_eval_gt_upload.md) | Uploaded FreiHAND evaluation GT package, merged `evaluation/anno` and `facemap`, and copied the root eval JSON files. |

### Benchmark Scripts And Clean Runs

| Date | Note | Why read it |
|---|---|---|
| 2026-07-02 | [0004_ho3d_v2_wilor_baseline.md](0004_ho3d_v2_wilor_baseline.md) | WiLoR HO3D v2 detector/gt_bbox scores and the joint-order caveat. |
| 2026-07-02 | [0005_ho3d_v3_wilor_jobs.md](0005_ho3d_v3_wilor_jobs.md) | HO3D v3 WiLoR/AnyHand job IDs, run dirs, GT upload caveat, and eval dependencies. |
| 2026-07-02 | [0007_ho3d_gtbbox_batch_export.md](0007_ho3d_gtbbox_batch_export.md) | Cross-image GT bbox batch export path and remote bbox metadata sampling. |
| 2026-07-02 | [0008_ho3d_generic_detector_batch.md](0008_ho3d_generic_detector_batch.md) | Generic HO3D benchmark script, detector batch smoke, and detector-batch parity caveat. |
| 2026-07-02 | [0009_ho3d_hamer_gtbbox_jobs.md](0009_ho3d_hamer_gtbbox_jobs.md) | HaMeR/AnyHand-HaMeR gt-bbox scores on HO3D v2 and v3. |
| 2026-07-03 | [0011_freihand_wilor_gtbbox_eval.md](0011_freihand_wilor_gtbbox_eval.md) | FreiHAND AnyHand-WiLoR gt-bbox smoke, full prediction, CPU eval scores, and joint-order fix. |
| 2026-07-03 | [0012_ho3d_generic_wilor_original_gtbbox_full.md](0012_ho3d_generic_wilor_original_gtbbox_full.md) | Full HO3D v2 GT-bbox run with generic `bench_ho3d.py` and original WiLoR checkpoint. |
| 2026-07-03 | [0013_freihand_four_gtbbox_baselines.md](0013_freihand_four_gtbbox_baselines.md) | Full FreiHAND GT-bbox HaMeR/WiLoR/AnyHand scores and paper-table comparison. |

## Stage 2: Hard Splits And Rope Signals

### Hard Image Smoke

| Date | Note | Why read it |
|---|---|---|
| 2026-07-03 | [0014_stage2_ho3d_v2_hard_mask_smoke.md](0014_stage2_ho3d_v2_hard_mask_smoke.md) | First hard-image generator, HO3D v2 clean vs mask smoke, limited-GT eval fix, and severity lesson. |
| 2026-07-03 | [0015_stage2_ho3d_v2_hard_mask_512.md](0015_stage2_ho3d_v2_hard_mask_512.md) | 512-sample HO3D v2 clean/mask comparison, image mask check, and conclusion that 0.45 is weak while 0.85 is a hard candidate. |
