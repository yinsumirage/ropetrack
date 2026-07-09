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
| 2026-07-03 | [0017_hand_predictor_migration.md](0017_hand_predictor_migration.md) | Conservative first step moving predictor ownership into `ropetrack/backends` while keeping AnyHand for parity review. |
| 2026-07-03 | [0018_eval_protocol_adapter.md](0018_eval_protocol_adapter.md) | Unified config-driven eval entrypoint and dataset adapters replacing the old bench scripts. |
| 2026-07-04 | [0019_hamer_tip_policy_and_protocol_check.md](0019_hamer_tip_policy_and_protocol_check.md) | HaMeR-native HO3D fingertip vertices and the now-active protocol check for sample order. |

## Stage 2: Hard Splits And Rope Diagnostics

| Date | Note | Why read it |
|---|---|---|
| 2026-07-03 | [0014_stage2_ho3d_v2_hard_mask_smoke.md](0014_stage2_ho3d_v2_hard_mask_smoke.md) | First hard-image generator, HO3D v2 clean vs mask smoke, limited-GT eval fix, and severity lesson. |
| 2026-07-03 | [0015_stage2_ho3d_v2_hard_mask_512.md](0015_stage2_ho3d_v2_hard_mask_512.md) | 512-sample HO3D v2 clean/mask comparison, image mask check, and conclusion that 0.45 is weak while 0.85 is a hard candidate. |
| 2026-07-03 | [0016_stage2_ho3d_v2_fingertip_fix_512.md](0016_stage2_ho3d_v2_fingertip_fix_512.md) | Corrected 512-sample fingertip square/blur comparison after fixing HO3D projection; square is stronger, both are milder than mask0.85. |
| 2026-07-04 | [0020_stage2_hard_original_eval.md](0020_stage2_hard_original_eval.md) | Full HO3D v2/FreiHAND mask70, tip_square80, and finger_end80 hard roots plus original WiLoR/HaMeR GT-bbox scores. |
| 2026-07-05 | [0021_rope_labels_and_diagnostic_tools.md](0021_rope_labels_and_diagnostic_tools.md) | Rope JSONL label generator, visualization hook, and pred-vs-label rope diagnostic scorer before training. |
| 2026-07-05 | [0022_rope_phase12_hpc_run.md](0022_rope_phase12_hpc_run.md) | Full HPC CPU run for rope labels, visualization samples, and clean/hard rope diagnostic scores. |
| 2026-07-05 | [0023_rope_diagnostic_reliability_run.md](0023_rope_diagnostic_reliability_run.md) | Report-oriented rope reliability analysis with hard-clean deltas, per-finger/bin tables, and scatter/worst-case figures. |

## Stage 3: Rope Correction — P0/P1 Optimization (winner in RELEASE.md)

### Learned-Refiner Probes (negative results that shaped the design)

| Date | Note | Why read it |
|---|---|---|
| 2026-07-05 | [0024_rope_refiner_train_smoke.md](0024_rope_refiner_train_smoke.md) | First cached rope refiner training smoke on 512 FreiHAND hard-training samples, including WiLoR MANO cache export and cache-level pose L1 result. |
| 2026-07-05 | [0025_rope_refiner_full_training_runs.md](0025_rope_refiner_full_training_runs.md) | Full FreiHAND mask70 hard-training cached refiner runs for WiLoR and HaMeR, plus the default-limit fix from 20 to full dataset. |
| 2026-07-05 | [0026_rope_refiner_hard_eval_probe.md](0026_rope_refiner_hard_eval_probe.md) | The canonical 45-dim MLP negative result: fits train cache, worsens held-out eval (observability principle). Code removed in the consolidation pass. |

### Test-Time Optimization: Probes, P0 Diagnostics, P1 Winner

| Date | Note | Why read it |
|---|---|---|
| 2026-07-05 | [0027_rope_optimization_probe.md](0027_rope_optimization_probe.md) | Non-learning per-sample rope optimization probe with five finger curl scalars, showing small aligned-metric gains on FreiHAND mask70. |
| 2026-07-05 | [0028_rope_optimization_cross_split_and_ho3d.md](0028_rope_optimization_cross_split_and_ho3d.md) | Cross-split rope optimization results; the HO3D rows predate the 0029 joint-order fix and are superseded by 0038. |
| 2026-07-07 | [0029_p0_oracle_slice_deadzone_tooling.md](0029_p0_oracle_slice_deadzone_tooling.md) | P0 oracle/action-space/slice/deadzone tooling, plus the HO3D wrapper joint-order fix that invalidates the 0028 HO3D optimize row. |
| 2026-07-07 | [0030_p0_wilor_freihand_jobs.md](0030_p0_wilor_freihand_jobs.md) | Submitted WiLoR FreiHAND P0 rope/oracle action-space matrix jobs for mask70 and finger_end80. |
| 2026-07-07 | [0031_p0_ho3d_v2_mask70_jobs.md](0031_p0_ho3d_v2_mask70_jobs.md) | Submitted fixed-code HO3D v2 mask70 P0 rope/oracle action-space jobs for WiLoR and HaMeR. |
| 2026-07-07 | [0032_p0_wilor_freihand_results.md](0032_p0_wilor_freihand_results.md) | FreiHAND P0 results: oracle ceiling is large, flex15 has oracle potential, but rope/mult5 remains the best current teacher. |
| 2026-07-07 | [0033_p1_mult5_convergence_jobs.md](0033_p1_mult5_convergence_jobs.md) | Submitted zero-code rope/mult5 convergence scan to test loss-scale under-optimization before gating conclusions. |
| 2026-07-07 | [0034_p1_batch_a_flex5_gate_jobs.md](0034_p1_batch_a_flex5_gate_jobs.md) | Submitted same-recipe Batch A cells for flex5 and residual gating before the stronger P1-0 recipe is selected. |
| 2026-07-07 | [0035_p1_batch_b_strong_jobs.md](0035_p1_batch_b_strong_jobs.md) | Submitted strong-recipe Batch B after P1-0 selected lr32/400-step optimization and Batch A passed the gate threshold sanity check. |
| 2026-07-07 | [0036_p1_batch_b_strong_results.md](0036_p1_batch_b_strong_results.md) | Batch B results: strong rope optimization plus gating gives large gains, with `flex15+gate010` beating flex5 and mult5 — the frozen winner recipe. |
| 2026-07-07 | [0037_p1_guardrails_and_teacher_jobs.md](0037_p1_guardrails_and_teacher_jobs.md) | Submitted guardrail jobs for noise, strong oracle, clean split, HaMeR backend, HO3D winner, and FreiHAND train-teacher generation. |
| 2026-07-07 | [0038_p1_guardrails_and_teacher_results.md](0038_p1_guardrails_and_teacher_results.md) | Guardrail results: moderate rope noise passes, strong oracle restores the ceiling, HO3D transfers, and the train teacher is ready. |

## Stage 4: P2 Distillation, Multi-Dataset Teachers, P3 Probes

| Date | Note | Why read it |
|---|---|---|
| 2026-07-07 | [0039_p2_student_training_jobs.md](0039_p2_student_training_jobs.md) | Submitted P2 alpha-student training/eval jobs plus queue fillers for multi-disturbance teachers, noise curve points, and HO3D strong oracle. |
| 2026-07-07 | [0040_ho3d_v3_train_audit.md](0040_ho3d_v3_train_audit.md) | CPU audit of HO3D v3 train format: 83,325 jpg frames, valid hand joints/pose/camera metadata, and no train handBoundingBox, so use projected-joint bbox fallback. |
| 2026-07-07 | [0041_ho3d_v3_train_teacher_jobs.md](0041_ho3d_v3_train_teacher_jobs.md) | Submitted stride-4 HO3D v3 training hard-root/rope-label CPU job and dependent WiLoR winner-teacher GPU job for later multi-dataset student training. |
| 2026-07-07 | [0042_p2_student_results_and_report_tables.md](0042_p2_student_results_and_report_tables.md) | P2 alpha-student results and generated report tables/figures: student recovers nearly all teacher gain, shuffle control collapses, noise curve and HO3D oracle are summarized. |
| 2026-07-07 | [0043_p2_multi_teacher_jobs.md](0043_p2_multi_teacher_jobs.md) | HO3D v3 stride4 train teacher passed health checks, then submitted the four-teacher student, shuffle control, noaug noise probe, and eval/score dependency jobs. |
| 2026-07-07 | [0044_p2_multi_teacher_results.md](0044_p2_multi_teacher_results.md) | The RELEASE MODEL results: four-teacher student preserves FreiHAND, improves HO3D/HaMeR, shuffle collapses, augmented beats noaug under noise. |
| 2026-07-07 | [0045_p2_multi_v2_jobs.md](0045_p2_multi_v2_jobs.md) | Submitted HO3D v3 finger_end80 train teacher, HO3D v2 finger_end80 eval teacher, and five-teacher multi-v2 student/eval/score dependency jobs. |
| 2026-07-07 | [0046_p3_feature_cache_jobs.md](0046_p3_feature_cache_jobs.md) | Submitted P3 frozen WiLoR backbone feature-cache jobs for FreiHAND mask70 eval and train splits after adding the extraction script and tests. |
| 2026-07-07 | [0047_p2_multi_v2_and_p3_feature_results.md](0047_p2_multi_v2_and_p3_feature_results.md) | Multi-v2 negative ablation: an under-converged teacher in an overlapping input region is targeted label noise; the four-teacher model stays release. |
| 2026-07-07 | [0048_p3_v0_and_report_close_jobs.md](0048_p3_v0_and_report_close_jobs.md) | Submitted report-close jobs for release HO3D finger_end80 and latency, plus the first P3 v0 image-feature head train/eval batch. |
| 2026-07-07 | [0049_p3_v0_and_report_close_results.md](0049_p3_v0_and_report_close_results.md) | Report-close: release beats multi-v2 on HO3D finger_end80, latency recorded, and P3 pooled-feature v0 is a clean negative result (next: token/localized features). |
| 2026-07-08 | [0050_remote_release_asset_hygiene.md](0050_remote_release_asset_hygiene.md) | Created the durable P2 release checkpoint copy, data-root README, disk accounting, rope-label audit, and git-tracked report figures. |
| 2026-07-08 | [0051_final_real_data_smoke.md](0051_final_real_data_smoke.md) | Final real-data smoke after consolidation: WiLoR export, release student regression, default optimize mode, hard/rope generators, analysis tools, and loud run-meta failure all verified on HPC. |
| 2026-07-09 | [0052_e1_e2_calibration_scissors.md](0052_e1_e2_calibration_scissors.md) | E1 bias/scale tolerance and E2 pose45 scissors: 5-rope remains the information bottleneck, release student passes the bias_std=0.05 retention gate, and no bias-aug retrain is required before a controlled hardware mini-demo. |
| 2026-07-10 | [0053_temporal_refiner_jobs.md](0053_temporal_refiner_jobs.md) | Verified 20,832-row HO3D v3 stride-4 asset alignment, then submitted gated/ungated oracle-chain pose45 teachers and the CPU-to-GPU-to-CPU eval asset dependency chain. |
