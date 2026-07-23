# Experience Index

The numbered notes remain a single chronological evidence chain. Use this topic
map for navigation; do not move historical files into category folders.

## Browse By Topic

| Topic | Start here | Main evidence |
|---|---|---|
| Environment, data layout, baseline backends | [Stage 1](#stage-1-clean-baseline-reproduction) | `0000-0013`, `0017-0019`, `0050-0051` |
| Hard splits and rope signal diagnostics | [Stage 2](#stage-2-hard-splits-and-rope-diagnostics) | `0014-0023` |
| P0-P2 refiner and frozen release | [RELEASE.md](../RELEASE.md) | `0024-0052` |
| Temporal/state experiments | [0055](0055_dense_history_utility.md), [0056](0056_temporal_oracle_state.md), [0074](0074_hot3d_natural_visibility_and_arctic_ceiling.md), [0080](0080_code_branch_artifact_temporal_audit.md) | Dense history and generic natural-visibility gates are stopped |
| Dataset acquisition, integrity, and protocols | [0060](0060_egodex_export_and_adapter.md), [0069](0069_hot3d_v4_integrity_audit.md), [0079](0079_normal_joint_no_leak_final.md), [0081](0081_dexycb_s1_first_round.md), [0083 EgoVerse](0083_egoverse_capacity_and_parts.md), [0083 InterHand](0083_interhand26m_oneview_first_round.md) | EgoDex, ARCTIC, HOT3D, Ego-Exo4D, HO3D, DexYCB, EgoVerse, InterHand |
| DirectPose architecture, capacity, and cross-dataset validation | [0075](0075_arctic_direct_gt_rgb_rope.md) through [0089](0089_direct_pose_observability_and_posterior_gate.md) | Frozen-token local-pose line, robustness, LoRA stop, gradient/product safety, observability, and conditional-posterior gate |
| Repository structure and reproducibility | [0085](0085_repository_package_and_script_consolidation.md) | Dataset contract, package ownership, script categories, and local/remote parity checks |

For current decisions rather than historical order, read
[`docs/current-code-and-artifact-map.md`](../docs/current-code-and-artifact-map.md).

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
| 2026-07-10 | [0053_temporal_refiner_jobs.md](0053_temporal_refiner_jobs.md) | Temporal teacher submissions and verified pose45 upper bounds, the flat HO3D eval-meta protocol fix, and pinned CPU-gated GPU retry jobs. |
| 2026-07-10 | [0054_temporal_stride4_results_and_past_v2_jobs.md](0054_temporal_stride4_results_and_past_v2_jobs.md) | Corrected stride-4 accuracy and motion results, strict-past V2 screening jobs, and dense episode export dependencies. |
| 2026-07-11 | [0055_dense_history_utility.md](0055_dense_history_utility.md) | Mixed clean-mask-recovery HO3D v3 episodes show that K16/K96 sensor-only history does not beat matched dense K1; shuffled and no-history controls close the claim. |
| 2026-07-15 | [0056_temporal_oracle_state.md](0056_temporal_oracle_state.md) | Clean-prefix oracles show that explicit last-trusted visual state plus current rope/K1 beats dense K1 through 60 masked frames; velocity and fixed-shape alternatives do not explain the gain. |
| 2026-07-15 | [0057_temporal_state_followups.md](0057_temporal_state_followups.md) | Fresh-state and false-update controls identify strict state-management requirements; a pooled-image linear gate recovers the phase-oracle gain causally, while cached-only visibility and rope-residual arbitration fail. |
| 2026-07-15 | [0058_temporal_visibility_transfer.md](0058_temporal_visibility_transfer.md) | The frozen image-linear gate transfers without retraining to finger-end and crop episodes, preserving >1.2 mm masked gains through frame 60 with positive dynamics and rope-shuffle controls; light-mask, tip-square, and blur failures bound deployment claims. |
| 2026-07-16 | [0059_temporal_visibility_mixed_and_lifetime.md](0059_temporal_visibility_mixed_and_lifetime.md) | Mixed visibility training learns blur but regresses prior domains; trusted state is robust through about 60 masked frames, crosses over near 120, and is significantly worse by 240, while K1 retains a fingertip advantage over framewise flex15. |
| 2026-07-16 | [0060_egodex_export_and_adapter.md](0060_egodex_export_and_adapter.md) | EgoDex MP4/HDF5 schema, joint-first export contract, no-native-MANO boundary, dataset/eval/rope integration, and real CPU/GPU smoke. |
| 2026-07-16 | [0061_egodex_full_eval_jobs.md](0061_egodex_full_eval_jobs.md) | Submitted all-episode stride-10 EgoDex export, WiLoR base, release student, noise/dropout, rope-shuffle, 400-step teacher, oracle, and joint-only score dependency chain. |
| 2026-07-17 | [0062_egodex_quality_audit.md](0062_egodex_quality_audit.md) | Full test confidence/geometry/visual audit: 9.33% rows lack native confidence, low-confidence labels visibly fail, ARKit link lengths remain stable, and the release student over-corrects better-confidence rows. |
| 2026-07-17 | [0063_egodex_public_derivatives_and_training_value.md](0063_egodex_public_derivatives_and_training_value.md) | Public-derivative audit: BeingBeyond released a 52 GB MANO-formatted EgoDex preview but not validated cleaned GT; H-RDT uses raw wrist/tip trajectories, and a confidence-aware Part 1 pilot gates any full-data training. |
| 2026-07-17 | [0064_arctic_p2_protocol_and_smoke.md](0064_arctic_p2_protocol_and_smoke.md) | Frozen ARCTIC P2-val split/alignment, native MANO joint/mesh protocol, left-hand batch fix, cross-sequence overlays, and WiLoR/release smoke before the full run. |
| 2026-07-17 | [0065_arctic_p2_val_full_results.md](0065_arctic_p2_val_full_results.md) | Full 38,921-sample ARCTIC P2-val WiLoR/release result: zero failures, mixed sub-mm Flex15 geometry deltas despite 34.9% rope closure, and a separate positive training-suitability judgment. |
| 2026-07-17 | [0066_egodex_full_dataset_audit.md](0066_egodex_full_dataset_audit.md) | Full test+five-part inventory: 318,082 paired clips, 79.1M frames/732.5h/1.862TB, 111 task-partitioned categories, 4,992 sessions, complete integrity, heterogeneous confidence, and the gated Part 1 experiment. |
| 2026-07-17 | [0067_egodex_part1_confidence_student.md](0067_egodex_part1_confidence_student.md) | Part 1 task/session-balanced pilot: EgoDex-trained students improve the full test and unseen tasks, confidence filtering is only a marginal gain, and trusted FreiHAND/HO3D transfer is required before scaling. |
| 2026-07-17 | [0068_egodex_student_freihand_ho3d_transfer.md](0068_egodex_student_freihand_ho3d_transfer.md) | Frozen EgoDex-only student transfers positively to FreiHAND and HO3D mask70, retaining 85-88% of the four-teacher release gain and justifying a balanced all-task expansion rather than full-data training. |
| 2026-07-18 | [0073_egodex_teacher_expansion_matrix.md](0073_egodex_teacher_expansion_matrix.md) | All-task EgoDex teacher expansion improves four matched WiLoR protocols and three release comparisons, but HO3D teacher repetition regresses every cell; keep the full mixture as an experimental candidate, not a release replacement. |
| 2026-07-17 | [0069_hot3d_v4_integrity_audit.md](0069_hot3d_v4_integrity_audit.md) | Exact v4 audit plus an independently owned copy at `/data/wentao/datasets/hot3d`: all 424 VRS files freshly match official size/SHA-1, all 294 public-GT sequences parse, and only three valid optional eye-gaze warnings remain. |
| 2026-07-17 | [0070_egoexo4d_v2_shared_integrity_audit.md](0070_egoexo4d_v2_shared_integrity_audit.md) | Cached official V2 manifests prove the shared default Ego-Exo4D download stopped partway: annotations/metadata are complete, takes are 28.27% by bytes, hand RGB covers 226/351 annotated takes, and hand VRS covers 0/351. |
| 2026-07-18 | [0071_arctic_shared_copy_integrity.md](0071_arctic_shared_copy_integrity.md) | Final shared-data gate: 339 official sequences and 2.19M images copied with an empty diff, 246 official archives matched 1.60M extracted JPEG CRCs, and two isolated missing-view frames are disclosed. |
| 2026-07-18 | [0072_hot3d_aria_protocol_and_smoke.md](0072_hot3d_aria_protocol_and_smoke.md) | Aria public-GT raw-VRS/MPS/fisheye/MANO protocol, verified overlays, and the first zero-failure WiLoR/release smoke; the gated student is effectively identity on this easy 16-sample slice. |
| 2026-07-18 | [0073_arctic_train_integration_and_checkpoint_smoke.md](0073_arctic_train_integration_and_checkpoint_smoke.md) | P2 train is integrated as 310,078 manifest/keypoint/rope rows; visible-hand overlays, MANO decoding, WiLoR and frozen Flex15 smokes pass, while the 38,921-sample val result remains the benchmark conclusion. |
| 2026-07-18 | [0074_hot3d_natural_visibility_and_arctic_ceiling.md](0074_hot3d_natural_visibility_and_arctic_ceiling.md) | Expanded HOT3D rejects visibility/state gates, while full EgoDex supervision improves the same Flex15 student on HOT3D and ARCTIC and beats capped data; ARCTIC rope-only scaling remains rejected, and PA/root-relative claims are separated. |
| 2026-07-18 | [0075_arctic_direct_gt_rgb_rope.md](0075_arctic_direct_gt_rgb_rope.md) | Direct ARCTIC GT plus localized WiLoR tokens cuts PA joint error by 0.751 mm; RGB-only, shuffled-train, inference-shuffle, and inference-zero controls show complementary image and paired-rope value. |
| 2026-07-18 | [0076_direct_pose_transfer_and_joint_training.md](0076_direct_pose_transfer_and_joint_training.md) | ARCTIC-only transfers to controlled occlusion but not held-out natural HOT3D; adding 2,166 participant-separated HOT3D hands cuts held-out PA by 1.372 mm while retaining 93.4% of the ARCTIC gain, so expand trusted data before LoRA. |
| 2026-07-18 | [0077_hot3d_cv_data_scale_capacity_lora.md](0077_hot3d_cv_data_scale_capacity_lora.md) | Strict three-fold HOT3D participant CV assigns 0.637 mm to trusted data expansion, only 0.087 mm to a 7.9x larger head, no selected gain to matched last-two-block Q/V LoRA, and exposes a non-dominating ARCTIC/FreiHAND-versus-HO3D mixture trade-off. |
| 2026-07-19 | [0078_normalized_rope_orientation_and_robustness.md](0078_normalized_rope_orientation_and_robustness.md) | Normalized-rope matrix finds a usable continuous-error range but severe dropout brittleness; paired rope improves HOT3D root-relative orientation, yet camera error and ARCTIC transfer reject it as a general global-pose solution. |
| 2026-07-19 | [0079_normal_joint_no_leak_final.md](0079_normal_joint_no_leak_final.md) | Content-level audit invalidates leaked 0076 HOT3D-all, reclassifies 0077 as reused participant-disjoint validation, fixes the HO3D train/eval tip convention, and finds that normal triple training clearly improves HO3D but is statistically flat/slightly worse than dual on ARCTIC/HOT3D, so continue without promotion or test-set mixture tuning. |
| 2026-07-19 | [0080_code_branch_artifact_temporal_audit.md](0080_code_branch_artifact_temporal_audit.md) | Maps active/frozen/legacy entrypoints and durable artifacts, proves the old temporal branch is fully contained in the active branch, preserves the dirty detached worktree, bounds remote cleanup candidates, and permits only one sequence-disjoint localized-state temporal gate. |
| 2026-07-20 | [0081_dexycb_s1_first_round.md](0081_dexycb_s1_first_round.md) | Validates official DexYCB S1 splits, projection/native-MANO coordinates, deterministic train27k, external transfer, matched RGB-only versus ideal-rope training, one-shot test scoring, and raw-tree immutability; stops the RGB-only/full-scale/mixture path while validating only the GT-derived rope effect. |
| 2026-07-20 | [0082_egoverse_download_link_probe.md](0082_egoverse_download_link_probe.md) | Direct no-proxy Secrets/SQL/R2 probes byte-verify four RL2/ETH/Song/Scale episodes, identify 100-frame array padding, source-dependent keypoint/language fields, and useful episode-level rather than CPU-core parallelism. |
| 2026-07-20 | [0083_egoverse_capacity_and_parts.md](0083_egoverse_capacity_and_parts.md) | Records full Academic coverage, Aria temporal failures, completed 689.39 GB Mecka coverage, and persistent-connection Scale continuations `193648/193719`. |
| 2026-07-20 | [0083_interhand26m_oneview_first_round.md](0083_interhand26m_oneview_first_round.md) | Validates the InterHand2.6M v1.0 30fps one-view external anchor, rejects a Capture9-starving train27k-v1 selector, verifies corrected capacity-balanced v2, stops old transfer and the ideal-rope recipe, and preserves the one-shot test boundary. |
| 2026-07-21 | [0084_direct_pose_error_decomposition.md](0084_direct_pose_error_decomposition.md) | Existing-prediction decomposition passes five-dataset ID/parity gates, locates the cross-dataset camera mismatch in translation rather than universal rotation, and permits only a minimal delta-camera-translation screen while keeping global orientation and LoRA stopped. |
| 2026-07-22 | [0085_repository_package_and_script_consolidation.md](0085_repository_package_and_script_consolidation.md) | Consolidates stable dataset/evaluation/refiner logic into the package, categorizes scripts, records the dataset contract, and verifies unchanged behavior with 451 local tests plus exact remote HOT3D, DexYCB, and InterHand scoring parity smokes. |
| 2026-07-22 | [0086_existing_prediction_per_finger_gate.md](0086_existing_prediction_per_finger_gate.md) | Five-domain existing-prediction finger/tail oracle plus matched InterHand input-only correct/shuffled controls; keeps the current head, separates InterHand from joint training, and defers fusion redesign pending physical evidence. |
| 2026-07-22 | [0087_direct_pose_gradient_conflict_audit.md](0087_direct_pose_gradient_conflict_audit.md) | Five-domain training-only gradient/one-step audit; exact missing-sensor fallback passes, but equal/near-equal four-core mixing fails, so the four local-decoder adaptation cells are not run. |
| 2026-07-23 | [0088_direct_pose_product_validation.md](0088_direct_pose_product_validation.md) | Existing-head conflict attribution, exact invalid-channel fallback, HOT3D/ARCTIC product perturbation matrix, ARCTIC noise ceiling, and PALM boundary; keeps DirectPose conditional and decoder/shared retraining stopped. |
| 2026-07-24 | [0089_direct_pose_observability_and_posterior_gate.md](0089_direct_pose_observability_and_posterior_gate.md) | Five-domain rope/token observability and K-sensitivity: five ideal ropes have a 40D local null-space lower bound, current HOT3D tokens do not selectively collapse under low visibility, and K=8 is the smallest significant offline conditional-posterior ceiling while ARCTIC rejects always-on retrieval. |
