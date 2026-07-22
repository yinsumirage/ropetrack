# Current Code, Artifact, Branch, And Temporal Map

Status: current as of 2026-07-21. Read this before dated plans. Dated
`docs/` files and `experience/` notes remain the evidence trail; they are not
all active instructions.

## Research Status

- **Released/validated:** P0-P2 is closed. The formal release remains the 0044
  four-teacher alpha student pinned by `RELEASE.md`.
- **Current/continue:** `DirectPoseHead` is the active experimental hand-shape
  line. It trains a small 45D residual head through the MANO joint loss using
  frozen localized WiLoR tokens and normalized rope. The WiLoR image backbone
  is not trained end to end.
- **Not promoted:** the frozen ARCTIC+HOT3D+HO3D mixture improves HO3D but is
  statistically flat/slightly worse than the matched dual model on ARCTIC and
  HOT3D. Do not tune another mixture on the same final scores.
- **DexYCB S1:** the outer adapter, official unseen-subject protocol, projection,
  and native-MANO decode are validated. The matched 27k RGB-only head is
  stopped because root-relative test error clearly regresses; GT-derived
  ideal rope has a large paired effect, but does not justify full-S1 scaling,
  mixture promotion, or a physical-sensor claim. See 0081.
- **InterHand2.6M v1.0 30fps:** the one-view adapter, coordinate/left-right
  gates, frozen external anchor, and corrected capture/sequence-balanced
  train27k-v2 are validated. Corrected RGB-only improves official-val PA/root
  but worsens camera/mesh; ideal rope further worsens PA/camera/mesh. Stop old
  transfer, the rope recipe, full-view, scale-up, and mixture expansion. The
  original train27k-v1 supervision is invalid because Capture9 was starved;
  corrected v2 was not rerun on the already-observed one-shot test. See 0083.
- **Stopped:** dense K16/K96 history, larger GRU/Transformer variants on the
  same signals, the tested natural-HOT3D visibility/usefulness gates, and the
  global-orientation head.
- **Next bounded screen:** the five-dataset existing-prediction decomposition
  localizes the shared camera mismatch to translation, not universal rotation.
  A minimal 3D `delta cam_t` branch may be tested on validation data while the
  local DirectPose path stays frozen. Do not revive `delta global_orient`, add
  scale/betas, tune final/test mixtures, or use LoRA. See 0084.
- **Sensor boundary:** correct paired GT-derived rope has a causal hand-shape
  effect, but this is ideal simulated geometry. Physical sensor calibration,
  drift, latency, missing channels, and real deployment remain unproved.
- **Gradient/adaptation gate:** the five-domain training-only audit in 0087
  validates exact per-finger/all-missing fallback but rejects equal/near-equal
  ARCTIC+HOT3D+HO3D+DexYCB mixing. HOT3D-DexYCB is significantly conflicting;
  DexYCB PA-vs-root is significantly aligned rather than opposed. InterHand
  remains stress-only. The four local-decoder adaptation cells were therefore
  not built or run; no PCGrad/sampler/dataset-conditioned replacement is
  selected.
- **Product perturbation follow-up:** 0088 keeps the existing h128 head as a
  conditional HOT3D-centered path and keeps decoder adaptation stopped.
  Correct rope cuts HOT3D PA from `8.984` to `5.732 mm`; the fixed
  low-visibility slice gains `4.123 mm`, versus `2.346 mm` for context.
  Exact explicit/non-finite/out-of-range per-finger fallback passes and
  all-missing is numerically WiLoR. HOT3D passes the frozen robustness matrix;
  ARCTIC retains 73.3% of clean gain at simulated `noise=0.03` but only 55.4%
  at `0.04`. New shared-head/decoder training remains stopped pending a
  training-only non-regression proposal and physical sensor evidence.

The authoritative normal-mixture no-leak record is
`experience/0079_normal_joint_no_leak_final.md`; the DexYCB S1 first-round
record is `experience/0081_dexycb_s1_first_round.md`; the InterHand one-view
record is `experience/0083_interhand26m_oneview_first_round.md`; and the
cross-dataset error decomposition is
`experience/0084_direct_pose_error_decomposition.md`.

## Active Code Paths

| Area | Tracked entry and call chain | Checks/evidence | Status |
|---|---|---|---|
| DirectPose normal training | `prepare_arctic.py` / `prepare_hot3d.py` / `prepare_ho3d_normal_train.py` -> `eval.py --save-mano-cache` -> `extract_feature_cache.py --save-tokens` -> `apply_rope_refinement.py` cache -> `direct_pose_head.py train --extra-bundle ...` | `test_prepare_*`, `test_eval_pipeline.py`, `test_extract_feature_cache.py`, `test_direct_pose_head.py`; 0075-0079 | Active experiment. Bundle/protocol/stitch Slurm glue is intentionally run-local under ignored `.local_checks/` and archived in the HPC run root, not a public package API. |
| DirectPose apply/score | `direct_pose_head.py apply` -> MANO decode from `apply_rope_refinement.py` -> project `pred.json` -> `score_predictions.py` | `test_direct_pose_head.py`, `test_apply_rope_refinement.py`, `test_score_predictions.py`; 0078-0079 | Active. Perturbation flags are the normalized-rope robustness controls. |
| Existing-prediction decomposition | `analyze_pose_error_decomposition.py` reads project `pred.json` + explicit sample order, audits IDs, computes the proper nested oracle envelope, group bootstrap, subgroups, and artifact verification | `test_pose_error_decomposition.py`; 0084 | Stable CPU analysis. Raw alignment candidates and legacy parity are kept separate; generated per-sample/results stay in the remote run root. |
| DirectPose gradient audit and safety gate | thin `scripts/evaluation/audit_direct_pose_gradients.py` -> `ropetrack.refine.direct_pose_audit`; consumes frozen training bundles/checkpoint, emits gradient/transfer matrices, exact fallback gate, and verifier artifacts | `test_direct_pose_audit.py`; 0087 | Completed/STOP for equal four-core mixing. Reuse the verifier and exact fallback; do not run the gated decoder cells from this result. |
| DirectPose product fallback/perturbation | `scripts/rope_refiner/direct_pose_head.py` exact per-finger fallback plus run-local fixed perturbation/scoring launchers over existing checkpoints and caches | `test_direct_pose_head.py`; real-checkpoint invalid gate and product verifier; 0088 | Experimental conditional path. Clean and explicit-invalid behavior is structurally safe; HOT3D robustness passes, but the original ARCTIC `noise=0.05` retention gate blocks robust retraining and a deployable claim. Hardware validity metadata remains required. |
| P0-P2 teacher/release | `apply_rope_refinement.py --mode optimize|student`; `train_alpha_student.py`; core `refine/{actions,alpha_student,analysis,cache,oracle}.py` | release golden check in `RELEASE.md`; broad refiner tests; 0027-0052 | Frozen supported path. Do not replace the release checkpoint with DirectPose outputs. |
| Dataset adapters/export | `ropetrack/datasets/hand_pose.py`, dataset YAMLs, `prepare_{arctic,egodex,hot3d,dexycb}.py`, `prepare_ho3d_normal_train.py`, `make_hard_images.py`, `make_rope_labels.py` | adapter/export/hard/rope tests; 0018, 0040, 0060-0073, 0079, 0081 | Reusable. Dataset-specific coordinate, side, and tip conventions remain explicit. DexYCB test export requires a frozen recipe. |
| DexYCB protocol/evaluation | `prepare_dexycb.py` -> `validate_dexycb_coordinates.py` -> standard WiLoR/token/DirectPose paths -> `score_dexycb.py`; `freeze_dexycb_recipe.py` and `verify_dexycb_artifacts.py` enforce one-shot test and raw-tree checks | `test_prepare_dexycb.py`, `test_validate_dexycb_coordinates.py`, `test_score_dexycb.py`, `test_freeze_dexycb_recipe.py`; 0081 | Validated adapter and evaluation path. The 27k RGB-only recipe, full-S1 scale-up, and addition to the frozen joint mixture are stopped. Ideal GT-derived rope remains an oracle/simulated observation. |
| InterHand one-view protocol/evaluation | `prepare_interhand26m.py` -> `validate_interhand26m_coordinates.py` -> standard WiLoR/token/DirectPose paths -> `score_interhand26m.py`; `interhand26m_protocol.py` enforces freezes/raw checks and train capacity balance | `test_interhand26m.py`, `test_apply_rope_refinement.py`; 0083 | External anchor and corrected train27k-v2 validated. RGB-only is diagnostic only; current ideal-rope recipe, old transfer, full-view/scale-up, and mixture expansion are stopped. Historical train27k-v1 supervision is invalid. |
| Benchmark export/eval | `eval.py` -> `ropetrack.eval.config` -> `ropetrack.eval.pipeline` -> `HandPredictor` + dataset adapter -> optional `score_predictions.py` | eval/config/pipeline/backend/protocol tests; 0017-0019, 0051 | Supported. `score_sliced_predictions.py` is the hard/rope slice scorer; temporal scoring is separate. |
| Rope diagnostics/report | `rope_diagnostics/*`, `analyze_alpha_deadzone.py`, `summarize_runs.py`, `plot_report_figures.py`, `make_qualitative_panels.py` | dedicated tests except the thin visualization CLI; 0021-0023, 0029, 0042-0052 | Supported analysis. Generate tables; do not hand-copy metrics. |
| Global orientation | `direct_global_orient_head.py` reuses `DirectPoseHead` and MANO decode | self-check plus `test_direct_global_orient_head.py`; 0078, 0084 | Experimental/rejected. HOT3D root-relative gain worsens camera error and fails ARCTIC transfer; the cross-dataset decomposition supports translation-only screening, not revival of this head. |

`scripts/README.md` is the command catalog. The table above decides whether a
command is current, frozen, or experimental.

## Artifact Contracts

Generated data and results are never committed.

- Benchmark export: `<run>/run_meta.json` pins `sample_order`;
  `<run>/mano_cache.npz` stores predicted MANO state; the project prediction is
  `<run>/eval_input/pred.json` with payload `[xyz_predictions,
  vertex_predictions]`.
- Refiner apply: `<out>/refiner_eval_cache.npz` is the aligned input cache;
  `<out>/base_pred.json` and `<out>/pred.json` share one decoder; alpha,
  residual, and summary files stay beside them.
- Frozen P2 release checkpoint:
  `/data/wentao/ropetrack/releases/p2_four_teacher_student/student.pt`.
  Its original run is
  `/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104`.
- DirectPose checkpoint contract: `model.pt` contains `model_state`, model
  config, Git/protocol provenance, and train/validation sample hashes;
  `train_log.json` records selection. Apply writes `pred.json`,
  `sample_id.npy`, `refined_hand_pose.npy`, and `summary.json`.
- Frozen normal-joint run:
  `/data/wentao/ropetrack/runs/direct_pose_normal_joint_20260719`. Durable
  evidence is under `audit/`, `protocol/`, `scores/`, and
  `students/fold{0,1,2}/model.pt`. The corrected train-only HO3D export is
  `/data/wentao/ropetrack/processed/ho3d_v3/normal_train_27000_evaltips_20260719`.
- DexYCB S1 first-round run:
  `/data/wentao/ropetrack/runs/direct_pose_dexycb_s1_20260720`; processed
  manifests/targets are under
  `/data/wentao/ropetrack/processed/dexycb/s1_v1`. Durable evidence is
  `protocol/artifact_verification.json`, `protocol/recipe_freeze.json`,
  `coordinate/{pretrain,final}/`, `external_transfer/summary.json`,
  `val/scores/scores.json`, and the one-shot `test/scores/scores.json`.
  Checkpoints are `students/{rgb_only,rgb_rope}/model.pt`; neither is a release
  replacement.
- Earlier DirectPose runs are
  `direct_pose_head_20260718`, `direct_pose_transfer_20260718`,
  `direct_pose_scale_20260718`, and `direct_pose_ab_20260719` under the same
  `/data/wentao/ropetrack/runs/` root. They are experiment evidence, not
  release checkpoints.
- Product/conflict evidence is under
  `/data/wentao/ropetrack/runs/direct_pose_conflict_attribution_20260723` and
  `/data/wentao/ropetrack/runs/direct_pose_product_validation_20260723`.
  The latter contains the frozen protocol, raw matrix, report, verifier,
  invalid-value gate, and the separately frozen ARCTIC noise-dose diagnostic.

Keep small checkpoints, protocols, hashes, scores, manifests, and archived
launch scripts. Large feature/activation caches and prediction matrices are
recomputable cleanup candidates, but deleting them requires explicit user
approval and must preserve the small evidence files first.

## Temporal Code: Keep, Freeze, Or Stop

| Code | Current reachability | Test/evidence | Decision |
|---|---|---|---|
| `ropetrack/refine/temporal.py` | Imported by `make_hard_images.py`, `train_alpha_student.py`, `train_temporal_student.py`, `apply_rope_refinement.py`, and oracle scripts | `test_temporal_refiner.py`, refiner/apply tests; 0053-0059 | Keep as a tested compatibility/research library. Schema-v1 checkpoint loading is legacy compatibility, not a new-training recommendation. |
| `train_temporal_student.py`, `apply_rope_refinement.py --mode temporal`, `score_temporal_predictions.py` | CLI help/import chain passes | `test_temporal_refiner.py`, `test_apply_rope_refinement.py`, `test_score_temporal_predictions.py`; 0053-0055 | Mark legacy experimental. Do not restart K16/K96 or scale the sequence model. |
| `temporal_oracle_state.py`, `temporal_state_followups.py`, `apply_learned_visibility_state.py`, `probe_image_visibility_gate.py`, `evaluate_visibility_shift.py` | Tested CLIs; some import one another | matching `test_*.py`; 0056-0059 | Keep as reusable oracle/state evaluation pieces. They support a single future localized-state gate, not a generic temporal stack. |
| `analyze_clean_prefix_state.py`, `temporal_rope_state_arbitration.py`, `probe_visibility_gate.py`, `compose_visibility_shift_scores.py` | Only old experiment workflows call them | matching `test_*.py`; 0056-0059 | Freeze as legacy evidence for rejected velocity, rope-arbitration, and mixed-gate families. |
| `hot3d_natural_state.py`, `probe_hot3d_natural_gate.py` | CLI help/import chain passes | matching tests; 0074 | Freeze as 0074 reproduction code. The expanded participant-held-out result stopped the simple natural-HOT3D state gate. |

No tracked temporal file was deleted in this audit: every candidate either has
a current import, a runnable tested compatibility path, or unique experiment
reproduction value. Ignored `.local_checks/` launchers are per-run recipes,
not supported entrypoints; do not copy them into the package or commit their
predictions/caches.

## Branch And Worktree Decisions

As audited at `997da91`:

- `codex/temporal-oracle-state` is the active research branch. Continue here.
- `codex/temporal-ho3d-refiner` (`ba31824`) is an exact ancestor of the active
  branch: it has 22 commits beyond `main`, zero commits absent from the active
  branch, and needs no merge or cherry-pick. Keep only as a historical marker;
  delete local/remote refs later only with explicit approval.
- `main` (`bcb81dd`) is the P0-P2/E1-E2 release baseline and an ancestor of
  both branches. It has no unique commit to recover. Update it later through a
  deliberate review/PR if the active research history should become canonical;
  this audit does not merge it.
- `C:/Users/gwt/.codex/worktrees/f2fc/ropetrack` is detached at old commit
  `e2f6d31` and contains uncommitted protocol/note changes. Equivalent tip
  policy and protocol work exists on the active branch, but the exact dirty
  patch would be lost if the worktree were removed. Leave it untouched until
  the user explicitly approves archival/removal.

## Only Justified Future Temporal Experiment

Do not continue temporal work merely because DirectPoseHead now trains. First
run one bounded **sequence-disjoint localized-state** experiment:

1. Freeze the current h128 DirectPose framewise model and frozen WiLoR tokens.
   Use normal ARCTIC train subjects, HOT3D training participants, and HO3D v3
   train sequences. Split state/gate training by whole subject/participant/
   sequence; keep ARCTIC `s05`, HOT3D held-out participants, and HO3D official
   evaluation outside training and threshold selection.
2. Compare framewise DirectPose with one causal state rule: keep the last
   trusted localized visual pose/token state, cap its age at 60 frames, and use
   current RGB tokens plus current valid rope for the update. At expiry, fall
   back to the current frame; do not extrapolate velocity.
3. Freeze deterministic occlusion and rope-dropout schedules before scoring.
   Required controls are framewise DirectPose, phase/quality-oracle state,
   learned state, shuffled stored state, and shuffled/current-missing rope.
4. Continue only if the oracle first improves masked PA by at least 0.5 mm on
   at least two datasets. Promote the learned rule only if its sequence-level
   95% CI excludes zero on those datasets, it recovers at least half of the
   oracle gain, clean PA regresses by no more than 0.10 mm on every anchor,
   10% rope dropout retains at least 70% of the clean DirectPose gain, and 20%
   dropout is no worse than the explicit RGB/base fallback.
5. Stop the temporal line if the oracle gate fails, the gain depends on one
   dataset, or the learned rule misses any clean/dropout safety gate. Do not
   respond with K16/K96, a GRU/Transformer, another velocity grid, or a larger
   usefulness gate.

This is a future HPC experiment, not a claim that current temporal code is a
release path.
