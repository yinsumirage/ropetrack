# Scripts

Current benchmark entrypoints:

- `eval.py`: config-driven benchmark export and eval entrypoint.
- `score_predictions.py`: PA/mesh/F-score evaluator invoked by `eval.py --run-eval`.
- `analyze_pose_error_decomposition.py`: existing-prediction camera/root/T/RT/Sim3 decomposition with sample-ID, parity, group-bootstrap, and artifact-verification gates.
- `evaluation/audit_direct_pose_gradients.py`: training-only five-domain DirectPose gradient cosine, one-step transfer, and exact missing-sensor fallback audit; consumes an externally frozen protocol and writes only to the remote run root.
- `score_sliced_predictions.py`: occlusion/rope-residual sliced scorer (details below).
- `make_hard_images.py`: hard-image split generator.
- `make_rope_labels.py`: GT fingertip-to-wrist rope label generator.
- `prepare_egodex.py`: EgoDex MP4/HDF5 exporter for image + camera-frame 21-joint manifests.
- `audit_ho3d_train_split.py`: format audit before touching an HO3D train split.
- `rope_refiner/apply_rope_refinement.py`: teacher optimization / student apply.
- `rope_refiner/train_alpha_student.py`: P2 alpha-student distillation trainer.
- `rope_refiner/summarize_runs.py` + `plot_report_figures.py` + `make_qualitative_panels.py`: report tables/figures/panels.
- `rope_refiner/analyze_alpha_deadzone.py`: curl-vs-closure diagnostic.
- `rope_head/extract_feature_cache.py`: P3 frozen backbone feature caching.
- `rope_diagnostics/score_rope_predictions.py`: rope diagnostic scorer for exported `pred.json`.
- `rope_diagnostics/analyze_rope_errors.py`: rope reliability analysis tables/figures.
- `rope_diagnostics/visualize_mesh_comparison.py`: mesh/prediction visual check helper.
- `prepare_ho3d_normal_train.py`: sequence-balanced, train-only HO3D v3 normal export with the evaluation tip convention.
- `prepare_interhand26m.py`: deterministic InterHand2.6M v1.0 30fps one-view manifests, train27k, and freeze-gated test export.
- `validate_interhand26m_coordinates.py`: official-source, projection, native-MANO, and WiLoR/DirectPose left-hand gate.
- `score_interhand26m.py`: side-aware joint/mesh diagnostics and frame-group bootstrap scoring.
- `analyze_interhand26m_rope.py`: post-hoc population, paired-hand overlap, rope-residual, MANO-fit, per-finger, and qualitative diagnostic figures; never a checkpoint selector.
- `interhand26m_protocol.py`: pre-val/test freezes, raw-tree signatures, and final artifact verification.
- `rope_refiner/direct_pose_head.py`: active experimental DirectPose train/apply CLI over frozen localized tokens and normalized rope.

Status and replacement decisions live in
`../docs/current-code-and-artifact-map.md`. In particular,
`rope_refiner/direct_global_orient_head.py` and the temporal/state/gate CLIs
remain reproducibility tools, not current release entrypoints. Do not restart
K16/K96 or larger generic temporal models from their continued presence.

Typical usage:

```powershell
python scripts\eval.py --dataset ho3d_v2 --method wilor_anyhand --run-eval
```

The error-decomposition entrypoint is CPU-only and takes run-specific absolute
paths from an external JSON config, so dataset artifacts are never hardcoded in
the metric implementation:

```bash
python scripts/analyze_pose_error_decomposition.py \
  --config <run-root>/pose_error_decomposition_config.json \
  --output-root <run-root>
```

InterHand2.6M uses the project one-view protocol documented in
`../docs/interhand26m-oneview-protocol.md`. Run its 3-4 GB annotation parsing
on a high-memory CPU Slurm node, not a login node. Before freeze, `test_index`
uses only data JSON identifiers/input metadata; the separate `test` export
requires the matching `test_freeze.json`:

```bash
# Current corrected capture/sequence-balanced train subset. The historical
# oneview_v1/train27k directory is preserved but invalid for supervision.
python scripts/prepare_interhand26m.py \
  --raw-root /data/wentao/datasets/interhand2.6m_v1_30fps \
  --output-root /data/wentao/ropetrack/processed/interhand2.6m_v1_30fps/oneview_v1_train27k_v2 \
  --splits train

# Build the aligned DirectPose cache without running a teacher optimizer.
python scripts/rope_refiner/apply_rope_refinement.py --dataset interhand26m \
  --rope-labels <ROOT>/rope_labels.jsonl --pred-dir <BASE>/eval_input \
  --run-meta <BASE>/run_meta.json --mano-cache <BASE>/mano_cache.npz \
  --out-dir <RUN_SPLIT> --cache-only
```

EgoDex uses paired MP4/HDF5 episodes. The source provides per-frame skeletal
SE(3) transforms and confidence, but **not native MANO parameters or mesh
vertices**. Keep raw data under `/data/wentao/datasets/egodex`; put decoded,
project-specific subsets under `/data/wentao/ropetrack/processed/egodex`.

```bash
# CPU: decode a sampled evaluation subset. Each hand is one sample; the same
# image can therefore appear in one left-hand row and one right-hand row.
python scripts/prepare_egodex.py \
  --input-root /data/wentao/datasets/egodex/test \
  --output-root /data/wentao/ropetrack/processed/egodex/test_eval \
  --split evaluation --hands both --frame-stride 10

# CPU: build RopeTrack GT rope labels directly from EgoDex's 21 joints.
python scripts/make_rope_labels.py --dataset egodex \
  --input-root /data/wentao/ropetrack/processed/egodex/test_eval \
  --output /data/wentao/ropetrack/processed/egodex/test_eval/rope_labels.jsonl \
  --split evaluation --viz-dir /data/wentao/ropetrack/runs/egodex_viz --viz-count 16

# GPU: predict WiLoR joints/vertices and a MANO cache, then score the joint GT.
# EgoDex has no GT mesh, so mesh/F-score fields are reported as unavailable.
python scripts/eval.py --dataset egodex_test \
  --root /data/wentao/ropetrack/processed/egodex/test_eval \
  --save-mano-cache --run-eval
```

For the current RopeTrack teacher/student training path, export a training
root from `part1`, then omit `--run-eval` while generating its predicted MANO
cache. The resulting run directory and rope labels are ordinary inputs to
`apply_rope_refinement.py --dataset egodex` and `train_alpha_student.py`:

```bash
python scripts/prepare_egodex.py \
  --input-root /data/wentao/datasets/egodex/part1 \
  --output-root /data/wentao/ropetrack/processed/egodex/part1_train_s10 \
  --split training --hands both --frame-stride 10
python scripts/eval.py --dataset egodex_test --split training \
  --root /data/wentao/ropetrack/processed/egodex/part1_train_s10 \
  --out-dir /data/wentao/ropetrack/runs/egodex_part1_train_s10 \
  --save-mano-cache
```

For temporal training use `--split training --frame-stride 1` on selected
episodes. Both hands share the `<task>__<episode>` sequence key so an episode
cannot leak across train/validation; right-hand frame numbers receive a fixed
offset, making the two hand streams distinct contiguous segments.
The exported `mano_cache.npz` is a model prediction. It must not be described
as EgoDex GT MANO. Apple's own 2D visualization also warns that projected
joints may be slightly displaced in the synthesized Vision Pro RGB, so inspect
the generated overlays before trusting a projected-joint bbox margin.

ARCTIC P2 uses the same manifest-backed path. The raw dataset root is the
innermost `arctic_data/data`; full training exports joints and rope labels but
not the unused GT meshes:

```bash
python scripts/prepare_arctic.py --split train --skip-image-check
python scripts/make_rope_labels.py --dataset arctic --split training \
  --input-root /data/wentao/ropetrack/processed/arctic/p2_train \
  --output /data/wentao/ropetrack/processed/arctic/p2_train/rope_labels.jsonl
python scripts/eval.py --dataset arctic_p2_train --split training \
  --out-dir /data/wentao/ropetrack/runs/arctic_p2_train_wilor --save-mano-cache
```

Use `--skip-image-check` only after an external image-tree integrity audit;
omit it for small mapping smokes. The resulting WiLoR run, MANO cache, and rope
labels are ordinary inputs to `apply_rope_refinement.py` and
`train_alpha_student.py`.

Hard split roots are generated as normal dataset roots and then selected by
dataset config name:

```bash
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/mask70 --effect mask --severity 0.70 --limit 0
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/tip_square80 --effect tip_square --severity 0.80 --limit 0
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/finger_end80 --effect finger_end --severity 0.80 --limit 0
python scripts/eval.py --dataset ho3d_v2_mask70 --method wilor_anyhand --run-eval
```

HO3D v3 TRAIN pipeline (multi-dataset P2 teacher; assumptions verified by
`scripts/audit_ho3d_train_split.py`, experience/0040 — train metas have no
handBoundingBox, so GT bboxes are projected from `handJoints3D`). Stride is
baked into the hard root's own `train.txt`, so downstream steps stay aligned
by construction; both generators are CPU-only:

```bash
# CPU: strided hard root (stride 4 -> ~20.8k of 83,325 video frames)
python scripts/make_hard_images.py --dataset ho3d --split training --stride 4 --input-root /data/wentao/ropetrack/HO3D_v3 --output-root /data/wentao/ropetrack/hard/ho3d_v3/mask70_train --effect mask --severity 0.70 --limit 0
# CPU: matching rope labels (same stride, same source list)
python scripts/make_rope_labels.py --dataset ho3d --split training --stride 4 --input-root /data/wentao/ropetrack/HO3D_v3 --output /data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl
# GPU: export + winner teacher on the hard train root
python scripts/eval.py --dataset ho3d_v3_mask70_train --method wilor_original --split training --protocol-check-samples 0 --save-mano-cache
python scripts/rope_refiner/apply_rope_refinement.py --mode optimize --objective rope --action-space flex15 --gate-residual-threshold 0.1 --dataset ho3d ...
# then multi-dataset distillation:
python scripts/rope_refiner/train_alpha_student.py --teacher-dir <freihand_teacher> <ho3d_v3_teacher> --action-space flex15 --out-dir <student_multi>
```

Rope labels are JSONL so each row keeps sample id, raw distance, chain length,
normalized value, validity, and normalization metadata:

```bash
python scripts/make_rope_labels.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output /data/wentao/ropetrack/rope/ho3d_v2_rope.jsonl --viz-dir /data/wentao/ropetrack/runs/rope_viz/ho3d_v2 --viz-count 16
python scripts/rope_diagnostics/score_rope_predictions.py /data/wentao/ropetrack/runs/clean_baseline/ho3d_v2_wilor_original/eval_input /data/wentao/ropetrack/rope/ho3d_v2_rope.jsonl /data/wentao/ropetrack/runs/rope_scores/ho3d_v2_wilor_original --dataset ho3d --run-meta /data/wentao/ropetrack/runs/clean_baseline/ho3d_v2_wilor_original/run_meta.json
python scripts/rope_diagnostics/analyze_rope_errors.py /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/scores /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/diagnostics
```

FreiHAND train roots for teacher generation (see RELEASE.md for the full
provenance chain):

```bash
python scripts/make_hard_images.py --dataset freihand --split training --input-root /data/wentao/ropetrack/FreiHAND --output-root /data/wentao/ropetrack/hard/freihand/mask70_wilor_training --effect mask --severity 0.70 --limit 0
python scripts/make_rope_labels.py --dataset freihand --split training --input-root /data/wentao/ropetrack/FreiHAND --output /data/wentao/ropetrack/rope/freihand_training_rope.jsonl
```

`rope_refiner/apply_rope_refinement.py` has two modes:

- `--mode optimize` (default): per-sample teacher optimization, no training.
- `--mode student`: apply a distilled alpha-student checkpoint.

(The exploratory 45-dim MLP refiner and its `--mode checkpoint` path were
removed in the consolidation pass; the negative result stays documented in
experience/0026.)

Optimize mode supports the P0 probes from
`docs/2026-07-06-rope-refinement-next-plan.md`:

- `--objective rope|oracle_tip|oracle_chain`: rope-label MSE, or GT-joint
  ceiling probes (`oracle_*` needs `--gt-xyz <split>_xyz.json`, same row
  order as `run_meta.json` `sample_order`).
- `--action-space mult5|mult15|flex15|flex5`: original per-finger curl scale,
  per-joint scale, additive per-joint flexion, or additive per-finger coupled
  flexion (`flex5`: the finger's 9-dim rope gradient normalized as one unit
  vector — matched capacity with no within-finger null space). Flex
  directions are frozen rope-gradient directions saved to
  `flex_directions.npy`.
- `--gate-residual-threshold 0.1`: P1 gating — only fingers whose base rope
  residual exceeds the threshold (normalized units) may move; ungated
  fingers keep `alpha = 0` and are excluded from the rope loss. Gating
  stats are recorded in `summary.json`.
- `--rope-noise-std 0.05 --rope-dropout 0.2 --rope-noise-seed 0`: simulated
  imperfect sensor (H5 ablation) — seeded gaussian noise on the normalized
  rope reading (0.05 is roughly +/-2.5 mm) and per-finger dropout that marks
  readings invalid. `gt_rope_norm` in the cache keeps the clean labels; the
  gate and loss both consume the perturbed reading.
- Optimizer defaults are now the published working recipe from
  `experience/0027` (`steps=120 lr=2.0 alpha_l2=0.001 max_alpha=0.5`);
  the old conservative defaults provably did nothing.
- Every run writes `rope_residuals.npz` plus a `summary.json` with alpha
  stats and rope residual closure, both computed through the same MANO
  decode path as `base_pred.json`/`pred.json`.

P2 distillation (teacher -> one-pass student):

```bash
# 1. generate teacher targets on the TRAINING split with the frozen winner recipe
python scripts/rope_refiner/apply_rope_refinement.py --mode optimize --objective rope \
  --action-space flex15 --gate-residual-threshold 0.1 \
  --opt-steps 400 --opt-lr 32 --opt-alpha-l2 0.001 \
  ... --out-dir <teacher_train_dir>
# 2. train the student (imitation + noise augmentation + val/early-stop)
python scripts/rope_refiner/train_alpha_student.py --teacher-dir <teacher_train_dir> \
  --action-space flex15 --out-dir <student_dir>
# 2b. mandatory control: gains must vanish with shuffled rope
python scripts/rope_refiner/train_alpha_student.py --teacher-dir <teacher_train_dir> \
  --action-space flex15 --out-dir <student_shuffled_dir> --shuffle-rope
# 3. evaluate the student through the exact same decode/scoring path as the teacher
python scripts/rope_refiner/apply_rope_refinement.py --mode student \
  --checkpoint <student_dir>/student.pt ... --out-dir <student_eval_dir>
```

The student predicts the teacher's alphas in one forward pass (no 400-step
optimization at inference); the residual gate stays a hard rule from the
checkpoint config, and `--rope-loss-weight` optionally adds a differentiable
rope-consistency term (needs `--mano-cache`).

P3 rope-conditioned head prep (`scripts/rope_head/`):

```bash
# cache frozen backbone features (same crops as the benchmark export;
# hook on model.backbone during a normal forward). GPU, gt_bbox only.
python scripts/rope_head/extract_feature_cache.py --dataset freihand_mask70 --method wilor_original --split evaluation --output /data/wentao/ropetrack/features/freihand_mask70_eval_wilor.npz
python scripts/rope_head/extract_feature_cache.py --dataset freihand_mask70 --method wilor_original --split training --freihand-root <mask70_train_hard_root> --output /data/wentao/ropetrack/features/freihand_mask70_train_wilor.npz
```

`--pooling meanmax` doubles the feature dim; `--save-tokens` stores the full
fp16 token grid for the v1 cross-attention head (~16 GB for a 32.5k split -
default off).

Report tooling (no hand-copied tables):

```bash
# aggregate every cell (summary.json + sliced + scores) into TSV/Markdown/JSON
python scripts/rope_refiner/summarize_runs.py <run_root1> <run_root2> --output-dir <tables_dir>
# figures from the aggregated JSON
python scripts/rope_refiner/plot_report_figures.py --summary <tables_dir>/runs_summary.json --figure dose_response --cell-filter sweep/ --output <figs>/dose_response.png
python scripts/rope_refiner/plot_report_figures.py --summary <tables_dir>/runs_summary.json --figure noise --cell-filter noise/ --output <figs>/noise_curve.png
```

P0 analysis entrypoints (CPU, numpy-only):

```bash
python scripts/score_sliced_predictions.py <apply_out_dir> <scores_out_dir> --dataset freihand --gt-dir <hard_root> --hard-manifest <hard_root>/hard_manifest.jsonl --cache <apply_out_dir>/refiner_eval_cache.npz
python scripts/rope_refiner/analyze_alpha_deadzone.py --cache <apply_out_dir>/refiner_eval_cache.npz --alpha <apply_out_dir>/alpha.npy --residuals <apply_out_dir>/rope_residuals.npz --action-space mult5 --output-dir <deadzone_out_dir>
```

`score_sliced_predictions.py` slices PA-aligned per-joint errors by
occluded/clean fingers and rope-residual buckets (H1); its `all_joints`
slice reproduces `xyz_procrustes_al_mean3d`. `analyze_alpha_deadzone.py`
correlates base finger curl with correction size and residual closure (H2).

Dataset roots live in `configs/datasets/*.yaml`. Method/backend checkpoint
settings live in `configs/experiments/clean_baseline.yaml`.

DexYCB official S1 uses a guarded manifest workflow:

```bash
# CPU: audits official toolkit + BOP S1 + the prior copy report, then exports
# train27k/smoke and S1 val. The raw root is read only.
python scripts/prepare_dexycb.py --raw-root /data/wentao/datasets/dexycb \
  --output-root /data/wentao/ropetrack/processed/dexycb/s1_v1 \
  --audit-summary /data/wentao/ropetrack/runs/handdata_audit_20260719/dexycb/summary.json \
  --toolkit-root .local_checks/dex-ycb-toolkit \
  --bop-manifest /data/wentao/datasets/dexycb/bop/s1/test_targets_bop19.json \
  --splits train,val

# Test export is deliberately impossible until the matched checkpoints,
# visibility thresholds, and evaluation recipe are frozen.
python scripts/prepare_dexycb.py <same audit arguments> --splits test \
  --test-freeze-file <run>/protocol/recipe_freeze.json
```

`validate_dexycb_coordinates.py` independently checks pinhole reprojection and
the official manopth PCA45 decode. `score_dexycb.py` reports PA,
root-relative, camera-frame, orientation/translation diagnostics, fixed
subject/camera/visibility slices, and episode-bootstrap paired deltas.
`verify_dexycb_artifacts.py` compares pre/post raw-tree signatures and closes
the no-leak/artifact gate.

Do not add empty script files. Add each script when it can run against local
data or a tiny fixture.
