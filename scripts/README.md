# Scripts

Current benchmark entrypoints:

- `eval.py`: config-driven benchmark export and eval entrypoint.
- `eval_parallel.py`: local evaluator used by benchmark exports.
- `make_hard_images.py`: hard-image split generator.
- `make_rope_labels.py`: GT fingertip-to-wrist rope label generator.
- `rope_diagnostics/score_rope_predictions.py`: rope diagnostic scorer for exported `pred.json`.
- `rope_diagnostics/visualize_mesh_comparison.py`: mesh/prediction visual check helper.

Typical usage:

```powershell
python scripts\eval.py --dataset ho3d_v2 --method wilor_anyhand --run-eval
```

Hard split roots are generated as normal dataset roots and then selected by
dataset config name:

```bash
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/mask70 --effect mask --severity 0.70 --limit 0
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/tip_square80 --effect tip_square --severity 0.80 --limit 0
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/finger_end80 --effect finger_end --severity 0.80 --limit 0
python scripts/eval.py --dataset ho3d_v2_mask70 --method wilor_anyhand --run-eval
```

Rope labels are JSONL so each row keeps sample id, raw distance, chain length,
normalized value, validity, and normalization metadata:

```bash
python scripts/make_rope_labels.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output /data/wentao/ropetrack/rope/ho3d_v2_rope.jsonl --viz-dir /data/wentao/ropetrack/runs/rope_viz/ho3d_v2 --viz-count 16
python scripts/rope_diagnostics/score_rope_predictions.py /data/wentao/ropetrack/runs/clean_baseline/ho3d_v2_wilor_original/eval_input /data/wentao/ropetrack/rope/ho3d_v2_rope.jsonl /data/wentao/ropetrack/runs/rope_scores/ho3d_v2_wilor_original --dataset ho3d --run-meta /data/wentao/ropetrack/runs/clean_baseline/ho3d_v2_wilor_original/run_meta.json
python scripts/rope_diagnostics/analyze_rope_errors.py /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/scores /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/diagnostics
```

First cached FreiHAND rope-refiner scaffold:

```bash
python scripts/make_hard_images.py --dataset freihand --split training --input-root /data/wentao/ropetrack/FreiHAND --output-root /data/wentao/ropetrack/hard/freihand_train/mask45 --effect mask --severity 0.45 --limit 0
python scripts/make_rope_labels.py --dataset freihand --split training --input-root /data/wentao/ropetrack/FreiHAND --output /data/wentao/ropetrack/rope/freihand_training_rope.jsonl
python scripts/rope_refiner/build_freihand_refiner_cache.py --input-root /data/wentao/ropetrack/FreiHAND --rope-labels /data/wentao/ropetrack/rope/freihand_training_rope.jsonl --pred-dir /data/wentao/ropetrack/runs/freihand_train_baseline/eval_input --run-meta /data/wentao/ropetrack/runs/freihand_train_baseline/run_meta.json --output /data/wentao/ropetrack/runs/refiner_cache/freihand_training.npz
```

`build_freihand_refiner_cache.py` currently defaults `--base-hand-pose-source target`,
so `base_hand_pose` is copied from GT MANO pose. Replace that path when baseline
MANO pose export exists.

`rope_refiner/apply_rope_refinement.py` has two modes:

- `--mode checkpoint`: apply the exploratory MLP refiner checkpoint.
- `--mode optimize`: run per-sample rope optimization without training.

Dataset roots live in `configs/datasets/*.yaml`. Method/backend checkpoint
settings live in `configs/experiments/clean_baseline.yaml`.

Do not add empty script files. Add each script when it can run against local
data or a tiny fixture.
