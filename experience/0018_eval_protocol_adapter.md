# 0018 Eval Protocol Adapter

Date: 2026-07-03

## Purpose

Separate dataset-specific evaluation protocol details from backend inference.
HO3D and FreiHAND keep different camera-axis and joint-order rules, but the
model prediction path should stay shared.

## Change

- Added `ropetrack.eval.protocols` for:
  - dataset name normalization,
  - model-output camera conversion,
  - MANO vertices to benchmark joint order.
- Added `scripts/eval.py` as the unified entrypoint.
- Moved dataset loading, bbox collection, batched backend inference, prediction
  formatting, and export metadata into `ropetrack.eval`.
- Reused existing `configs/datasets/*.yaml` for dataset roots and protocol
  defaults.
- Expanded `configs/experiments/clean_baseline.yaml` with runnable
  `wilor_anyhand` and `hamer_anyhand` method definitions.
- Removed the old `scripts/bench_ho3d.py`, `scripts/bench_freihand.py`, and
  `scripts/bench_eval.py` entrypoints.

## Joint Policy

HaMeR official eval exports FreiHAND with model `pred_keypoints_3d` directly,
and only applies a special reorder for HO3D. To preserve that behavior:

- FreiHAND config defaults to `joint_source: model_keypoints`.
- HO3D v2/v3 config defaults to `joint_source: mano_vertices`.

## Current Scope

This is still an eval-side adapter, not a full training data layer. It preserves
the existing GT-bbox benchmark paths and does not change backend loading,
detection, batching, or JSON export shape.

Example:

```powershell
python scripts\eval.py --dataset ho3d_v2 --method wilor_anyhand --run-eval
```

## Verification

```powershell
python -m unittest tests.test_eval_config tests.test_eval_datasets tests.test_eval_pipeline tests.test_protocols
python -m unittest discover -s tests
```

Result:

```text
Targeted eval tests passed.
Full local tests passed.
```
