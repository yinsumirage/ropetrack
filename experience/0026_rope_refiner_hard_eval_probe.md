# 0026 Rope Refiner Hard Eval Probe

Date: 2026-07-05

## Purpose

Apply the full-training cached rope refiners to FreiHAND `mask70` hard
evaluation and score held-out benchmark metrics.

This run uses the evaluation rope label as the simulated rope input:

`/data/wentao/ropetrack/rope_labels/freihand/evaluation_rope.jsonl`

## Code

Added:

- `scripts/rope_refiner/apply_rope_refinement.py`
- `tests/test_apply_rope_refinement.py`

The apply script writes both:

- `base_pred.json`: MANO-cache pose through the same MANO vertices/joints path.
- `pred.json`: refined pose through the same MANO vertices/joints path.

This makes base vs refined a same-decoder comparison.

## Jobs

First GPU apply attempt:

- Job `166876`
- Failed because MANO was called with 45-dim axis-angle hand pose while the
  layer expected rotation matrices.

Second GPU apply attempt:

- Job `166882`
- Still failed because `use_pca=False` alone was not enough for this smplx path.

Final GPU apply:

- Job `166890`
- Completed with exit code `0:0`
- Runtime: `00:05:25`

CPU scoring:

- Job `166891`
- Completed with exit code `0:0`
- Runtime: `00:09:26`

Run root:

`/data/wentao/ropetrack/runs/rope_refiner_eval_20260705`

## Fix

Before calling MANO, convert cached axis-angle pose to rotation matrices:

- `base_global_orient`: `[N, 3] -> [N, 1, 3, 3]`
- `hand_pose`: `[N, 45] -> [N, 15, 3, 3]`
- call MANO with `pose2rot=False`

## Results

WiLoR same-decoder hard-eval comparison:

| Metric | Base | Refined | Delta |
|---|---:|---:|---:|
| `xyz_procrustes_al_mean3d` | 1.0068 | 1.0975 | +0.0908 |
| `mesh_al_mean3d` | 1.0051 | 1.1043 | +0.0993 |
| `f_al_score_5` | 0.5805 | 0.5082 | -0.0722 |
| `f_al_score_15` | 0.9426 | 0.9336 | -0.0089 |

HaMeR same-decoder hard-eval comparison:

| Metric | Base | Refined | Delta |
|---|---:|---:|---:|
| `xyz_procrustes_al_mean3d` | 1.0824 | 1.2014 | +0.1191 |
| `mesh_al_mean3d` | 1.0852 | 1.2156 | +0.1304 |
| `f_al_score_5` | 0.5377 | 0.4558 | -0.0819 |
| `f_al_score_15` | 0.9342 | 0.9181 | -0.0160 |

Full summary:

`/data/wentao/ropetrack/runs/rope_refiner_eval_20260705/summary_scores.json`

## Interpretation

The refiner fits the full hard-training cache but hurts held-out hard eval.

The likely issue is not that the eval pipeline is disconnected; base and refined
use the same MANO decoder and scoring path. The issue is that the current MLP
learns a training-set pose correction that does not generalize.

Next gates:

- add validation split and early stopping;
- lower LR/steps or add stronger delta regularization;
- add rope noise during training;
- compare against a zero-rope or shuffled-rope refiner;
- only then consider a larger model.
