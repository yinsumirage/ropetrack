# 0027 Rope Optimization Probe

Date: 2026-07-05

## Purpose

Test a non-learning rope correction after the 45-dim MLP refiner overfit:

- no GT MANO supervision;
- no network training;
- optimize only five per-sample finger curl scalars;
- use rope loss plus regularization;
- score on FreiHAND `mask70` hard evaluation.

## Method

Added an optimization mode to:

`scripts/rope_refiner/apply_rope_refinement.py`

The optimized pose is:

`hand_pose = base_hand_pose + alpha[finger] * base_finger_pose`

where `alpha` has shape `[N, 5]`.

Finger groups:

- thumb: MANO pose triplets `[12, 13, 14]`
- index: `[0, 1, 2]`
- middle: `[3, 4, 5]`
- ring: `[9, 10, 11]`
- pinky: `[6, 7, 8]`

The objective is rope-only:

`rope_loss + alpha_l2 * mean(alpha^2)`

No eval GT joints/verts are used during optimization.

## Jobs

Conservative run:

- GPU apply: `166939`
- CPU score: `166940`
- Run root: `/data/wentao/ropetrack/runs/rope_optimization_eval_20260705`
- Hyperparameters: `steps=80`, `lr=0.05`, `alpha_l2=0.1`, `max_alpha=0.25`

Aggressive run:

- GPU apply: `166950`
- CPU score: `166951`
- Run root: `/data/wentao/ropetrack/runs/rope_optimization_eval_aggressive_20260705`
- Hyperparameters: `steps=120`, `lr=2.0`, `alpha_l2=0.001`, `max_alpha=0.5`

## Conservative Result

The conservative run barely moved:

- mean `|alpha|`: about `2.5e-5`
- metrics changed only at numerical-noise scale.

## Aggressive Result

Aggressive alpha stats:

- WiLoR mean `|alpha|`: `0.0059`, max `|alpha|`: `0.1426`
- HaMeR mean `|alpha|`: `0.0057`, max `|alpha|`: `0.1125`

WiLoR hard-eval result:

| Metric | Base | Optimized | Delta |
|---|---:|---:|---:|
| `xyz_procrustes_al_mean3d` | 1.0068 | 0.9906 | -0.0162 |
| `mesh_al_mean3d` | 1.0051 | 0.9893 | -0.0158 |
| `f_al_score_5` | 0.5805 | 0.5848 | +0.0044 |
| `f_al_score_15` | 0.9426 | 0.9438 | +0.0013 |

HaMeR hard-eval result:

| Metric | Base | Optimized | Delta |
|---|---:|---:|---:|
| `xyz_procrustes_al_mean3d` | 1.0824 | 1.0681 | -0.0143 |
| `mesh_al_mean3d` | 1.0852 | 1.0712 | -0.0140 |
| `f_al_score_5` | 0.5377 | 0.5418 | +0.0041 |
| `f_al_score_15` | 0.9342 | 0.9356 | +0.0014 |

Full summary:

`/data/wentao/ropetrack/runs/rope_optimization_eval_aggressive_20260705/summary_scores.json`

## Interpretation

The optimization probe is much more promising than the learned 45-dim MLP:

- the MLP made held-out hard eval worse;
- constrained per-sample optimization gives small but consistent improvements;
- no GT joints/verts are used during optimization, only rope labels and MANO
  cache;
- the correction remains small and interpretable.

This is still a small gain. The next useful work is to tune the optimization
objective and test on more hard splits before building a learned module.
