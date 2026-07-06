# 0028 Rope Optimization Cross-Split And HO3D Probe

Date: 2026-07-05

## Purpose

Continue the non-learning rope optimization probe from `0027`:

- test FreiHAND hard splits beyond `mask70`;
- add HO3D v2 hard eval support to the same optimization path;
- confirm whether the small FreiHAND `mask70` gain generalizes across
  perturbations and datasets.

This still does not train a network and does not use GT MANO parameters.

## Method

The apply script now supports both datasets:

`scripts/rope_refiner/apply_rope_refinement.py`

Relevant mode:

`--mode optimize --dataset freihand|ho3d`

The optimization variable remains five per-sample finger curl scalars, one per
finger. The loss is rope-only:

`mean((pred_rope_norm - label_rope_norm)^2) + alpha_l2 * mean(alpha^2)`

MANO is used only as the differentiable geometry layer that maps the adjusted
pose to joints/vertices so the predicted rope lengths can be recomputed. Eval
GT joints/vertices are not used during optimization.

Hyperparameters for this probe:

- `steps=120`
- `lr=2.0`
- `alpha_l2=0.001`
- `max_alpha=0.5`

## Jobs

FreiHAND extra hard splits:

- GPU apply: `166978`
- CPU score: `166979`
- Run root:
  `/data/wentao/ropetrack/runs/rope_optimization_freihand_extra_20260705`
- Splits: `tip_square80`, `finger_end80`

HO3D v2 `mask70`:

- First GPU apply: `166980`, failed because the hard root protocol precheck
  tried to use the clean protocol assumptions.
- Fixed by disabling protocol precheck for this hard-root apply job with
  `--protocol-check-samples 0`.
- GPU apply retry: `166987`
- CPU score: `166988`
- Run root:
  `/data/wentao/ropetrack/runs/rope_optimization_ho3d_v2_mask70_20260705`

## FreiHAND Results

### `tip_square80`

| Backend | Metric | Base | Optimized | Delta |
|---|---|---:|---:|---:|
| WiLoR | `xyz_procrustes_al_mean3d` | 0.6800 | 0.6700 | -0.0100 |
| WiLoR | `mesh_al_mean3d` | 0.6887 | 0.6791 | -0.0096 |
| WiLoR | `f_al_score_5` | 0.7133 | 0.7180 | +0.0047 |
| HaMeR | `xyz_procrustes_al_mean3d` | 0.7890 | 0.7777 | -0.0112 |
| HaMeR | `mesh_al_mean3d` | 0.7958 | 0.7850 | -0.0108 |
| HaMeR | `f_al_score_5` | 0.6500 | 0.6547 | +0.0047 |

### `finger_end80`

| Backend | Metric | Base | Optimized | Delta |
|---|---|---:|---:|---:|
| WiLoR | `xyz_procrustes_al_mean3d` | 0.9977 | 0.9820 | -0.0156 |
| WiLoR | `mesh_al_mean3d` | 0.9974 | 0.9821 | -0.0152 |
| WiLoR | `f_al_score_5` | 0.5686 | 0.5730 | +0.0044 |
| HaMeR | `xyz_procrustes_al_mean3d` | 1.1129 | 1.0962 | -0.0167 |
| HaMeR | `mesh_al_mean3d` | 1.1114 | 1.0952 | -0.0162 |
| HaMeR | `f_al_score_5` | 0.5233 | 0.5276 | +0.0043 |

Full summary:

`/data/wentao/ropetrack/runs/rope_optimization_freihand_extra_20260705/summary_scores.json`

## HO3D v2 `mask70` Results

| Backend | Metric | Base | Optimized | Delta |
|---|---|---:|---:|---:|
| WiLoR | `xyz_procrustes_al_mean3d` | 0.9436 | 0.9388 | -0.0049 |
| WiLoR | `mesh_al_mean3d` | 0.9637 | 0.9590 | -0.0048 |
| WiLoR | `f_al_score_5` | 0.5305 | 0.5319 | +0.0013 |
| HaMeR | `xyz_procrustes_al_mean3d` | 0.9322 | 0.9260 | -0.0062 |
| HaMeR | `mesh_al_mean3d` | 0.9626 | 0.9573 | -0.0053 |
| HaMeR | `f_al_score_5` | 0.5232 | 0.5251 | +0.0019 |

Full summary:

`/data/wentao/ropetrack/runs/rope_optimization_ho3d_v2_mask70_20260705/summary_scores.json`

## Interpretation

The rope optimization signal is now positive across:

- FreiHAND `mask70`;
- FreiHAND `tip_square80`;
- FreiHAND `finger_end80`;
- HO3D v2 `mask70`;
- both WiLoR and HaMeR cached predictions.

The gains are small, but the direction is consistent. This supports continuing
with rope as a real geometric constraint, while keeping the correction
constrained and interpretable.

Important caveats:

- HO3D raw absolute metrics are not reliable in this run; raw `xyz_mean3d` and
  `mesh_mean3d` remain very large, so the immediate conclusion should use
  aligned metrics only.
- The current optimization changes pose only through five curl scalars. This is
  intentionally limited and cannot fix arbitrary MANO pose errors.
- The first learned MLP refiner worsened held-out eval, so the next learned
  version should be guided by this optimization behavior rather than expanded
  blindly.
