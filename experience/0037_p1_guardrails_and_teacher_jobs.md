# 0037 P1 Guardrails, HO3D Winner, And Train-Teacher Jobs

Date: 2026-07-07

## Code Sync

Claude added rope-sensor perturbation flags to
`scripts/rope_refiner/apply_rope_refinement.py`:

```text
--rope-noise-std
--rope-dropout
--rope-noise-seed
```

Semantics: noise/dropout perturb `input_rope_norm` and `rope_valid`; both the
loss and gate consume the perturbed reading, while `gt_rope_norm` stays clean.

Local verification:

```text
python -m pytest tests/test_apply_rope_refinement.py tests/test_refine_actions.py -q
39 passed

python -m pytest tests -q
159 passed, 4 warnings
```

Code commit synced to HPC:

```text
8e1ecc7 Add rope sensor noise ablation
```

Also corrected 0036's interpretation: the default-recipe oracle rows are not a
valid ceiling after P1 selected a much stronger optimizer.

## Batch 1: Guardrails

Run root:

```text
/data/wentao/ropetrack/runs/rope_p1_guardrails_20260707_120955
```

Purpose: report guardrails before P2 training.

Cells:

- A noise ablation on the Batch B winner (`rope/flex15/gate010`) for
  FreiHAND mask70:
  `noise_std in {0.05, 0.1, 0.2}` x `dropout in {0, 0.2}`.
  The clean `noise=0/dropout=0` baseline is reused from Batch B.
- B strong oracle ceiling:
  `oracle_tip x {mult5, flex15} x {mask70, finger_end80}`.
- C clean FreiHAND eval split:
  `rope/flex15 x {gateoff, gate010}`.
- D HaMeR backend:
  `rope/flex15/gate010 x {mask70, finger_end80}`.

Implementation note: clean WiLoR export and HaMeR mask70 export were not
available as MANO caches, so the GPU job creates them before running the
corresponding apply cells. This is input preparation, not an extra result cell.

Jobs:

| Group | Kind | Job | Dependency |
|---|---|---:|---:|
| A/C/D | apply GPU | 169719 | - |
| A/C/D | score CPU | 169720 | 169719 |
| B | apply GPU | 169721 | - |
| B | score CPU | 169722 | 169721 |

GPU wall time is `02:00:00`; CPU wall time is `04:00:00`.

## Batch 2: HO3D Winner

Run root:

```text
/data/wentao/ropetrack/runs/rope_p1_ho3d_winner_20260707_121314
```

Purpose: apply the same Batch B winner recipe to HO3D v2 mask70 without
retuning hyperparameters.

Recipe:

```text
rope + flex15 + gate010 + steps=400/lr=32/alpha_l2=0.001
```

Backends:

```text
WiLoR, HaMeR
```

Jobs:

| Kind | Job | Dependency |
|---|---:|---:|
| apply GPU | 169738 | - |
| score CPU | 169739 | 169738 |

## Batch 3: FreiHAND Train Teacher

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_train_teacher_20260707_121353
```

Purpose: prepare the P2 distillation target on the FreiHAND mask70 training
split using the Batch B winner recipe.

Training root:

```text
/data/wentao/ropetrack/hard/freihand/mask70_wilor_training
```

Job:

| Kind | Job | Dependency |
|---|---:|---:|
| export + teacher GPU | 169740 | - |

This job first exports WiLoR MANO cache on the hard training split, then runs
`apply_rope_refinement.py` with the Batch B winner recipe. It requests
`04:00:00`.

## P2 Gate

Proceed to rope-conditioned head training only if all three checks pass:

1. HO3D winner supports the FreiHAND conclusion without retuning.
2. Train-split teacher artifacts are produced successfully.
3. The `noise_std=0.05` guardrail retains at least 60% of the clean Batch B
   winner gain.
