# 0034 P1 Batch A Flex5/Gating Jobs

Date: 2026-07-07

## Context

Claude added `flex5` and `--gate-residual-threshold`; local review kept the
first HPC follow-up small. Batch A uses the same default optimizer recipe as
0032 so the new cells can be compared directly against the existing P0 table:
`steps=120`, `lr=2.0`, `alpha_l2=0.001`, `max_alpha=0.5`.

Local checks before submission:

```text
python -m pytest tests/test_refine_actions.py tests/test_apply_rope_refinement.py -q
35 passed

python -m pytest tests -q
155 passed, 4 warnings
```

Plain `python -m pytest -q` is not a valid local check because it collects
third-party HaMeR/ViTPose tests under `third_party/` and fails on missing
`mmcv`/`xtcocotools`.

Code commit synced to HPC:

```text
9040b29 Add flex5 and residual gating
```

## Run

Run root:

```text
/data/wentao/ropetrack/runs/rope_p1_batch_a_flex5_gate_20260707_040307
```

Inputs:

```text
mask70 export      /data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932/exports/mask70_wilor
finger_end80 export /data/wentao/ropetrack/runs/rope_optimization_freihand_extra_20260705/finger_end80_wilor/export
rope labels        /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/labels/freihand_rope.jsonl
```

Cells per split:

```text
rope_flex5_gateoff
rope_flex5_gate010
rope_mult5_gate010
oracle_tip_flex5_gateoff
```

Jobs:

| Split | Kind | Job | Dependency |
|---|---|---:|---:|
| mask70 | apply GPU | 169101 | - |
| mask70 | score CPU | 169102 | 169101 |
| finger_end80 | apply GPU | 169103 | - |
| finger_end80 | score CPU | 169104 | 169103 |

Both GPU jobs request `02:00:00`; CPU score jobs request `04:00:00`.

## Readout Rule

For gated cells, first inspect `summary.json`:
`gating.frac_fingers_gated` should be roughly 15%-50%. If it is below 5%,
the threshold is too high; if it is above 80%, the threshold is too low. In
either case the `gate010` cell should be treated as a threshold miss and
re-run with a residual quantile threshold.

Do not interpret Batch A as the final gating result. The formal gating table
waits for the P1-0 convergence scan and should use the selected stronger
optimizer recipe.

## Minor Submission Note

An earlier draft of the submit script accidentally expanded `$cell` in a
generated CPU sbatch heredoc. It submitted one orphan apply job (`169099`),
which was still pending and was cancelled before the successful submission.
