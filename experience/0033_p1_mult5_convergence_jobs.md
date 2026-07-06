# 0033 P1 Mult5 Rope Convergence Jobs

Date: 2026-07-07

## Purpose

Test whether the low rope residual closure in `experience/0032` is partly just
under-optimization from loss-scale mismatch.

This is a zero-code hyperparameter scan on FreiHAND `mask70`, WiLoR,
`rope/mult5`.

## Run Root

```text
/data/wentao/ropetrack/runs/rope_p1_mult5_convergence_20260707_034547
```

Job manifest:

```text
/data/wentao/ropetrack/runs/rope_p1_mult5_convergence_20260707_034547/jobs.tsv
```

## Inputs

- Hard root: `/data/wentao/ropetrack/hard/freihand/mask70`
- Rope labels: `/data/wentao/ropetrack/runs/rope_phase12_20260705_031056/labels/freihand_rope.jsonl`
- Export/cache: `/data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932/exports/mask70_wilor`

## Grid

Fixed:

- objective: `rope`
- action space: `mult5`
- max alpha: `0.5`

Sweep:

- `lr`: `2`, `8`, `32`
- `steps`: `120`, `400`
- `alpha_l2`: `0.001`, `0`

## Jobs

| Tag | Apply | Score |
|---|---:|---:|
| `lr2_s120_l2_0p001` | 169048 | 169049 |
| `lr2_s120_l2_0` | 169050 | 169051 |
| `lr2_s400_l2_0p001` | 169052 | 169053 |
| `lr2_s400_l2_0` | 169054 | 169055 |
| `lr8_s120_l2_0p001` | 169056 | 169057 |
| `lr8_s120_l2_0` | 169058 | 169059 |
| `lr8_s400_l2_0p001` | 169060 | 169061 |
| `lr8_s400_l2_0` | 169062 | 169063 |
| `lr32_s120_l2_0p001` | 169064 | 169065 |
| `lr32_s120_l2_0` | 169066 | 169067 |
| `lr32_s400_l2_0p001` | 169068 | 169069 |
| `lr32_s400_l2_0` | 169070 | 169071 |

## Initial Queue Check

Submission succeeded. All GPU apply jobs were pending on priority; CPU score
jobs were pending on their apply-job dependencies.
