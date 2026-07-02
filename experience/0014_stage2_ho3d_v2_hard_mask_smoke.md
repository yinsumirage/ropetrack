# 0014 Stage 2 HO3D v2 Hard Mask Smoke

Date: 2026-07-03

## Purpose

Start Stage 2 with the smallest end-to-end hard-image smoke:

- Generate a new hard HO3D v2 mini root without touching the source dataset.
- Keep GT bboxes unchanged.
- Run AnyHand-WiLoR on clean vs hard images.
- Evaluate with the same parallel evaluator.

## Local Code

Added:

```text
scripts/make_hard_images.py
tests/test_make_hard_images.py
docs/2026-07-03-stage2-hard-splits-plan.md
```

Fixed:

```text
scripts/bench_ho3d.py
scripts/bench_freihand.py
```

Limited benchmark runs now write only the matching GT subset into `eval_input`.
This fixed a smoke eval failure where 32 predictions were paired with the full
11524-sample HO3D v2 GT files.

## Generated Data

Weak first attempt:

```text
/data/wentao/ropetrack/hard/ho3d_v2/mask_s1_32_20260703
effect: mask
severity: 0.45
samples: 32
CPU job: 162338
```

Stronger second attempt:

```text
/data/wentao/ropetrack/hard/ho3d_v2/mask_s2_32_20260703
effect: mask
severity: 0.85
samples: 32
CPU job: 162354
```

Both hard roots include:

```text
evaluation/
evaluation.txt
evaluation_xyz.json
evaluation_verts.json
hard_manifest.jsonl
```

The generator uses the clean full-run `run_meta.json` sample order to avoid the
slow HO3D v2 root-order inference path.

## Jobs

Weak mask smoke:

```text
GPU pred: 162339 completed, 32 clean + 32 hard samples, 0 failures
CPU eval: 162353 completed after fixing limited GT subset
run root: /data/wentao/ropetrack/runs/stage2_ho3d_v2_mask_s1_32_20260703
```

Severe mask smoke:

```text
GPU pred: 162355 completed, 32 hard samples, 0 failures
CPU eval: 162356 completed
run root: /data/wentao/ropetrack/runs/stage2_ho3d_v2_mask_s2_32_20260703
```

Cancelled:

```text
CPU job 162335
```

Reason: the first generator version used HO3D v2 root-order inference even for
`--limit 32`, which was too slow. Use `--sample-order-file` for HO3D v2.

## Scores

HO3D evaluator writes PA means in centimetres; the table reports millimetres.

| Split | AUCj | PA-MPJPE mm | AUCv | PA-MPVPE mm | F@5 | F@15 |
|---|---:|---:|---:|---:|---:|---:|
| clean32 | 0.894616 | 5.273 | 0.896157 | 5.193 | 0.792745 | 0.999940 |
| mask_s1_32 severity 0.45 | 0.896344 | 5.176 | 0.898071 | 5.095 | 0.802307 | 0.999920 |
| mask_s2_32 severity 0.85 | 0.845779 | 7.739 | 0.845318 | 7.741 | 0.619072 | 0.973316 |

## Interpretation

The weak mask is not a useful hard split. On 32 samples it did not degrade the
model, so severity `0.45` should not be used as evidence.

The severe mask is a useful first smoke: PA-MPJPE worsened by about `2.47 mm`,
PA-MPVPE worsened by about `2.55 mm`, and F@5 dropped by about `0.174`.

This proves the hard-root generation and GT-bbox evaluation chain can expose a
model weakness. It does not yet prove the final hard benchmark design; sample
count is too small and severity may be too aggressive.

## Next

1. Generate 256 or 512 samples for HO3D v2 `mask_s2` and run AnyHand-WiLoR.
2. Add one more perturbation type, likely `crop`, to avoid relying on one mask
   geometry.
3. Repeat on FreiHAND after the HO3D v2 hard path is stable.
4. Only after hard degradation is stable, generate rope-distance labels.
