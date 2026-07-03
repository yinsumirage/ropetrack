# 0016 Stage 2 HO3D v2 Fingertip Fix 512

Date: 2026-07-03

## Purpose

Run a corrected 512-sample fingertip perturbation check after visual validation
of the HO3D fingertip projection.

Only two variants were tested:

- `tip_square085_fix_512`
- `tip_blur085_fix_512`

No circle variant was tested.

## Inputs

All runs use AnyHand-WiLoR, GT bbox, and the same first 512 HO3D v2 samples as
the previous clean/mask 512 run.

Corrected hard roots:

```text
/data/wentao/ropetrack/hard/ho3d_v2/tip_square085_fix_512_20260703
/data/wentao/ropetrack/hard/ho3d_v2/tip_blur085_fix_512_20260703
```

Both generated roots reported:

```text
manifest 512
eval_txt 512
with_points 512
```

Run root:

```text
/data/wentao/ropetrack/runs/stage2_ho3d_v2_tipfix_512_20260703
```

## Jobs

```text
generation: 162621, cpu, elapsed 00:05:36, exit 0:0
gpu pred:   162622, gpu, elapsed 00:05:25, exit 0:0
cpu eval:   162623, cpu, elapsed 00:00:39, exit 0:0
```

GPU prediction:

```text
tip_square085_fix_512 samples 512 failures 0 candidates 512
tip_blur085_fix_512 samples 512 failures 0 candidates 512
```

## Scores

HO3D evaluator writes PA means in centimetres; the table reports millimetres.

| Split | AUCj | PA-MPJPE mm | AUCv | PA-MPVPE mm | F@5 | F@15 | Delta PA-MPJPE | Delta F@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean512 | 0.876891 | 6.154 | 0.878707 | 6.065 | 0.736867 | 0.994387 | 0.000 | 0.000000 |
| tip_square085_fix_512 | 0.863734 | 6.814 | 0.865024 | 6.749 | 0.708076 | 0.984857 | +0.660 | -0.028791 |
| tip_blur085_fix_512 | 0.867508 | 6.625 | 0.867272 | 6.636 | 0.722306 | 0.986413 | +0.471 | -0.014561 |

For comparison, the earlier centered `mask085_512` scored:

```text
PA-MPJPE 8.351 mm
F@5 0.593193
Delta PA-MPJPE +2.197 mm
Delta F@5 -0.143674
```

## Takeaway

After fixing the projection, fingertip perturbations are no longer a no-op.
`tip_square085` is stronger than `tip_blur085`, but both are much milder than
the centered `mask085` split.

Use `tip_square085` if the goal is a precise fingertip-targeted ablation. Use
`mask085` or a larger finger-part occlusion if the goal is a strong hard split.
