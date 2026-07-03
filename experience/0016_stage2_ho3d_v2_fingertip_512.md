# 0016 Stage 2 HO3D v2 Fingertip Perturbation 512

Date: 2026-07-03

## Purpose

Compare localized fingertip perturbations against the earlier centered bbox
mask on the same first 512 HO3D v2 samples.

All runs use AnyHand-WiLoR, GT bbox, and the same clean full-run sample order.

## Generator Fix

The first submitted generation chain was cancelled before GPU inference because
`with_points` was `0/512`: HO3D v2 eval `meta["handJoints3D"]` contains only
the root joint, not 21 joints. The generator now projects fingertip points from
`evaluation_xyz.json` using each frame's `meta["camMat"]`.

Relevant commits:

```text
af1c504 Add fingertip hard image variants
49d71e2 Use HO3D GT joints for fingertip masks
```

## Jobs

Cancelled bad chain:

```text
generation: 162523
gpu pred: 162524
cpu eval: 162525
reason: generated roots had with_points 0/512
```

Completed chain:

```text
generation: 162546
partition: cpu
elapsed: 00:13:34

gpu pred: 162547
partition: gpu
elapsed: 00:17:10
node: server05

cpu eval: 162548
partition: cpu
elapsed: 00:02:09
```

Generated roots all reported `manifest 512`, `eval_txt 512`, and
`with_points 512`.

GPU prediction reported 512 samples, 0 failures, and 512 candidates for every
split.

Run root:

```text
/data/wentao/ropetrack/runs/stage2_ho3d_v2_tip_512_20260703
```

## Scores

HO3D evaluator writes PA means in centimetres; the table reports millimetres.

| Split | AUCj | PA-MPJPE mm | AUCv | PA-MPVPE mm | F@5 | F@15 | Δ PA-MPJPE vs clean | Δ F@5 vs clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean512 | 0.876891 | 6.154 | 0.878707 | 6.065 | 0.736867 | 0.994387 | 0.000 | 0.000000 |
| mask045_512 | 0.874294 | 6.285 | 0.875916 | 6.204 | 0.724808 | 0.993776 | +0.131 | -0.012059 |
| mask085_512 | 0.832953 | 8.351 | 0.832371 | 8.382 | 0.593193 | 0.968929 | +2.197 | -0.143674 |
| tip_circle045_512 | 0.877676 | 6.117 | 0.879625 | 6.019 | 0.741116 | 0.994549 | -0.037 | +0.004249 |
| tip_circle085_512 | 0.875839 | 6.208 | 0.877909 | 6.104 | 0.735371 | 0.994384 | +0.054 | -0.001496 |
| tip_square045_512 | 0.876732 | 6.161 | 0.878731 | 6.063 | 0.738256 | 0.994374 | +0.007 | +0.001389 |
| tip_square085_512 | 0.874337 | 6.284 | 0.876411 | 6.180 | 0.729996 | 0.993728 | +0.129 | -0.006871 |
| tip_blur085_512 | 0.872412 | 6.380 | 0.874534 | 6.273 | 0.724015 | 0.993930 | +0.226 | -0.012852 |
| tip_mixed085_512 | 0.874198 | 6.289 | 0.876372 | 6.181 | 0.730427 | 0.994057 | +0.135 | -0.006440 |

## Image Check

For sample `SM1/0000`, measured changed-pixel fraction versus clean:

```text
tip_circle045: 0.0045
tip_circle085: 0.0170
tip_square045: 0.0059
tip_square085: 0.0210
tip_blur085: 0.0199
tip_mixed085: 0.0199
mask085: 0.0520
```

Local preview, ignored by git:

```text
.local_checks/tip_preview/tip_sheet.png
```

## Takeaway

Current fingertip-only perturbations are too local to be the main hard split.
Even the strongest variant, `tip_blur085`, only adds about 0.226 mm PA-MPJPE
and drops F@5 by about 0.013. The earlier centered `mask085` remains much
stronger, adding about 2.197 mm PA-MPJPE and dropping F@5 by about 0.144.

For the next hard split, use a larger hand-part occlusion instead of isolated
tip dots: fingertip plus distal-finger capsules, random joint-neighborhood
patches, or side/edge occlusion over fingers.
