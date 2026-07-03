# 0015 Stage 2 HO3D v2 Hard Mask 512

Date: 2026-07-03

## Purpose

Scale the first hard-mask smoke from 32 to 512 HO3D v2 samples and compare:

- clean512
- mask045_512
- mask085_512

All runs use AnyHand-WiLoR, GT bbox, and the same first 512 samples from the
clean full-run sample order.

## Image Check

For sample `SM1/0000`, the hard generator used this GT bbox:

```text
[289.52545166015625, 215.35614013671875, 429.73553466796875, 369.8903503417969]
```

Measured inside that bbox:

```text
mask 0.45 black/changed fraction: 0.2094
mask 0.85 black/changed fraction: 0.7355
```

So `mask 0.85` is being written correctly. It blacks out most of the bbox
center, but still leaves context, object silhouette, and part of the hand.

## Jobs

Hard root generation:

```text
job: 162487
partition: cpu
elapsed: 00:04:44
outputs:
  /data/wentao/ropetrack/hard/ho3d_v2/mask_s1_512_20260703
  /data/wentao/ropetrack/hard/ho3d_v2/mask_s2_512_20260703
```

GPU prediction:

```text
job: 162488
partition: gpu
elapsed: 00:06:27
node: server06
run root: /data/wentao/ropetrack/runs/stage2_ho3d_v2_mask_512_20260703
clean512: 512 samples, 0 failures
mask045_512: 512 samples, 0 failures
mask085_512: 512 samples, 0 failures
```

CPU eval:

```text
job: 162489
partition: cpu
elapsed: 00:00:50
workers: 8
```

Note: eval is fast now, but keep it on CPU jobs instead of using
`bench_ho3d.py --run-eval` inside a GPU allocation, because eval holds no GPU
work.

## Scores

HO3D evaluator writes PA means in centimetres; the table reports millimetres.

| Split | AUCj | PA-MPJPE mm | AUCv | PA-MPVPE mm | F@5 | F@15 |
|---|---:|---:|---:|---:|---:|---:|
| clean512 | 0.876891 | 6.154 | 0.878707 | 6.065 | 0.736867 | 0.994387 |
| mask045_512 | 0.874294 | 6.285 | 0.875916 | 6.204 | 0.724808 | 0.993776 |
| mask085_512 | 0.832953 | 8.351 | 0.832371 | 8.382 | 0.593193 | 0.968929 |

## Interpretation

`mask 0.45` is too weak. It produces only a small drop:

```text
PA-MPJPE +0.13 mm
PA-MPVPE +0.14 mm
F@5 -0.012
```

`mask 0.85` is a real hard split candidate:

```text
PA-MPJPE +2.20 mm
PA-MPVPE +2.32 mm
F@5 -0.144
```

The model still predicts reasonably because the corruption is a centered bbox
mask, not a targeted fingertip or full-hand semantic occlusion. The remaining
image context and object/hand boundaries are still informative.

## Next

Add a targeted fingertip/joint mask or side-occlusion split before claiming
Stage 2 hard benchmark quality. A center bbox mask is enough for pipeline
validation, but it is not the final perturbation design.
