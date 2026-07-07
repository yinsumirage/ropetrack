# 0049 P3 v0 And Report Close Results

Date: 2026-07-07

## Scope

This records the completed batch from `0048`:

- report close: release student on HO3D v2 finger_end80 and apply-level latency;
- P3 v0: pooled WiLoR feature student vs rope-shuffled image-only control.

Run root:

```text
/data/wentao/ropetrack/runs/rope_p3_v0_and_report_close_20260707_222251
```

Local pulled summaries:

```text
.local_checks/remote_results_20260707_p3v0_reportclose/
```

## Job Status

All jobs completed with exit code `0:0`.

| Job | Name | Elapsed |
|---:|---|---:|
| 170579 | p3v0_rep | 00:04:49 |
| 170580 | p3v0_te | 00:03:55 |
| 170581 | p3v0_sc | 00:13:42 |

## Report Close Results

Release model on HO3D v2 finger_end80:

| Cell | PA cm | All-joint delta mm | Occluded-tip delta mm | Closure | Gated fraction |
|---|---:|---:|---:|---:|---:|
| release_ho3d_finger_end80 | 0.836091 | -0.586 | -0.952 | 0.361 | 0.499 |

This is slightly better than the five-teacher multi-v2 result on the same axis
(-0.524 mm all-joint, -0.735 mm occluded-tip), so the `0044` four-teacher
student remains the release model.

Apply-level latency on FreiHAND mask70, 3960 samples:

| Path | Wall time | Per sample |
|---|---:|---:|
| 400-step teacher optimize apply | 110.32 s | 27.9 ms |
| release student apply | 51.66 s | 13.0 ms |

This is an end-to-end apply script timing, not pure MLP latency. It includes
cache loading, action application, and shared MANO decode/output writing. Use it
as a conservative pipeline wall-time number; the isolated student alpha forward
is much smaller.

## P3 v0 Results

Training health:

| Model | epochs | best epoch | best val L1 | zero baseline | in_dim |
|---|---:|---:|---:|---:|---:|
| P3 full | 99 | 78 | 0.014122 | 0.050972 | 1345 |
| P3 image-only (`--shuffle-rope`) | 107 | 86 | 0.047091 | 0.050972 | 1345 |

Eval on FreiHAND mask70:

| Cell | PA cm | All-joint delta mm | Occluded-tip delta mm | Clean-tip delta mm | Closure |
|---|---:|---:|---:|---:|---:|
| p3_head_v0_mask70 | 0.853492 | -1.533 | -4.891 | -2.162 | 0.405 |
| p3_head_v0_imageonly_mask70 | 0.996745 | -0.100 | -0.107 | -0.182 | 0.023 |

Interpretation:

- P3 v0 full is useful, but worse than the rope-only release student
  (-1.533 mm vs about -1.636 mm).
- Image-only is essentially a zero-gain control (-0.100 mm), so pooled WiLoR
  features alone do not repair this hard split.
- The marginal gain of full over image-only is mostly the rope path, not the
  image feature path.
- Do not move P3 into the main report headline. Treat this as a preliminary
  negative v0: pooled global features are too coarse or not aligned enough; the
  next P3 attempt should use token/grid features or a stronger localized image
  representation.

## Report Decision

- P2 remains closed with the `0044` four-teacher augmented multi student as the
  release model.
- Include P3 v0 only as next-plan/preliminary evidence if space permits:
  "cached pooled image features did not improve over rope-only; image-only
  control collapsed, motivating localized/token features."
