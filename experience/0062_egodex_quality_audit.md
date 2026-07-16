# EgoDex test annotation-quality audit

Date: 2026-07-17

## Scope and evidence boundary

This audits all 162,191 exported hand rows from 3,243 EgoDex test episodes at
stride 10. It checks native ARKit confidence, camera validity, adjacent-frame
kinematics, bone-length stability, qualitative 2D overlays, and WiLoR/refiner
PA joint error by confidence. It is not an independent mocap validation.

Official evidence:

- The [EgoDex paper](https://arxiv.org/abs/2505.11709) describes 829 hours,
  90M frames, 338k demonstrations, and 25 tracked hand joints per hand.
- The annotations are production-model predictions from Vision Pro/ARKit, not
  manual, mocap, or native MANO labels. The paper explicitly says they can be
  imperfect under heavy occlusion and very high-speed motion.
- The official benchmark evaluates wrist plus five fingertips per hand rather
  than treating every internal joint or a MANO mesh as validated GT.
- The [official repository](https://github.com/apple/ml-egodex) says zero
  confidence means fully occluded or otherwise undetected, some HDF5 files
  have no confidence group, and RGB reprojection can be offset because the
  passthrough image is synthesized from multiple cameras.
- The official README lists H-RDT as using EgoDex for pretraining and Being-H0
  as post-processing the annotations with MANO. This is downstream conversion,
  not evidence that EgoDex contains native MANO ground truth.

No source found an independent per-joint mocap-versus-ARKit error table for
EgoDex. Claims such as "precise" are therefore author claims, while the
quantities below are our own internal audit.

## Run

```text
script: scripts/audit_egodex_quality.py
final Slurm job: 183644 (COMPLETED, 00:01:51, exit 0:0)
output: /data/wentao/ropetrack/runs/egodex_quality_audit_20260717_v2
marker: SUCCESS
```

Job 183636 produced the first pass but mixed HDF5 files without native
confidence into the high-confidence bin because the exporter uses all-one
compatibility placeholders. Job 183644 explicitly separates those rows and
supersedes that confidence analysis.

## Native-confidence coverage

| Check | Result |
|---|---:|
| HDF5 files without `confidences` | 454 |
| rows without native confidence | 15,128 / 162,191 (9.33%) |
| individual joints below 0.25 | 7.55% |
| individual joints below 0.50 | 27.45% |
| fingertips below 0.25 | 18.62% |
| fingertips below 0.50 | 50.97% |

The all-one rows are exactly the rows lacking native confidence; no row with a
real confidence group has all 21 values equal to one. Do not interpret the
current exporter's all-one fallback as high confidence.

The lowest-confidence images are not merely slightly noisy. The selected
zero-confidence rows include hands outside the field of view, heavy
object/hand occlusion, and transforms whose projected skeleton is far away
from the visible hand. Some zero-confidence rows still look plausible, so zero
is an uncertainty/failure flag rather than a deterministic geometric-error
measurement. Either way, it is unsuitable as equal-weight GT.

High-confidence examples generally have a plausible hand skeleton. A small 2D
offset remains common and cannot alone prove bad 3D because Apple documents the
multi-camera passthrough perspective mismatch.

## Geometry and temporal checks

| Check | Result |
|---|---:|
| non-finite 3D rows | 0 |
| any joint behind camera | 1.49% |
| adjacent stride-10 pairs | 155,742 |
| max wrist-relative joint speed, p50 / p95 / p99 | 0.069 / 0.319 / 0.457 m/s |
| pairs above 0.5 m/s | 0.568% |
| maximum adjacent bone-length change | 5.29% |
| pairs above 10% bone-length change | 0 |

This rejects the idea that EgoDex is an unconstrained collection of 3D points.
Although it is not MANO, its ARKit skeleton has extremely stable link lengths
over adjacent frames. The worst speed examples are mostly visible fast motion
or hand entry/exit, with occasional tracking/projection failure. The dominant
quality issue is visibility/confidence, not widespread temporal bone stretching.

## Error by fingertip confidence

All values are PA joint error in mm. Missing confidence is kept separate.

| Mean fingertip confidence | Rows | WiLoR base | release student delta | rope flex15 teacher delta |
|---|---:|---:|---:|---:|
| `[0.00, 0.25)` | 9,881 | 14.138 | -1.186 | -1.680 |
| `[0.25, 0.50)` | 69,980 | 11.615 | +0.213 | -0.536 |
| `[0.50, 0.75)` | 60,461 | 10.211 | +0.815 | +0.057 |
| `[0.75, 0.90)` | 6,263 | 9.647 | +0.890 | +0.023 |
| `[0.90, 1.00]` | 478 | 8.922 | +0.982 | +0.206 |
| missing | 15,128 | 11.680 | +0.252 | -0.483 |

Among rows with native confidence, fingertip confidence versus base PA error
has correlation -0.366. This is strong enough to make confidence a required
evaluation/training field, but not strong enough to certify individual labels.

The release student is not uniformly bad: it improves the worst-confidence
bin, then over-corrects every better-confidence bin. That explains its worse
aggregate score. The iterative teacher has the same pattern more mildly: large
gains where labels/base predictions are hard, essentially no gain above 0.5,
and slight degradation at the highest confidence. Improving rope consistency
does not guarantee closer agreement with an already-good native skeleton.

## Decision

EgoDex is useful for large-scale temporal/manipulation pretraining, skeleton or
wrist-tip supervision, and robustness data. It should not be treated as strict
MANO/mesh ground truth or as an unfiltered 21-joint benchmark.

Before training on the five parts:

1. Preserve `confidence_available`; never encode missing confidence as true 1.
2. Exclude zero-confidence joints and use confidence-weighted joint loss.
3. Report native-confidence strata separately; use wrist+tips as the first
   protocol because that matches the official benchmark's validated use.
4. If MANO is needed, fit it only as a derived pseudo-label with a robust loss,
   store the fit residual, and reject poor fits. Never call it EgoDex MANO GT.
5. Do not deploy the current release MLP zero-shot. A future gate must use an
   observable deployment signal; native ARKit confidence is only available as
   an offline audit/training signal.

