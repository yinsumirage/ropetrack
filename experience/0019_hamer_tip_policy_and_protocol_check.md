# HaMeR Tip Policy And Protocol Check

Date: 2026-07-04

## Finding

HaMeR's MANO wrapper uses `smplx.vertex_ids['mano']` fingertips. In the current
environment that is:

```text
[744, 320, 443, 554, 671]
```

RopeTrack's HO3D `mano_vertices` path was still using the older reproduced
AnyHand/WiLoR HO3D ids:

```text
[744, 333, 444, 555, 672]
```

That is not HaMeR-native, so HO3D `mano_vertices` now uses the HaMeR/SMPL-X tip
ids.

## Loss Scale Note

The large raw HO3D camera-space metrics, around `1108` mm, are not comparable
to HaMeR `Keypoint3DLoss`. HaMeR subtracts the root joint before its 3D keypoint
L1 loss.

On the completed HO3D v2 AnyHand-HaMeR run, replacing only the tip ids changed
the root-relative diagnostic from:

```text
legacy tips: 16.21 mm MPJPE, 23.26 mm tip MPJPE
HaMeR tips:  16.03 mm MPJPE, 22.53 mm tip MPJPE
```

## Code Decision

- Keep `protocol_check_samples` / `protocol_tolerance_m`.
- Make them real: eval export now checks HO3D sample order by comparing the
  sample meta root joint against `evaluation_xyz.json` for the first configured
  samples before model inference.
- Do not change `score_predictions.py` units yet; `scores.txt` still writes
  mean distances in cm to match the existing evaluator behavior.
