# ARCTIC P2 validation full results

Date: 2026-07-17

## Scope and completed jobs

This is a local P2 validation result, not an official test/leaderboard result.
The evaluated split is `s05`, egocentric view 0, with 34 sequences and 38,921
valid hand samples (20,005 right, 18,916 left). WiLoR/Flex15 GPU job 183988
completed in 11:55 with zero failures; CPU scoring job 183991 completed in
40:18. The frozen release checkpoint was used without retraining.

The direct WiLoR prediction and the prediction reconstructed from its MANO
cache agree to below `1e-8` mm on all four reported mean errors. This closes
the full-run cache/left-hand path check.

## Primary geometry result

Errors are millimetres. Delta is signed `Flex15 - WiLoR`; negative is better.

| scope / method | samples | raw joint | PA joint | raw mesh | PA mesh |
|---|---:|---:|---:|---:|---:|
| all / WiLoR original | 38,921 | 51.8973 | 6.7032 | 51.6889 | 6.5707 |
| all / release Flex15 | 38,921 | 51.7974 | 6.7265 | 51.5846 | 6.5876 |
| all / delta | 38,921 | -0.0999 | +0.0233 | -0.1042 | +0.0169 |
| left / WiLoR original | 18,916 | 54.0350 | 7.1545 | 53.7547 | 7.0243 |
| left / release Flex15 | 18,916 | 53.7643 | 7.1597 | 53.4811 | 7.0241 |
| left / delta | 18,916 | -0.2707 | +0.0052 | -0.2735 | -0.0002 |
| right / WiLoR original | 20,005 | 49.8760 | 6.2765 | 49.7355 | 6.1417 |
| right / release Flex15 | 20,005 | 49.9376 | 6.3169 | 49.7914 | 6.1749 |
| right / delta | 20,005 | +0.0616 | +0.0405 | +0.0558 | +0.0332 |

Flex15 slightly improves aggregate raw joint/mesh error but slightly worsens
both PA errors. The effect is side-dependent: raw errors improve on left hands
and regress on right hands. These sub-millimetre mixed deltas do not establish
a zero-shot geometric gain on ARCTIC.

## Rope closure

- Mean absolute residual: `0.112012 -> 0.072948`.
- Median absolute residual: `0.099518 -> 0.068945`.
- Closure fraction: `0.348752`.
- Fingers improved: `0.597199` of 194,605 valid fingers.
- Frozen gate threshold: `0.1`; 45.07% of fingers and 83.92% of samples had
  at least one gated correction.

The release model is active and closes its rope objective, but that objective
does not transfer into a consistent ARCTIC geometry improvement. No temporal,
shuffle, retraining, or P1 experiment is needed to support this conclusion.

## Availability and interpretation

- Official `s03` test GT remains hidden; no leaderboard submission was made.
- ARCTIC object metrics such as MR-E are unsupported by this hand-only
  RopeTrack adapter and are unavailable.
- The zero-shot result measures WiLoR/release-model cross-domain behaviour, not
  annotation quality. The low PA errors and completed MANO/projection/overlay
  audits support internally coherent hand geometry, while the much larger raw
  errors show the dominant cross-domain/global-camera difficulty.

ARCTIC is suitable for later subject-disjoint hand-pose training. The final
shared-data integrity and copy gate passed (see experience 0071). It provides calibrated multiview
images, native side-specific MANO parameters/meshes, kinematic joints, explicit
validity, and an official split. That is materially stronger supervision than
the audited EgoDex labels. Use official kinematic joints rather than regressing
joints from posed vertices, preserve side validity and subject splits, and use
`s05` only for local validation.

## Acquisition status

Dataset acquisition is closed. The official 339-sequence shared extraction was
copied and verified at the RopeTrack data root: 3,051 view directories,
2,190,652 JPEGs, and 127,994,982,876 image bytes. The 246 locally downloaded
official archives were retained after published-checksum and per-member CRC
agreement; incomplete parts were deleted. Two one-view, one-frame source gaps
are recorded in experience 0071 and do not affect the frozen P2 validation.
