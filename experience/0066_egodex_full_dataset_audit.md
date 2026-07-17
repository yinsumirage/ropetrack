# EgoDex downloaded-set full audit

Date: 2026-07-17

## Scope and integrity

Read-only CPU audit jobs `184414_0..5` and merge job `184415` scanned the
downloaded `test` plus `part1`-`part5` data. The 97-hour `extra` archive was not
downloaded or included.

- 318,082 HDF5 episodes and 318,082 matching MP4 files.
- 79,108,559 frames, or 732.487 hours at the native 30 FPS.
- 1,862,078,777,537 bytes (1.862 TB / 1.694 TiB) for paired HDF5+MP4 files.
- Zero HDF5 open errors and zero non-empty Slurm error logs.
- Every episode has `annotated=True`.
- 273,609 episodes (86.02%) contain native confidence arrays; 44,473 do not.
- Outputs and SUCCESS markers are under
  `/data/wentao/ropetrack/runs/egodex_full_audit_20260717`.

## Dataset structure

The downloaded data has 111 filesystem task categories, 4,992 recording
session names, 300 raw environment descriptor strings, and 840 raw object
descriptor strings. Environment/object strings are not canonicalized: field
order and spelling variants can describe the same physical setup, so they are
metadata diversity indicators rather than exact scene counts. Session count is
the more defensible recording-level diversity number.

The official 194-task claim is not directly comparable with the 111 directory
categories: this audit counts raw directory labels, reversible/multi-stage
actions can represent multiple semantic tasks, and `extra` is absent.

The five training archives are task partitions, not random shards:

| Split | Task dirs | Sessions | Episodes | Hours | GB | Confidence episodes |
|---|---:|---:|---:|---:|---:|---:|
| test | 111 | 2,026 | 3,243 | 7.650 | 19.487 | 86.00% |
| part1 | 26 | 893 | 46,234 | 142.090 | 360.483 | 91.49% |
| part2 | 13 | 1,417 | 95,125 | 144.900 | 372.025 | 92.08% |
| part3 | 24 | 733 | 53,779 | 145.757 | 367.367 | 100.00% |
| part4 | 23 | 832 | 44,129 | 147.601 | 370.339 | 90.98% |
| part5 | 25 | 1,140 | 75,572 | 144.489 | 372.377 | 62.20% |

Pairwise task overlap between training parts is exactly zero; their union is
all 111 test task directories. Part 2 concentrates the most episodes into only
13 tasks and contains the official `basic_pick_place` emphasis.

The top task is `basic_pick_place` with 27,696 episodes, followed by
`vertical_pick_place` with 15,281. The top 20 tasks contain 51.62% of all
episodes, so all-data training needs task-balanced sampling rather than uniform
episode sampling.

## Episode lengths

Episode duration percentiles are:

| percentile | min | p25 | p50 | p75 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| seconds | 0.50 | 2.93 | 5.37 | 10.00 | 15.47 | 52.74 | 178.80 |

Most files are short action clips; long sequences are a small tail. A training
sampler should therefore choose clips or sessions first and frames second, or
the long tail will dominate frame-uniform sampling.

## Confidence sample

To avoid reading every confidence value in 79.1M frames, the audit sampled up
to nine evenly spaced frames per episode. This is an episode-balanced dataset
description, not a replacement for frame-level filtering.

| Signal | Mean | `[0,.25)` | `[.25,.5)` | `[.5,.75)` | `[.75,.9)` | `[.9,1]` |
|---|---:|---:|---:|---:|---:|---:|
| wrist | 0.912 | 6.20% | 0.93% | 0.61% | 2.21% | 90.04% |
| fingertips | 0.467 | 22.43% | 34.46% | 24.44% | 16.29% | 2.38% |

Wrist tracking is usually confident, while fingertip supervision is mostly
low-to-mid confidence. This supports using EgoDex for global hand trajectory
and temporal/image-domain learning, but rejects treating every fingertip label
as equal-quality geometric ground truth.

## Minimal high-signal experiments

1. Use one task partition (Part 1 is the already-planned manageable pilot) and train equal
   budget variants: high-confidence only, high+mid weighted, and all rows
   equally. Score separately on the 26 seen test tasks and 85 unseen tasks.
2. Add task-balanced sampling and compare against uniform-episode sampling;
   this directly tests the measured 51.62% top-20 concentration.
3. Only if clean confidence-aware training wins without FreiHAND/HO3D
   regression, compare raw ARKit joint supervision with MANO-fitted pseudo
   supervision on the identical subset.

Do not start with a full five-part run: it would conflate label quality, task
imbalance, and domain transfer in one expensive result.
