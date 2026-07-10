# Dense History Utility Design

**Date:** 2026-07-11  
**Branch:** `codex/temporal-ho3d-refiner`

## Goal

Test whether clean causal context improves hand correction during later HO3D
v3 occlusion, rather than testing unrelated all-clean or all-hard datasets.

## Data contract

Use only the registered dense HO3D v3 episode roots:

```text
30 clean context -> 60 mask70 -> 30 clean recovery
```

- Training: 83,325 frames, 321 segments, 575 complete episodes.
- Evaluation: 20,137 frames, 70 segments, 153 complete episodes.
- Sequence/gap boundaries reset history.
- `episode_phase` is used only by scoring, never as a model input.
- Training and validation remain sequence-disjoint.

## Chosen experiment

Reuse the existing flex15/gate010 teacher, framewise student, causal K1, and
schema-v2 strict-past residual. Do not add a new recurrent architecture before
the existing model proves that dense history contains useful signal.

Run these causal comparisons on the same dense teacher and split:

1. Matched framewise student.
2. Dense K1, raw-frame step 1.
3. K16 strict-past residual: about 0.5 seconds of history at 30 fps.
4. K96 strict-past residual: long enough to retain the clean prefix through
   the full 60-frame mask.
5. K96 shuffled-past control at the selected learning rate.
6. Embedded K1 with history disabled at the same inference cadence.

K16/K96 use hidden size 64 and learning rates `3e-4` and `1e-3`, seed 0.
A one-epoch K96 GPU memory smoke runs before the full grid.

## Evaluation

First score base WiLoR alone to verify that masked frames are actually harder
than context/recovery. Then apply all candidates to dense evaluation and score:

- context, masked, and recovery PA-MPJPE;
- masked occluded-tip PA-MPJPE;
- recovery frames and phase lag;
- velocity error, acceleration error, and jitter;
- paired sequence-bootstrap intervals against same-cadence K1.

The existing scorer is sufficient for the first gate. If K96 passes, add only
then an inference ablation that resets history at the first masked frame. That
ablation distinguishes clean-prefix memory from generic masked-frame history.

## Gates

History is useful only if all of the following hold:

- base masked PA-MPJPE is worse than context PA-MPJPE;
- K96 improves masked PA-MPJPE over same-cadence K1 by at least `0.15 mm`;
- the 95% sequence-bootstrap interval excludes zero;
- masked occluded-tip error does not regress;
- acceleration error and jitter improve by at least 10%;
- shuffled past loses at least half of the real-history gain.

If the first seed fails the validation/history-specificity gate, do not expand
seeds or model capacity. If K96 passes, run seeds 1 and 2 and the mask-boundary
reset ablation before making a final claim.

## Outputs

Store all generated teachers, checkpoints, predictions, scores, and Slurm logs
under `/data/wentao/ropetrack/runs`; commit only code, configs, tests, plans,
and experiment records.
