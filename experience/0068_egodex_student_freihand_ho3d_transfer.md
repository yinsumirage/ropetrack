# EgoDex student transfer to FreiHAND and HO3D

Date: 2026-07-17

## Question and frozen protocol

This evaluates whether the Part 1 EgoDex `min025` student learned a general
rope-to-hand correction or only EgoDex's ARKit annotation bias. The checkpoint
was frozen; there was no retraining or new base inference. It was applied to
the exact WiLoR exports, MANO caches, rope labels, GT, and mask70 manifests used
by the four-teacher release comparison:

- FreiHAND mask70: 3,960 samples.
- HO3D v2 mask70: 11,524 samples.

GPU apply job `184805` and CPU score job `184806` completed with exit `0:0`.
Both scorers consumed the exact expected sample counts and all SUCCESS markers
exist under:

```text
/data/wentao/ropetrack/runs/egodex_min025_transfer_20260717
```

## Results

PA joint error is in mm. The WiLoR and four-teacher release rows are the frozen
values from the same exports and protocols.

| Dataset | WiLoR base | EgoDex `min025` | Delta vs base | Four-teacher release | EgoDex gap to release |
|---|---:|---:|---:|---:|---:|
| FreiHAND mask70 | 10.068 | 8.674 | -1.393 | 8.432 | +0.242 |
| HO3D v2 mask70 | 9.436 | 8.579 | -0.857 | 8.465 | +0.115 |

Relative to the stronger release student, the EgoDex-only checkpoint retains
85.2% of the FreiHAND improvement and 88.2% of the HO3D improvement.

The hard-split diagnostics are also positive:

| Dataset | Occluded-tip delta | Clean-tip delta | Mesh PA | F@5 |
|---|---:|---:|---:|---:|
| FreiHAND mask70 | -4.282 mm | -1.983 mm | 8.731 mm | 0.6198 |
| HO3D v2 mask70 | -2.179 mm | -1.299 mm | 8.673 mm | 0.5741 |

The model does not merely move the aggregate metric: both occluded and clean
tip slices improve against the same base on both datasets.

## Interpretation

This passes the intended transfer gate. A student trained from only 1,227
selected EgoDex episodes improves two datasets with independent MANO/mesh GT,
after also improving the 85 unseen EgoDex task categories in the Part 1 pilot.
The learned correction therefore has a general component and is not explained
solely by fitting EgoDex labels.

It does not replace the four-teacher release: that model remains better by
0.24 mm on FreiHAND and 0.11 mm on HO3D. Nor does this validate EgoDex as MANO
ground truth. EgoDex supplied noisy ARKit skeleton supervision for teacher
construction; the independent datasets are what validate transfer.

This run covers the established mask70 protocols, not fully visible clean
images. The earlier EgoDex confidence audit still warns that aggressive
correction can hurt already-good hands, so clean-protocol preservation remains
a required check before changing the release model.

## Decision

Continue dataset expansion, but do not run all 318k training clips. The next
smallest useful training view is task-balanced across all five parts: up to 50
native-confidence episodes per each of the 111 tasks, stride 30, with the same
`min025` rule and sequence-disjoint validation. This is roughly 5,550 episodes,
about 4.5 times the pilot rather than the full 1.86 TB corpus.

Keep the four-teacher checkpoint as release. Promote an expanded EgoDex student
only if it improves EgoDex while preserving FreiHAND mask70, HO3D v2 mask70,
and a clean FreiHAND check. Do not spend more jobs sweeping nearby confidence
thresholds; the Part 1 ablation already showed that threshold choice was a
minor effect.
