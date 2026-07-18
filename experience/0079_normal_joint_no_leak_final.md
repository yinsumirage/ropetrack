# Normal-data joint training: no-leak audit and frozen final protocol

Date: 2026-07-19

## Historical audit before new training

Job `186416` read the actual ARCTIC manifests/run order, HOT3D NPZ bundles,
participant folds, and HO3D v3 split files. The machine-readable report is
`/data/wentao/ropetrack/runs/direct_pose_normal_joint_20260719/audit/history_audit.json`.
It did not infer separation from the prose in 0075-0078.

| Historical cell | Train -> evaluation | Sample/image/sequence/participant overlap | Eval GT in model selection or loss | Verdict |
|---|---|---:|---|---|
| 0075 ARCTIC-only C | P2 train subjects `s01,s02,s04,s06-s10` -> P2 val `s05` | `0/0/0/0` | no; internal episode validation only | **valid** |
| 0076 ARCTIC-only transfer | ARCTIC P2 train -> HOT3D natural | `0/0/0/0` cross-dataset | no | **valid transfer** |
| 0076 small ARCTIC+HOT3D, reported `HOT3D all` | ARCTIC plus 2,166 rows from the same 4,074-row HOT3D screen -> all rows | `2166/2166/35/7` | yes: 1,894 rows entered train loss and 272 entered internal validation | **contaminated; invalidate the all-participant claim** |
| 0076 small ARCTIC+HOT3D, P0014/P0015 only | HOT3D train excludes both participants -> their 1,908 rows | `0/0/0/0` | no | **valid** |
| 0077 small participant-CV stitch | each fold trains on other participants -> excluded participants only | zero in every fold | no for the scored participant fold | **valid** |
| 0077 expanded participant-CV stitch | each fold trains on six participants -> excluded three only | zero in every fold | no for the scored participant fold | **valid** |
| 0076/0077 HO3D rows | ARCTIC or ARCTIC+HOT3D -> HO3D v2 `mask70` | `0/0/0/0` cross-dataset | no | **valid artificial-occlusion transfer only; not normal-HO3D evidence** |
| Historical direct-GT HO3D training | none found | n/a | n/a | **absent** |

The corrected historical conclusion is therefore narrower than 0076: the
`6.231 mm HOT3D all` number is invalid, while the P0014/P0015 held-out result
and the later nine-participant CV stitch remain usable. No historical cell
trained a direct-GT head on HO3D, and no historical HO3D result answers the
normal-image joint-training question.

## Frozen normal-data protocol (written before final scores)

This run changes data coverage only. It keeps `DirectPoseHead` h128, frozen
localized WiLoR 4x3 tokens, correct normalized five-rope input, loss weights,
optimizer, batch size, and internal episode early stopping from 0075-0077.
There is no global-orient head, LoRA, larger head, sensor augmentation,
real-length input, artificial mask, or synthetic occlusion.

Training sources are fixed before evaluation:

- ARCTIC: all 31,006 normal P2-train stride-10 hands from 267 sequences;
- HOT3D: the existing 27,000-row normal public-GT bundle in each fold, with
  folds holding out `P0001-P0003`, `P0009-P0011`, and `P0012/P0014/P0015`;
- HO3D v3: 27,000 unmodified official train frames selected once by
  deterministic sequence-balanced sampling without replacement, seed
  `20260719`; official train and evaluation have 83,325/20,137 rows and zero
  sample, image, or sequence overlap.

The per-fold source counts are `31,006 : 27,000 : 27,000`. Equal per-row loss
therefore leaves the largest/smallest ratio at only 1.148; no source can swamp
the other two. The rule uses dataset/sequence/participant/official visibility
metadata only, never evaluation error. Only training-split GT builds rope
labels and direct supervision. Internal validation is the existing
deterministic episode split of the merged training sources. ARCTIC P2 val,
HOT3D held-out participants, and HO3D evaluation are never passed to the
trainer or early stopper.

HO3D is explicitly v3. Its official MANO-order, OpenGL-style training joints
are converted to OpenPose order and OpenCV camera metres (`[x,-y,-z]`) before
direct supervision. A numeric native-MANO decode and projected overlay smoke
must pass before the three full fold jobs start.

## One-shot final evaluation and decision rule

After the recipe is frozen, each of the three checkpoints is evaluated once
on normal, unmodified held-out data:

1. full ARCTIC P2 val (`s05`, 38,921 hands);
2. HOT3D natural participant-CV stitch (8,147 hands, including the existing
   natural low-visibility subgroup);
3. full HO3D v3 evaluation (20,137 hands).

The comparison is WiLoR base, the frozen ARCTIC-only checkpoint, the valid
expanded ARCTIC+HOT3D fold checkpoints, and the new three-dataset folds.
Primary judgment is PA joint error. Root-relative and camera-frame errors are
reported separately; HOT3D camera error remains diagnostic because the
current crop path does not model the full fisheye projection.

- **validated for promotion:** the three-dataset mean improves the matched
  dual-data result on all three normal datasets, with paired 95% CI excluding
  zero on at least two;
- **continue, not promote:** effects are mixed or statistically uncertain;
- **stop this fixed mixture:** a clear cross-dataset PA regression or a
  failure of the split/coordinate gates appears.

No second mixture, weight, seed, or checkpoint is chosen after these scores.

## Final results

Pending the frozen Slurm chain. This section will be completed only from
finished score artifacts and recorded exit codes.
