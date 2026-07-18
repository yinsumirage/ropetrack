# Normal-data joint training: no-leak audit and frozen final protocol

Date: 2026-07-19

## Historical audit before new training

Jobs `186416` and `186434` read the actual ARCTIC manifests/run order, HOT3D NPZ bundles,
participant folds, and HO3D v3 split files. The machine-readable report is
`/data/wentao/ropetrack/runs/direct_pose_normal_joint_20260719/audit/history_audit.json`.
It did not infer separation from the prose in 0075-0078.

| Historical cell | Train -> evaluation | Sample/image/sequence/participant overlap | Eval GT in model selection or loss | Verdict |
|---|---|---:|---|---|
| 0075 ARCTIC-only C | P2 train subjects `s01,s02,s04,s06-s10` -> P2 val `s05` | `0/0/0/0` | no; internal episode validation only | **valid** |
| 0076 ARCTIC-only transfer | ARCTIC P2 train -> HOT3D natural | `0/0/0/0` cross-dataset | no | **valid transfer** |
| 0076 small ARCTIC+HOT3D, reported `HOT3D all` | ARCTIC plus 2,166 rows from the same 4,074-row HOT3D screen -> all rows | `2166/2166/35/7` | yes: 1,894 rows entered train loss and 272 entered internal validation | **contaminated; invalidate the all-participant claim** |
| 0076 small ARCTIC+HOT3D, P0014/P0015 only | HOT3D train excludes both participants -> their 1,908 rows | `0/0/0/0` | no | **valid** |
| 0077 small participant-CV stitch | each fold trains on other participants -> excluded participants only | zero in every fold | no train/early-stop leakage, but the same CV was reused for model decisions | **valid participant-disjoint OOF validation; not untouched test** |
| 0077 expanded participant-CV stitch | each fold trains on six participants -> excluded three only | zero in every fold | no train/early-stop leakage, but the same CV was reused for capacity/mixture/LoRA decisions | **valid participant-disjoint OOF validation; not untouched test** |
| 0076/0077 HO3D rows | ARCTIC or ARCTIC+HOT3D -> HO3D v2 `mask70` | `0/0/0/0` cross-dataset | no | **valid artificial-occlusion transfer only; not normal-HO3D evidence** |
| Historical direct-GT HO3D training | none found | n/a | n/a | **absent** |

The corrected historical conclusion is therefore narrower than 0076: the
`6.231 mm HOT3D all` number is invalid, while the P0014/P0015 held-out result
and the later nine-participant CV stitch remain usable as validation evidence.
The latter is project-internal three-fold participant CV, not an official
HOT3D split and not nine-fold LOPO. Each prediction is participant-disjoint,
but repeated reuse for capacity, mixture, and LoRA judgments means it is no
longer a completely untouched final test.

The fold groups were hardcoded by participant ID only:
`P0001/P0002/P0003`, `P0009/P0010/P0011`, and
`P0012/P0014/P0015`. They were neither official nor statistically optimized.
Their evaluation row counts are `1,657/3,272/3,218`, and sequence/episode
counts are `29/53/51`, so the first fold is materially smaller. Context-row
fractions are `0.525/0.486/0.475`; mean official visibility ratios are
`0.566/0.546/0.558`. The manifest and raw sequence metadata expose no action,
activity, or task label, so action balance is unknown; the audit reports
per-fold object mixes only as a proxy and makes no action-balance claim.

No historical cell trained a direct-GT head on HO3D, and no historical HO3D
result answers the normal-image joint-training question.

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

HO3D is explicitly v3. The coordinate smoke first confirmed that the native
MANO decode reproduces all 16 kinematic train-meta joints to numerical
precision. Extending that gate to all 21 joints then found a real convention
boundary: raw train-meta tips are nearest to vertices
`[744,333,444,555,672]`, while the existing WiLoR decoder and frozen HO3D
evaluation protocol both use `[744,320,443,554,671]`; directly mixing them
would leave a 2.693 mm mean tip mismatch on the 32-row diagnostic (2.674 mm
over the full 27,000-row export; 0.641 mm averaged over 21 joints in the
diagnostic).
The first preparation was therefore rejected before training.

The corrected training export keeps the official first 16 joints and decodes
only the five evaluation-convention fingertip vertices from official train
MANO pose/beta, root-aligned to the same train-meta joints. It then converts
MANO order to OpenPose order and OpenGL-style axes to OpenCV camera metres
(`[x,-y,-z]`). This uses only train-split GT and a fixed coordinate convention,
not evaluation GT values or observed errors. A 32-sample numeric native-MANO
decode and projected overlay must pass before the three full fold jobs start.

## One-shot final evaluation and decision rule

After the recipe is frozen, each of the three checkpoints is evaluated once
on normal, unmodified held-out data:

1. full ARCTIC P2 val (`s05`, 38,921 hands);
2. HOT3D natural participant-CV stitch (8,147 hands, including the existing
   natural low-visibility subgroup); this is one pre-frozen reuse of the
   project CV, not a newly untouched test;
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
ARCTIC P2 val and HO3D v3 evaluation serve as independent dataset anchors for
the one-shot HOT3D CV result.

## Final results

All scores below are from job `186523`; negative deltas are improvements.
The paired 95% intervals use 2,000 fixed-seed group bootstrap replicates:
ARCTIC/HO3D by sequence and HOT3D by episode.

### Executed training matrix

| Cell | ARCTIC normal train | HOT3D normal train | HO3D v3 normal train | Held-out HOT3D | Internal train/val | Selected epoch |
|---|---:|---:|---:|---|---:|---:|
| ARCTIC-only reference | 31,006 | 0 | 0 | n/a | historical frozen checkpoint | historical |
| valid expanded dual fold 0/1/2 | 31,006 | 27,000 | 0 | the corresponding three-ID group | historical frozen checkpoints | historical |
| triple fold 0 | 31,006 | 27,000 | 27,000 | `P0001/P0002/P0003` | 75,666 / 9,340 | 13 |
| triple fold 1 | 31,006 | 27,000 | 27,000 | `P0009/P0010/P0011` | 73,012 / 11,994 | 30 |
| triple fold 2 | 31,006 | 27,000 | 27,000 | `P0012/P0014/P0015` | 74,574 / 10,432 | 19 |

Every triple checkpoint contains commit
`d4e5a400e4ef9d078392effa3325e90c9775b9e0`, the held-out participant list,
all source/sample hashes, the combined sample hash, and separate internal
train/validation hashes. Job `186567` re-opened all three checkpoints and
verified 85,006 merged rows per fold, all embedded zero-overlap gates, exact
commit provenance, final evaluation counts, and score-file completeness.

### Primary PA result

| Normal evaluation | WiLoR base | AR-only | Valid dual | New triple | Triple - dual | Paired 95% CI | Judgment |
|---|---:|---:|---:|---:|---:|---:|---|
| ARCTIC P2 val | 6.703 | 5.966 | 5.998 (5.933-6.054) | 6.018 (5.953-6.067) | +0.020 | [-0.004, +0.045] | uncertain slight regression |
| HOT3D OOF stitch | 8.984 | 8.534 | 5.732 | 5.751 | +0.019 | [-0.058, +0.093] | uncertain slight regression |
| HO3D v3 evaluation | 7.153 | 6.972 | 7.686 (7.637-7.752) | **6.556** (6.495-6.595) | **-1.130** | **[-1.590, -0.706]** | clear improvement |

Parentheses are the three checkpoint range on the two dataset anchors. The
new triple model still improves WiLoR base on all three datasets: -0.686 mm
on ARCTIC, -3.233 mm on HOT3D, and -0.598 mm on HO3D; the corresponding
base-paired intervals all exclude zero. That does not satisfy promotion,
because the matched comparison is the valid dual model, not base.

The HOT3D stitched score is row-weighted and therefore reflects the audited
fold-size imbalance. The unweighted fold means/ranges are 5.678
(5.199-6.455) for dual and 5.671 (5.246-6.406) for triple. Per fold:

| HOT3D held-out fold | Rows | Dual PA | Triple PA | Delta |
|---|---:|---:|---:|---:|
| P0001/P0002/P0003 | 1,657 | 5.380 | 5.246 | -0.134 |
| P0009/P0010/P0011 | 3,272 | 5.199 | 5.362 | +0.163 |
| P0012/P0014/P0015 | 3,218 | 6.455 | 6.406 | -0.049 |

This is a mixed fold result, and the first fold carries about half the rows
of either other fold. It must not be summarized as a universally improved
participant-CV model.

### Natural visibility and metric boundary

| HOT3D subgroup | Rows | Base PA | AR-only PA | Dual PA | Triple PA | Triple - dual |
|---|---:|---:|---:|---:|---:|---:|
| context | 3,990 | 7.643 | 7.921 | 5.297 | 5.324 | +0.026 |
| low visibility | 4,157 | 10.271 | 9.122 | 6.148 | 6.161 | +0.012 |

HO3D coverage produces no hidden low-visibility win on the reused HOT3D CV;
both natural subgroups are essentially flat to slightly worse than dual.

| Evaluation | Root base / dual / triple | Camera base / dual / triple | Triple - dual root | Triple - dual camera |
|---|---:|---:|---:|---:|
| ARCTIC | 15.293 / 14.870 / 14.838 | 51.897 / 51.198 / 51.249 | -0.032 | +0.051 |
| HOT3D | 66.574 / 65.342 / 65.613 | 138.581 / 139.539 / 139.473 | +0.271 | -0.067 |
| HO3D v3 | 14.571 / 14.649 / 13.835 | 1646.927 / 1647.601 / 1647.313 | -0.813 | -0.288 |

All values are mm. PA, root-relative, and camera-frame results are separate
claims. HOT3D camera remains fisheye-boundary diagnostic. HO3D camera is also
diagnostic here: its raw translation convention yields metre-scale absolute
error despite normal PA/root values, and the triple camera value is 0.386 mm
worse than base.

### Decision

**Continue the data-coverage line, but do not promote this fixed triple
mixture.** It strongly and significantly fixes the missing HO3D domain while
retaining large gains over WiLoR base, but it fails the predeclared matched
criterion: triple does not improve dual on ARCTIC and HOT3D, and neither small
positive delta is distinguishable from zero. This is not a stop signal for
HO3D supervision, but it is a stop on tuning another mixture against these
same final scores. A future mixture decision needs training-only validation
or a new untouched participant/external anchor.

### Artifacts and jobs

- Run root: `/data/wentao/ropetrack/runs/direct_pose_normal_joint_20260719`;
- history audit: `audit/history_audit.json` (jobs `186416`, `186434`);
- corrected HO3D train export:
  `/data/wentao/ropetrack/processed/ho3d_v3/normal_train_27000_evaltips_20260719`;
- coordinate gate: `coordinate_gate_full/summary.json`;
- frozen protocols and no-leak report: `protocol/fold{0,1,2}.json` and
  `protocol/no_leak_report.json`;
- checkpoints: `students/fold{0,1,2}/model.pt`;
- final metrics: `scores/final_metrics.json`;
- independent artifact verification: `audit/final_verification.json`.

Retained successful Slurm jobs are `186436` (initial deterministic selection),
`186455` (1-GPU smoke), `186461` (full GPU cache array), `186491` (tip
diagnosis), `186509` (corrected train-only GT export), `186510` (final 21-joint
coordinate gate), `186519` (bundle and no-leak protocols), `186520` (three
training folds), `186521` (21 final apply cells), `186522` (HOT3D participant
stitch), `186523` (final scorer), and `186567` (artifact verification); all
listed final-path jobs completed with exit code 0.

Diagnostic jobs `186473` and `186481` deliberately failed the expanded tip
gate and exposed the convention mismatch. Job `186497` then exposed a
standalone-script import bug, fixed in commit `d4e5a40`. The first bundle job
`186511` found that a full export does not copy GT into `eval_input`; corrected
job `186519` reads the frozen training export directly. Dependent pending jobs
were canceled before model or score artifacts were produced and resubmitted
against the corrected exact worktree.
