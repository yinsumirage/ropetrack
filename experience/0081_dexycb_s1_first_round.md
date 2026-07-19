# DexYCB S1 protocol, coordinate, and DirectPose first round

Date: 2026-07-20

## Decision

- **Adapter and coordinate protocol: validated.** Official S1 subject, sample,
  and synchronized-view separation passes; native MANO decoding and camera
  projection reproduce the labels to numerical precision for both hands.
- **WiLoR external transfer: established baseline.** On the one-shot official
  test population, WiLoR base is 4.783 mm PA, 8.260 mm root-relative, and
  57.847 mm camera-frame joint error.
- **Frozen old DirectPose external transfer: mixed.** The three fixed folds
  improve S1-val PA by 0.282 mm and root-relative error by 0.091 mm on average,
  but worsen camera-frame error by 0.262 mm. This is not a clean global-pose
  transfer result.
- **DexYCB RGB-only supervision: stop this recipe.** Its test PA change is
  statistically uncertain (-0.044 mm, 95% CI [-0.092, +0.003]) while
  root-relative error clearly worsens by 0.639 mm, 95% CI
  [+0.571, +0.708]. Do not scale it merely because PA moved slightly.
- **Additional ideal-rope value: validated within the GT-derived boundary.**
  Relative to the matched RGB-only model, RGB+rope improves test PA by
  1.300 mm, root-relative error by 1.348 mm, and camera-frame error by
  0.115 mm; all three paired intervals exclude zero. This says nothing yet
  about a physical rope sensor, whose calibration, noise, drift, dropout, and
  latency remain unvalidated.
- **Do not run full S1 train now and do not add DexYCB to the frozen
  ARCTIC+HOT3D+HO3D mixture.** The current RGB-only supervision fails its
  matched shape gate, while the positive result uses ideal GT-derived rope.
  Any future DexYCB recipe must be chosen from S1 train/internal validation
  and a predeclared S1-val protocol, not by revisiting this test result.

## Official protocol and semantic evidence

The implementation is grounded in the official
[DexYCB toolkit](https://github.com/NVlabs/dex-ycb-toolkit) at commit
`64551b001d360ad83bc383157a559ec248fb9100`, the
[DexYCB paper](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf), the
[BOP challenge protocol](https://bop.felk.cvut.cz/challenges/bop-challenge-2019/),
and the [BOP dataset format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).
The toolkit setup literals, local copy audit, and BOP target manifest were
cross-checked independently in `split_report.json`.

| Official S1 split | Subjects | Sequences | Synchronized frames | All views |
|---|---:|---:|---:|---:|
| train | 01-06, 10 | 700 | 50,886 | 407,088 |
| val | 07 | 100 | 7,324 | 58,592 |
| test | 08-09 | 200 | 14,536 | 116,288 |

All train/val/test subject, sample-id, episode, and synchronized-view
intersections are zero. `sample_id` is
`subject/sequence/camera_serial/frame_index`; `episode_id` is
`subject/sequence`, deliberately excluding camera. The BOP S1 target list has
29,438 unique scene/image pairs versus the toolkit's 29,440 every-fourth-frame
keyframes. The two omitted pairs are exactly the images whose best object
visible fractions are 0.01394 and 0.09904, below BOP's 10% target threshold.

The full dataset is not right-hand-only: official sequence metadata gives 501
right and 499 left sequences. The suggested constant `is_right=true` would be
wrong, so the adapter records the official per-sequence side. Official all-view
counts are retained for protocol auditing, while training/evaluation exports
skip the toolkit's `joint_3d=-1`, `pose_m=0` no-hand sentinel and invalid
clipped bboxes:

| Export | Accepted | Rejected | Main rejected reasons |
|---|---:|---:|---|
| train27k | 27,000 | 0 after deterministic refill | 34,072 candidate views skipped before refill |
| S1 val | 46,859 | 11,733 | 10,936 sentinel; 797 empty clipped bbox |
| S1 test | 103,086 | 13,202 | 12,160 sentinel; 1,042 empty clipped bbox |

The bbox is the same for every method: finite official `joint_2d`, 25% margin
per side, clip to `[0,0,639,479]`, reject fewer than four finite joints, span
below four pixels, or an empty clipped box. Model error never affects bbox or
sample selection. RGB is the only image input; depth, artificial masks,
`mask70`, and synthetic occlusion are absent. Natural object occlusion remains.

## Coordinate and native MANO gates

Official semantics were verified before training rather than inferred from
array names:

- `joint_3d` is in metres in the per-camera OpenCV frame (+x right, +y down,
  +z forward); `joint_2d` is in the matching color-image pixels.
- Color intrinsics are the official row-major
  `[[fx,0,ppx],[0,fy,ppy],[0,0,1]]`. In the toolkit sequence loader, each
  extrinsic 3x4 `T[serial]` maps that camera's points to the shared
  master/world frame and its explicit inverse maps shared coordinates back to
  the camera. The adapter does not apply an extrinsic because every label file
  already stores joints and MANO translation in its own camera frame.
- The 21 joints map to RopeTrack/OpenPose order using
  `[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]` after MANO
  decoding.
- `pose_m` is 3 camera-frame global axis-angle values, 45 articulated PCA
  coefficients, and 3 translation values. Native decoding uses
  `flat_hand_mean=False`, PCA45, the subject's 10 betas, and official right/left
  MANO layers. Right tip vertices are `[745,317,444,556,673]`; left uses
  `[745,317,445,556,673]`.

| Gate | Pre-train: train+val only | Final after freeze |
|---|---:|---:|
| samples / subjects / sequences / cameras | 243 / 8 / 179 / 8 | 304 / 10 / 211 / 8 |
| right / left samples | 119 / 124 | 152 / 152 |
| reprojection mean / median / P95 / max px | 0.0000125 / 0 / 0.0000610 / 0.0001365 | 0.0000127 / 0 / 0.0000610 / 0.0001365 |
| native MANO camera mean / P95 / max mm | 0.0000615 / 0.0001238 / 0.0002486 | 0.0000613 / 0.0001238 / 0.0002486 |
| native MANO root-relative mean / P95 mm | 0.0000551 / 0.0001238 | 0.0000549 / 0.0001238 |
| best fixed rotation | 0.0 degrees | 0.0 degrees |

The sampling explicitly covers near/far depth and low/high official hand
segmentation area for every subject/camera. Each gate saves 32 RGB overlays;
green projected `joint_3d` and magenta label `joint_2d` coincide visually.
`pose_m[48:51]` is a MANO translation parameter rather than the wrist joint:
the final mixed-hand diagnostic records wrist-minus-parameter mean
`[2.59, 6.58, 6.43]` mm and x standard deviation 95.85 mm, while full decoded
joints still match labels at 0.000061 mm mean. Therefore translation is kept as
a separate diagnostic and is never substituted for root GT.
The pre-train gate excludes S1 test subjects; the final 10-subject gate runs
only after checkpoint and evaluation recipe freeze.

## Fixed train27k and matched training

`dexycb_s1_train27k_v1` uses only official S1-train subjects. A
capacity-constrained episode round-robin water-fill and stable frame hash pick
at most one camera for each `(subject, sequence, frame)`. All 700 sequences
are present; per-sequence selections range from 25 to 39, and every underfull
sequence is proven exhausted rather than silently imbalanced. Each of the
eight cameras contributes exactly 3,375 rows. The complete-sequence internal
split is 24,296 train / 2,704 validation; no episode crosses sides.

RGB-only and RGB+normalized-five-rope share the exact sample hashes, internal
split, WiLoR MANO cache, 4x3x1280 frozen tokens, h128 head, optimizer, loss
weights, seed, and early-stopping rule. Their only difference is
`rope_mode=zero` versus `rope_mode=correct`. The selected epochs are 11 and
38. The rope values are GT-derived ideal five-rope geometry, not sensor data.

## External transfer on official S1 val

The frozen ARCTIC+HOT3D+HO3D checkpoints are all reported; no fold was selected
using DexYCB.

| Metric | WiLoR base | Old fold mean | Fold range | Mean - base |
|---|---:|---:|---:|---:|
| PA joint mm | 5.231 | 4.950 | 4.869-5.041 | -0.282 |
| Root-relative joint mm | 8.779 | 8.688 | 8.584-8.762 | -0.091 |
| Camera-frame joint mm | 55.378 | 55.639 | 55.556-55.789 | +0.262 |
| Root translation mm | 54.434 | 54.424 | 54.416-54.429 | -0.010 |
| Global orientation deg | 8.622 | 8.425 | 8.327-8.483 | -0.197 |

The old head transfers local shape/orientation modestly but not camera-frame
accuracy. Treat it as mixed transfer, not a general DexYCB winner.

## Matched official val and one-shot test

Lower is better. The test scorer ran once after the freeze file recorded both
checkpoint hashes, internal-val selections, visibility thresholds, and
`test_score_reads_before_freeze=0`.

| Split / method | PA joint mm | Root-relative mm | Camera-frame mm | Root translation mm | Global orientation deg |
|---|---:|---:|---:|---:|---:|
| S1 val WiLoR base | 5.231 | 8.779 | 55.378 | 54.434 | 8.622 |
| S1 val RGB-only | 5.199 | 9.363 | 55.370 | 54.448 | 7.694 |
| S1 val RGB+rope | 3.894 | 8.084 | 55.152 | 54.425 | 7.655 |
| S1 test WiLoR base | 4.783 | 8.260 | 57.847 | 57.401 | 7.668 |
| S1 test RGB-only | 4.739 | 8.899 | 57.733 | 57.418 | 7.005 |
| S1 test RGB+rope | 3.439 | 7.551 | 57.618 | 57.400 | 6.970 |

Paired bootstrap uses 2,000 fixed-seed replicates. Negative deltas improve.

| Test comparison | PA delta (95% CI) | Root-relative delta (95% CI) | Camera delta (95% CI) | Root delta (95% CI) | Orientation delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| RGB-only - base | -0.044 [-0.092,+0.003] | +0.639 [+0.571,+0.708] | -0.114 [-0.150,-0.077] | +0.017 [+0.015,+0.018] | -0.663 [-0.768,-0.567] |
| RGB+rope - RGB-only | -1.300 [-1.359,-1.243] | -1.348 [-1.411,-1.287] | -0.115 [-0.161,-0.066] | -0.019 [-0.020,-0.017] | -0.035 [-0.063,-0.010] |

The camera-frame metric is dominated by translation: all methods retain the
same WiLoR translation parameter, and the head predicts articulated pose, not
a new camera translation. It is therefore expected that the large PA/root
rope gain produces only a small camera-frame gain. This boundary is reported,
not hidden.

### Test subjects

| Subject | Rows | Base PA / root | RGB-only PA / root | RGB+rope PA / root |
|---|---:|---:|---:|---:|
| 20201002-subject-08 | 45,278 | 4.998 / 8.870 | 4.854 / 9.316 | 3.528 / 8.170 |
| 20201015-subject-09 | 57,808 | 4.614 / 7.782 | 4.649 / 8.572 | 3.369 / 7.065 |

RGB-only root-relative error worsens for both unseen subjects; its PA also
worsens on subject 09. RGB+rope improves PA and root-relative error on both.

### Test cameras

| Camera serial | Rows | Base PA | RGB-only PA | RGB+rope PA |
|---|---:|---:|---:|---:|
| 836212060125 | 13,016 | 4.756 | 4.748 | 3.434 |
| 839512060362 | 13,016 | 4.789 | 4.749 | 3.450 |
| 840412060917 | 12,944 | 4.790 | 4.809 | 3.486 |
| 841412060263 | 12,976 | 4.875 | 4.916 | 3.521 |
| 932122060857 | 12,200 | 5.155 | 5.005 | 3.625 |
| 932122060861 | 12,902 | 4.862 | 4.756 | 3.453 |
| 932122061900 | 13,016 | 4.535 | 4.473 | 3.284 |
| 932122062010 | 13,016 | 4.525 | 4.472 | 3.270 |

RGB+rope improves PA on all eight cameras. RGB-only regresses on two cameras
and, as the score JSON records, worsens root-relative error on every camera.
The full per-camera root, camera-frame, translation, and orientation tables
remain machine-readable in `test/scores/scores.json`.

### Visibility / natural occlusion proxy

The buckets use pre-frozen official hand-segmentation visible-pixel thresholds;
they do not use model error.

| Bucket | Rows | Base PA / root | RGB-only PA / root | RGB+rope PA / root |
|---|---:|---:|---:|---:|
| high visible | 40,819 | 4.561 / 7.524 | 4.508 / 8.222 | 3.309 / 6.840 |
| mid visible | 29,686 | 4.760 / 7.938 | 4.702 / 8.623 | 3.407 / 7.268 |
| low visible | 32,581 | 5.082 / 9.475 | 5.062 / 9.998 | 3.631 / 8.699 |

Ideal rope helps all three buckets. The absolute errors remain worst in the
low-visible bucket, so it does not erase natural-occlusion difficulty.

## Scale decision

Do not launch the nominal full 407,088-view train automatically. That official
count contains 50,488 no-hand sentinels before bbox filtering, so a valid full
export would contain at most 356,600 rows. Linearizing the measured H200 jobs
from 27k suggests roughly 50-60 minutes for each full WiLoR/MANO or token cache
and roughly 50-60 minutes per matched cached-head branch if early stopping is
similar: about 4-6 aggregate H200 GPU-hours plus CPU export/I/O, or roughly
2-3 wall-clock hours with two GPUs before queue delay. The compute is feasible,
but the scientific gate fails: RGB-only shape deteriorates and ideal rope is
not yet a physical observation. Revisit scale only after a train/internal-val
change fixes RGB-only root-relative behavior without consulting S1 test.

For the same reason, do not add DexYCB to the frozen three-dataset joint
mixture. This one-shot test cannot be used to tune another mixture, and the
current RGB-only signal does not justify promotion.

## Artifacts

- Processed root:
  `/data/wentao/ropetrack/processed/dexycb/s1_v1`.
- Split/no-leak report: `split_report.json`; root protocol: `protocol.json`.
- Train/val/test manifests and targets:
  `train27k/{training.jsonl,training_xyz.json,training_mano.npz,protocol.json,rejected_samples.jsonl,selection_rejected_samples.jsonl}`,
  `val/{evaluation.jsonl,evaluation_xyz.json,evaluation_mano.npz,protocol.json,rejected_samples.jsonl}`,
  and corresponding files under `test/`.
- Run root:
  `/data/wentao/ropetrack/runs/direct_pose_dexycb_s1_20260720`.
- Coordinate gates and overlays:
  `coordinate/pretrain/{coordinate_gate.json,overlays/}` and
  `coordinate/final/{coordinate_gate.json,overlays/}`.
- Frozen recipe: `protocol/recipe_freeze.json`.
- Checkpoints: `students/rgb_only/model.pt` and
  `students/rgb_rope/model.pt`.
- External transfer: `external_transfer/summary.json` and
  `external_transfer/scores/scores.json`.
- Matched scores: `val/scores/scores.json` and the only official test score
  `test/scores/scores.json`; both have generated Markdown companions.
- Slurm states: `protocol/slurm_jobs.json`.
- Final verification: `protocol/artifact_verification.json`, status `PASS`.

The raw-tree before and after signatures are identical: 7,602,796 files,
140,786,432,785 bytes, SHA-256
`36e5f162b9c843340b83b0da54d4294a26aa5a99d7da60995847848b21174db3`.
The signature includes relative path, size, and `mtime_ns`, while excluding
only the known derived directories `img_feats`, `img_feats_dino`, and
`videos_v4`. Raw DexYCB was not downloaded, copied, moved, renamed, deleted,
or modified.

## Slurm record

The artifact-producing path below is also recorded machine-readably; every row
completed with exit code 0:

- `187896` stage1 prepare/gates;
- `187984` smoke scorer recovery;
- `187985_0-3` train27k/val WiLoR and token caches;
- `187986` bundle construction;
- `187987_0-2` frozen old-head apply; `187988` external score;
- `187989_0-1` matched training; `187990_0-1` S1-val apply;
- `187991` S1-val score and recipe freeze;
- `187992` frozen test export/final coordinate gate;
- `187993_0-1` test caches; `187994` test bundle;
- `187995_0-1` test apply; `187996` one-shot score/raw signature;
- `188038` final independent artifact verifier.

The first GPU smoke models and predictions were produced in `187970`, but the
job exited 1 only when the initial scorer expected `cam_t` instead of the
actual exported `base_cam_t`. Commit `3b31a69` fixed that field and six focused
tests passed; `187984` then scored the unchanged smoke artifacts successfully.

Earlier fail-fast attempts are retained rather than presented as successful:

- `187674` cancelled after exposing sentinel selection; dependent
  `187681`, `187685`, `187691-187696`, and `187705-187709` cancelled.
- `187712` failed the incorrect right-only assumption; `187713-187725`
  cancelled.
- `187765` failed the naive 29,440 BOP target equality; `187766-187775` and
  `187777-187779` cancelled. The missing ID is an unrelated job.
- `187801` cancelled after bbox rejects were detected; `187802-187814`
  cancelled. The partial derived export is preserved as
  `s1_v1_attempt4_rejected_bbox`.
- `187851` cancelled after detecting unacceptable sequence imbalance;
  `187852-187864` cancelled. The partial export is preserved as
  `s1_v1_attempt5_sequence_imbalance`.
- `187874` failed an over-strict equality rule, proving four sequences are
  capacity-bound; `187875-187887` cancelled. The partial export is preserved
  as `s1_v1_attempt6_capacity_bound`.
- `187897` failed for the missing shared pretrained-model symlink;
  `187898-187909` cancelled. `187913` then exposed an output-directory race;
  `187914-187925` cancelled.
- Queued chains `187930-187942`, `187944-187956`, and `187957-187969` were
  cancelled before execution while resource/time and unique-root handling were
  corrected. IDs between those ranges belong to unrelated cluster jobs.
- `187970` completed all smoke model artifacts but failed the final old scorer;
  downstream `187971-187982` was cancelled and replaced by the successful
  scorer `187984` plus final chain above.

No failed attempt read S1 test scores. The only official test score access is
`test/scores/test_score_access.json`, status `completed`, after the frozen
recipe.

## Code and tests

The tracked implementation adds three S1 dataset configs; minimal DexYCB
support in the shared hand-pose adapter/protocol/rope/oracle paths; preparation,
coordinate validation, recipe freeze, scorer, and artifact-verification CLIs;
DirectPose application support; documentation; and focused unit tests. No
file under `third_party/` was edited.

`python -m pytest -q tests` passes: **430 passed, 4 existing warnings**. A bare
repository-wide `pytest -q` is intentionally not the project test command: it
also discovers the isolated HaMeR/ViTPose submodule's upstream tests and fails
collection in the local lightweight environment without that backend's
`mmcv`/`pycocotools` stack. The RopeTrack-owned suite is green, and all remote
runtime/artifact checks above are independently PASS.
