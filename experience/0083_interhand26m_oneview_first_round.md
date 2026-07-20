# InterHand2.6M one-view protocol and corrected DirectPose first round

Date: 2026-07-20

## Decision

- **Use now:** the `interhand26m_v1_30fps_oneview_v1` adapter, deterministic
  one-view external anchor, per-hand GT bbox, coordinate/left-right gates,
  frame-group bootstrap scorer, and frozen-test boundary are validated.
- **External transfer: stop the tested old heads.** Every frozen normal-joint
  and DexYCB DirectPose checkpoint is worse than WiLoR base on official-val and
  one-shot test PA/root-relative error. No old checkpoint transfers cleanly.
- **Corrected InterHand RGB-only: continue only as a narrow local-pose
  diagnostic.** On frozen official-val it improves PA by 0.384 mm and
  root-relative error by 0.359 mm, but worsens camera-frame error by 0.721 mm
  and MPVPE by 0.651 mm. It is not a promoted camera-frame model.
- **Corrected ideal rope: stop this recipe.** Relative to matched RGB-only it
  improves root-relative error by 0.423 mm, but worsens PA by 0.283 mm,
  camera-frame error by 1.138 mm, and MPVPE by 1.281 mm. All frame-bootstrap
  intervals exclude zero; the same tradeoff appears on single and interacting
  frames.
- **Do not run full-view, a larger train subset, or another mixture now.** The
  one-view anchor already exposes the global/camera/mesh tradeoff. Fix that on
  train/internal-val and a predeclared official-val analysis before scaling.
- **No corrected-v2 test rerun.** The first test score was already observed
  before a metadata-only capacity audit exposed the v1 train selector bug.
  Retesting new checkpoints would violate the one-shot boundary. The v1
  supervised test rows remain diagnostic only; WiLoR and frozen external-model
  test rows remain valid because they do not depend on the bad train subset.

## Data and immutable raw boundary

The exact release is **InterHand2.6M v1.0 30fps**, grounded in the
[official InterHand2.6M repository](https://github.com/facebookresearch/InterHand2.6M)
at commit `5d0e456f2345ef524bf71374141fbf3e11dd93f8`. It is not
Re:InterHand and is not claimed to be a newer InterHand release.

- Read-only raw root:
  `/data/wentao/datasets/interhand2.6m_v1_30fps`.
- Canonical inputs: only `images/` and `annotations/`; `downloads/` was never
  used for preparation, training, or scoring.
- Copy job `187560`: `COMPLETED`, exit `0:0`, elapsed `06:57:42`.
- Inventory: 5,716,488 train, 2,276,049 val, and 4,477,362 test JPGs;
  12,469,899 JPGs total. The 14 canonical annotation files and official
  `skeleton.txt`/`subject.txt` are retained under the raw root.
- Before and after signatures are identical: 12,469,915 files,
  373,564,519,095 bytes, SHA-256
  `1c49c87cefc3f89c0b9adff8045d531a71bbf8a99a9bed291ff95b416ba3dc64`.
  The signature covers sorted relative path, size, and `mtime_ns`. Raw data was
  not deleted, moved, renamed, or modified.

Audit evidence remains under
`/data/wentao/ropetrack/runs/handdata_audit_20260719/interhand` and the
experiment raw signatures are
`direct_pose_interhand26m_20260720/protocol/raw_signature_{before,after}.json`.

## Project one-view and single-hand protocol

The unit is `(split, capture, sequence, frame)`, not each camera image. The
minimum SHA-256 rank of `(protocol, frame_group_id, camera_id)` selects one
camera without labels or model error. Valid left and right hands become
separate samples but retain one `frame_group_id`; paired hands and other camera
views can never cross a split. This is a RopeTrack project anchor, not the
official all-view leaderboard protocol.

All models use the same side-specific GT bbox: valid projected official joints,
25% margin on each axis, image clipping, at least four valid points, and at
least four pixels of span. Detector error is out of scope. RGB is referenced
by relative path and is never copied.

| Export | Samples | Frames | Episodes | Side / hand type |
|---|---:|---:|---:|---|
| corrected train27k v2 | 27,000 | 25,001 | 2,076 | L 13,441 / R 13,559; single-L 10,771 / single-R 10,689 / interacting 5,540 |
| official val one-view | 19,341 | 15,641 | 11 | L 9,403 / R 9,938; interacting 9,361 |
| frozen test one-view | 40,824 | 31,629 | 635 | L 19,946 / R 20,878; interacting 21,020 |

The corrected train subset uses official train only. Subjects 0, 1, and 12
are excluded because `subject.txt` also assigns them to test captures. Exact
sample, frame, sequence, and subject intersections for train/val/test are all
zero. Internal validation is complete-sequence disjoint: 24,372 train and
2,628 validation samples, with zero episode intersection.

### v1 selector failure and v2 correction

The first `train27k_oneview_v1` manifest passed the original artifact verifier
but failed the later capacity audit: Capture9 had 6,554 available samples and
106 available episodes but zero selected rows. The selector iterated a
lexicographically sorted set of episode/camera/hand-type/side strata and
returned immediately at 27,000, starving late-sorted captures. Therefore the
two v1 InterHand-supervised checkpoints are protocol-invalid diagnostics.

Commit `051e4c11fe1e5a6cefc5c1f27c7966d04bf2034a` replaces that rule with a
capacity-constrained capture-then-episode water-fill while retaining stable
camera/hand-type/side strata and complete paired frame groups. The verifier now
requires the capacity report.

| Capture group | Available | Selected | Status |
|---|---:|---:|---|
| 2 / 3 / 5 / 6 / 7 / 8 / 9 | 6,554-9,114 each | 3,539-3,540 each | all non-exhausted, max-min 1 |
| 11-26 represented captures | 77-190 each | exactly all available | exhausted |

No available capture is starved; underfull non-exhausted capture and episode
lists are empty. The 76 selected cameras have 64-407 rows; rare cameras are
retained without duplicating frames.

Hashes:

- corrected v2 manifest:
  `0e1b8d4ac1cd3fca96e872a66d5805aa8f57664e0fd18dc130a981c9b3625971`;
- corrected v2 ordered sample IDs:
  `657bad5a814cdc2bdae06186929d7a7850adf875164bdf29d79212e996f5cfe2`;
- official val manifest:
  `d0aaf2188bf1588590d7c73a8401019eab72568d2624f2e888f7d311a702a80e`;
- frozen test manifest:
  `1182a11c8d021943166f9df0fb99b17f7215538aef9d2a603f17a6e735610010`;
- frozen test candidate IDs:
  `c8d71a7ba33cd7d09a7f337d73e656b039ed540698074025fb751ef56d6df776`.

## Coordinate, MANO, and left/right gates

Official source and local values jointly establish:

- world joints are millimetres; camera joints are
  `R @ (world - campos)`;
- projection is `[x/z*fx+cx, y/z*fy+cy]` with +x right, +y down, +z forward;
- official joints 0:21 are right/root20 and 21:42 are left/root41; the
  per-side RopeTrack mapping is
  `[20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16]`;
- NeuralAnnot uses full pose48, shape10, world-metre translation, side-specific
  MANO, `use_pca=False`, and `flat_hand_mean=False`.

The corrected pretrain gate covers 433 samples, train+val, 23 captures,
347 sequences, 89 cameras, right/left 219/214, and interacting/left/right
211/109/113. It saves 32 overlays under
`coordinate/pretrain/overlays/`.

| Gate | Mean | Median | P95 | Max |
|---|---:|---:|---:|---:|
| world→camera→2D reprojection px | 0.0000216 | 0.0000181 | 0.0000500 | 0.0003976 |
| native MANO camera-joint mm | 5.952 | 4.648 | 14.570 | 689.642 |
| native MANO root-relative mm | 5.897 | 4.415 | 15.702 | 690.774 |
| WiLoR/DirectPose left-right round trip mm | 0.0000280 | 0.0000167 | 0.0001193 | 0.0001520 |

The final frozen-test gate from the first run covers 563 samples and 30
captures with reprojection mean `0.0000246 px`. Its sparse MANO-fit outliers
reach 1,222.6 mm. The dedicated outlier audit finds 27/11/7/6/4 samples over
25/50/100/200/500 mm. Projection and side conventions remain exact, but MPVPE
is reported as a secondary label-fit-sensitive diagnostic. Joint GT and
MANO-valid populations are never conflated.

## Frozen external models

`preval_model_freeze.json` was written before any InterHand val score, and
`test_freeze.json` before test GT export or ideal rope generation.

| Method | Input | Checkpoint SHA-256 |
|---|---|---|
| WiLoR base | RGB | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| normal fold0 | tokens + ideal rope | `0f9f09953bdbf284f613c0c6336baf2c4c21fb097310cf826f7cc5873ca1d578` |
| normal fold1 | tokens + ideal rope | `114feee7e93323c6979656313cd255744bb202c26c93bff5a9b8ee6189ef50d5` |
| normal fold2 | tokens + ideal rope | `e3fdab428dcffbf9a22e75fa1659ccfb2e229a3fca9dd8f8343eab6efad30170` |
| DexYCB RGB-only | tokens, zero rope | `cc79acc38b4da74875ee8f7936241ffd15bce0c471fc582febba558962d48d04` |
| DexYCB RGB+rope | tokens + ideal rope | `327d6e9a80cd966cad5aa708646784b82cd8a55a7fbfd690aae0b0309fcc915a` |

The v1 InterHand checkpoints frozen into the one-shot test are
`de24f6b...ed996` and `2344d9a...63680`; they are now explicitly invalid for
supervision claims because of the Capture9 omission, not selected or replaced
using test error.

## Frozen external transfer

Lower is better. All rows use exactly the same official-val IDs and bbox.

| Method | PA mm | Root-relative mm | Camera mm | MPVPE mm |
|---|---:|---:|---:|---:|
| WiLoR base | 8.811 | 19.062 | 81.994 | 81.816 |
| normal fold0 | 10.197 | 20.138 | 83.220 | 83.171 |
| normal fold1 | 9.769 | 19.785 | 83.239 | 83.223 |
| normal fold2 | 10.141 | 20.278 | 83.514 | 83.487 |
| DexYCB RGB-only | 9.822 | 19.545 | 82.341 | 82.164 |
| DexYCB RGB+rope | 9.969 | 19.471 | 83.350 | 83.218 |

Every old-model PA delta versus WiLoR is positive with a 95% interval above
zero; for example normal fold1 is `+0.958 [0.928,0.988]` mm and DexYCB
RGB-only is `+1.010 [0.980,1.040]` mm. Root-relative deltas are also worse.
This is a clear stop, not mixed transfer.

The corresponding one-shot test external rows confirm the conclusion:
WiLoR is 8.055 PA / 16.011 root-relative; the three normal folds are
9.050-9.560 / 16.866-17.290, and the DexYCB pair is 9.176-9.425 /
16.396-16.602 mm.

## Corrected matched training and official-val result

RGB-only and RGB+normalized-five-rope share the exact v2 sample IDs, bbox,
internal split, WiLoR cache, localized 4x3 tokens, h128 head, AdamW recipe,
seed, epoch budget, early stopping, and minimum-internal-val-PA rule. Their
only difference is `rope_mode=zero` versus `correct`. Both select epoch 15.
Internal-val PA is 5.640 mm for RGB-only and 5.990 mm for RGB+rope.

Checkpoint hashes:

- RGB-only:
  `cda2bb16fe159e592478dda0bd2b87b3051ed6cfba4e1393c3e1bcf74898a7bc`;
- RGB+rope:
  `644f3569bac557d6dcf4b6632af071db9d73835dc6ea85a226f87857f7da7b3d`.

| Official val method | PA mm | Root-relative mm | Camera mm | MPVPE mm | Root mm | Orient deg |
|---|---:|---:|---:|---:|---:|---:|
| WiLoR base | 8.811 | 19.062 | 81.994 | 81.816 | 85.209 | 9.638 |
| corrected RGB-only | 8.428 | 18.703 | 82.715 | 82.467 | 85.209 | 9.638 |
| corrected RGB+rope | 8.710 | 18.280 | 83.853 | 83.748 | 85.209 | 9.638 |

Paired deltas use 2,000 fixed-seed bootstrap replicates over underlying frame
groups. Negative is better.

| Comparison | PA delta (95% CI) | Root-relative delta | Camera delta | MPVPE delta |
|---|---:|---:|---:|---:|
| RGB-only - WiLoR | -0.384 [-0.398,-0.370] | -0.359 [-0.376,-0.342] | +0.721 [+0.701,+0.740] | +0.651 [+0.634,+0.670] |
| RGB+rope - RGB-only | +0.283 [+0.257,+0.307] | -0.423 [-0.450,-0.396] | +1.138 [+1.107,+1.169] | +1.281 [+1.249,+1.314] |

The same qualitative result holds by hand type:

| Bucket / method | PA mm | Root-relative mm |
|---|---:|---:|
| single WiLoR / RGB-only / RGB+rope | 8.459 / 8.046 / 8.405 | 18.542 / 18.097 / 17.654 |
| interacting WiLoR / RGB-only / RGB+rope | 9.188 / 8.834 / 9.035 | 19.617 / 19.348 / 18.946 |

The DirectPose head changes articulated pose but retains WiLoR translation and
global orientation. Root and orientation deltas are therefore numerically
zero. MRRPE is unsupported: the crop-camera path does not provide a trusted
absolute two-hand root, and no relative-root value is fabricated.

### Historical v1 one-shot test diagnostic

For transparency, the already-observed v1-supervised test rows were RGB-only
7.853 PA / 15.955 root / 75.056 camera / 74.820 MPVPE and RGB+rope
8.304 / 15.666 / 75.797 / 75.981. Their paired deltas show the same tradeoff,
but they cannot validate supervision because the v1 train manifest omitted
Capture9. They are not used to tune v2, select a checkpoint, or justify a new
test run. The old test score file SHA-256 remains
`da4423beea66437599070201993e13f9e70dc6af61372a8239a7bbc4327972de`.

## Artifacts

- Original processed root:
  `/data/wentao/ropetrack/processed/interhand2.6m_v1_30fps/oneview_v1`.
- Corrected train root:
  `/data/wentao/ropetrack/processed/interhand2.6m_v1_30fps/oneview_v1_train27k_v2`.
- Original frozen external/test run:
  `/data/wentao/ropetrack/runs/direct_pose_interhand26m_20260720`.
- Corrected train/val run:
  `/data/wentao/ropetrack/runs/direct_pose_interhand26m_trainfix_20260720`.
- Corrected coordinate gate and overlays:
  `coordinate/pretrain/{coordinate_gate.json,overlays/}`.
- Corrected checkpoints: `students/{rgb_only,rgb_rope}/model.pt` with adjacent
  `train_log.json`.
- Corrected official-val score:
  `val/scores/{scores.json,scores.md}`.
- Corrected final verification:
  `protocol/final_verification.json`, status `PASS`.
- Archived exact Slurm scripts:
  `protocol/slurm_scripts/`.
- Original full job/attempt registry:
  `direct_pose_interhand26m_20260720/protocol/jobs.json`.

`test_freeze.json` records complete checkpoint, protocol, manifest, input-mode,
and apply-command provenance for the original one-shot model list. The
corrected run deliberately contains no `test/` directory.

## Slurm record

Original successful path: `188916`, `188932`, `189013`, `189039`,
`189049_0-3`, `189050`, `189051_0-4`, `189052`, `189053_0-1`,
`189054_0-1`, `189055`, `189056`, `189057`, `189058_0-1`, `189059`,
`189060_0-6`, `189061`, `189062`, `189087`, `189138`, `189218`, and
`189221` all completed with exit `0:0`.

Retained original attempts: `188933` failed on nested validity schema;
`189014` produced smoke artifacts but failed the old MANO scorer path;
`189000`, `188934`, `189001`, and `189151` were cancelled before a promoted
result. The successful replacements are recorded above; no failed attempt
selected a test model.

Corrected v2 path: `189295`, `189296`, `189297_0-1`, `189298`,
`189299_0-1`, `189300_0-1`, `189301`, and final verifier `189343` all
completed `0:0`. Verifier-only `189302` failed because its queued sbatch lacked
the newly added `--old-processed` argument; `189340` then exposed array-root
`sacct` parsing. Both ran after scoring, changed no artifact, and are retained.
The root-cause array parser fix produced the final PASS in `189343`.

## Code and checks

The tracked implementation adds three original dataset configs plus corrected
`interhand26m_train27k_v2`; minimal shared adapter/protocol/rope/cache support;
prepare, coordinate, scorer, freeze, and verifier commands; DirectPose apply
support; and focused tests. No `third_party/` file was edited and no dependency
was added.

The final local RopeTrack-owned suite passes: **443 passed, 4 existing
warnings**. `git diff --check` passes. The worktree also contains an unrelated
parallel EgoVerse experience/index change, which is preserved and excluded
from this task's commits.

## 2026-07-21 follow-up: rope-loss diagnosis and readable population slices

### DirectPose output clarification

The active head still modifies all 15 articulated MANO hand joints. Each
joint is a three-value axis-angle rotation, so the output is 45 scalar pose
residuals. The implementation groups these as five finger queries. Each query
receives that finger's three joint rotations (`3 x 3 = 9` values), four rope
features (`base`, `measured`, `measured-base`, `valid`), and a five-value
finger identity. It cross-attends to the frozen 4x3 image tokens and emits
nine bounded residuals. Thus the output is **9 per finger, 45 total**, not nine
for the whole hand. Betas, global orientation, and crop translation remain
the frozen WiLoR values. This differs from the older P2 student, whose output
was 15 action coefficients rather than a full 45D DirectPose residual.

### CPU post-hoc data and failure-region analysis

Run root:
`/data/wentao/ropetrack/runs/direct_pose_interhand26m_diagnostics_20260721`.
The analysis reads only the already-scored official-val artifacts; it does not
select a checkpoint and never accesses test.

- train27k: 27,000 samples / 25,001 frames / 2,076 episodes / 18 subjects /
  76 cameras;
- official val: 19,341 samples / 15,641 frames / 11 episodes / one subject /
  139 cameras;
- official-val bboxes are visibly larger and its out-of-frame-joint tail is
  much heavier than train/internal-val;
- 7,400 val samples belong to complete left/right pairs. Paired bbox-IoU
  median is 0.260 and P95 is 0.631. The median fraction of the other hand's
  joints inside the target crop is 0.476 and P75 is 0.810.

For the historical `RGB+rope+rope-loss - RGB-only` official-val predictions:

- increasing two-hand bbox overlap does **not** increase PA harm. The paired
  bins move from `+0.687` mm at IoU `<=0.01` to `-0.080` mm at IoU `>0.30`;
  the last CI `[-0.165,+0.003]` is effectively neutral;
- when 25-75% of the other hand's joints fall inside the target crop, PA
  improves by `-0.285/-0.206` mm. Extreme `>75%` occupancy regresses by
  `+0.306` mm. Two-hand contamination is therefore conditional, not the main
  average failure;
- rope-residual quintiles all improve root-relative error, but Q4/Q5 worsen PA
  by `+0.623/+0.439` mm. PA harm therefore concentrates in larger-disagreement
  cases, consistent with but not alone proving over-correction;
- PA harm is largest on thumb and ring. Interacting middle-finger joints are
  the only per-finger row with a small mean improvement (`-0.029` mm);
- native-MANO root-relative joint mismatch is 5.731 mm mean / 8.720 mm P95,
  but rope harm is not monotonic with this mismatch. Annotation/MANO
  inconsistency is not supported as the sole explanation.

The figures and machine-readable report are under `analysis/`:
`data_distribution.png`, `rope_effect_buckets.png`,
`per_finger_effect.png`, `qualitative_extremes.jpg`, `report.json`, and
`report.md`.

### Minimal input-versus-loss ablation

The follow-up kept the exact corrected train27k IDs, episode split, frozen
WiLoR cache, 4x3 tokens, h128 head, seed, optimizer, epoch budget, and
minimum-internal-PA checkpoint rule. It added only two cells with
`rope_weight=0`: correct rope input and shuffled rope input.

| Internal-val cell | Rope input | Rope loss weight | Best epoch | PA mm | Root-relative mm |
|---|---|---:|---:|---:|---:|
| prior RGB-only | zero | 0.1 (inactive because validity is zero) | 15 | 5.640 | 10.883 |
| prior RGB+rope | correct | 0.1 | 15 | 5.990 | 11.419 |
| correct input, no rope loss | correct | 0.0 | 27 | **5.400** | **10.630** |
| shuffled input, no rope loss | shuffled within train/val | 0.0 | 23 | 5.619 | 10.753 |

Removing the rope loss while retaining correct rope improves PA/root by
`-0.240/-0.253` mm relative to RGB-only and by `-0.219/-0.123` mm relative to
the matched shuffled-input control. Relative to the old correct-input cell,
removing the rope loss improves PA/root by `-0.589/-0.790` mm. This one-seed
internal screen supports a conflict in the auxiliary rope-consistency
objective, not the claim that InterHand rope carries no information.

Checkpoint SHA256:

- correct input, no rope loss:
  `d0903f09ff5fbe52b202d3a6e2f4ace3fb32554fe667115004de47ea69dc7103`;
- shuffled input, no rope loss:
  `6f85493631ca9e614ae3bd600db3bd2e8a90694c19f3795b00d465617dd87f4e`.

### Follow-up decision and operations

**Stop the current `rope_weight=0.1` auxiliary objective.** Keep the head and
correct rope input unchanged for now; no larger head, fusion rewrite, larger
train subset, mixture expansion, official-test rerun, or full-view run is
justified by this screen. The input-only result is promising but remains a
single-seed internal-validation diagnostic. A promotion attempt would first
need a seed repeat and a predeclared fresh external boundary, not adaptive
reuse of the already-observed InterHand test.

GPU array `189589_0-1` completed `0:0`. CPU analysis `189590` failed only
because the first diagnostic loader incorrectly treated the five-element
global `finger_order` field as a sample array; the shared sample-axis check and
regression test fixed it. CPU analyses `189597`, `189609`, and final figure
regeneration `189624` completed `0:0`. Verifier `189613` passed the
intermediate report; final verifier `189625` is the authoritative follow-up
verification and completed `0:0` with status `PASS`. Its final analysis-report
SHA256 is `6ddd0482cef89c6ba6e60d131e5e9ccc64a61d37bb3679fdd248777773bbd689`.
Local RopeTrack tests now pass **446 tests with the same four existing
warnings**, and `git diff --check` passes.
