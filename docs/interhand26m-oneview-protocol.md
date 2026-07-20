# InterHand2.6M v1.0 30fps One-View Protocol

Status: validated project external anchor and corrected train27k-v2 protocol.
This is not the official all-view InterHand leaderboard protocol. The first
train27k-v1 selector is invalid because it starved Capture9; see 0083.

## Data boundary

- Exact dataset name: **InterHand2.6M v1.0 30fps**.
- Read-only raw root:
  `/data/wentao/datasets/interhand2.6m_v1_30fps`.
- Canonical inputs are only `images/` and `annotations/`; `downloads/` is not
  an input.
- Derived root:
  `/data/wentao/ropetrack/processed/interhand2.6m_v1_30fps/oneview_v1`.
- Run root:
  `/data/wentao/ropetrack/runs/direct_pose_interhand26m_20260720`.
- RGB is referenced by relative path and is never copied.

The raw tree is protected by matching before/after signatures over sorted
relative path, file size, and `mtime_ns`. Test joint/MANO values and ideal rope
are inaccessible until `protocol/test_freeze.json` exists.

## Official semantics pinned by source

The coordinate gate pins the official Facebook Research InterHand2.6M source
at commit `5d0e456f2345ef524bf71374141fbf3e11dd93f8` and checks the source literals
and file hashes used below.

- `joint_3d` is world-coordinate millimetres.
- World to camera is `R @ (world - campos)`.
- Projection is `[x/z * fx + cx, y/z * fy + cy]`; camera axes are +x right,
  +y down, +z forward.
- Official joints `0:21` are right with root 20; `21:42` are left with root
  41. Each side maps to RopeTrack OpenPose order with
  `[20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16]`.
- NeuralAnnot pose is the full 48D axis-angle pose, shape is 10D, and
  translation is world-coordinate metres. Native decode uses side-specific
  MANO, `use_pca=False`, `flat_hand_mean=False`, the prescribed left
  `shapedirs` compatibility fix, and the official InterHand joint regressor.
- Global orientation is transformed from world to camera by the official
  rotation convention; axes are never selected by minimizing evaluation
  error.

`validate_interhand26m_coordinates.py` separately checks projection parity,
native MANO camera/root-relative errors, and the actual WiLoR-to-DirectPose
left-hand round trip. Its overlays use green for frozen GT projection,
magenta for an independent formula, cyan for native MANO, and red for WiLoR.

## `interhand26m_v1_30fps_oneview_v1`

The underlying unit is `(split, capture, sequence, frame)`, not a camera
image. The selected camera is the minimum SHA-256 rank of
`(protocol, frame_group_id, camera_id)`. The rule uses identifiers and input
metadata only.

- At most one camera view is retained per underlying frame.
- A valid single-hand annotation yields one sample; an interacting frame can
  yield paired left and right samples.
- Paired sides share `frame_group_id` and can never cross a split.
- `episode_id` is the complete `(split, capture, sequence)`.
- Per-side GT bboxes come from valid projected joints, add a fixed 25% margin
  on each axis, clip to image bounds, and reject fewer than four points, a
  span below four pixels, or an empty clipped box.
- Native occlusion visibility is not fabricated. `projected_in_frame` is
  reported only as a geometric image-boundary proxy, separately from 3D joint
  validity.
- Mesh metrics use only `mano_valid` rows. Joint-only and MANO-valid
  populations remain separate.

The project sample ID is
`split/Capture/sequence/camera/frame/side`; `frame_group_id` omits camera and
side. The manifest also records subject, camera intrinsics, the camera
transform reference, hand type/validity, side, joint validity, bbox, and
compact GT references.

## Train27k and internal validation

`interhand26m_train27k_oneview_v2` contains exactly 27,000 official-train
single-hand instances with complete 21-joint and native-MANO GT. Complete
frame groups are selected by capacity-constrained capture-then-episode
water-fill, with deterministic camera/hand-type/side strata inside each
episode. Every available capture is represented; non-exhausted captures differ
by at most one sample and low-capacity captures are proven exhausted. Subjects
0, 1, and 12 are excluded because `subject.txt` also assigns them to official
test captures; sample/frame/sequence/subject overlap is zero.

The historical v1 selector sorted all strata lexicographically and returned as
soon as it reached 27,000. It selected zero rows from Capture9 despite 6,554
available instances. Its supervised checkpoints and test scores are retained
only as diagnostics. The external anchor and frozen external-model scores do
not depend on that train selector and remain valid.

Internal validation selects complete episodes with NumPy seed 0 and a fixed
10% ceiling. Both DirectPose variants share the same IDs, bbox, WiLoR MANO
cache, localized 4x3 tokens, h128 head, AdamW recipe, seed, epoch budget,
early stopping, and minimum-internal-validation-PA checkpoint rule. The only
difference is `rope_mode=zero` versus `rope_mode=correct`.

## Frozen comparisons and scoring

`protocol/preval_model_freeze.json` is written before any InterHand val score.
It admits only checkpoints with checkpoint/train-log/protocol/sample hashes,
explicit input mode and apply command, plus an independent final-verification
PASS. The frozen external set is WiLoR base, all three normal-joint folds, and
the verified DexYCB RGB-only/RGB+rope pair. Older standalone ARCTIC
checkpoints without an independent PASS are excluded.

After official-val evaluation and matched training, `test_freeze.json` pins
the model list, checkpoints, candidate IDs, bbox buckets, scorer, bootstrap
recipe, and raw before-signature. Test GT export, ideal rope generation,
application, and scoring then happen once. A crash retry is permitted only
when every recorded frozen input hash is byte-identical.

Scores report PA-MPJPE, root-relative MPJPE, camera-frame MPJPE, MANO-valid
MPVPE, palm/global-orientation geometry, and root/translation separately.
Buckets cover side, single/interacting, capture, camera, valid-joint count,
projected-in-frame count, bbox size, and MANO-valid versus joint-only.
Absolute scores and paired signed deltas use frame-group bootstrap 95% CIs;
paired left/right rows are resampled as one cluster. Relative-root error is a
diagnostic only because the crop-camera translation is not promoted as a
trusted absolute two-hand root estimate.

Ideal five-rope input is derived from GT geometry. It is not no-GT RGB
inference and is not evidence for a validated physical rope sensor.

## First-round result

On 19,341 frozen official-val samples, corrected v2 RGB-only changes WiLoR by
`-0.384` mm PA (95% CI `[-0.398,-0.370]`) and `-0.359` mm root-relative, but
camera/MPVPE worsen by `+0.721/+0.651` mm. Adding ideal rope to the matched
head changes RGB-only by `+0.283` PA, `-0.423` root-relative,
`+1.138` camera, and `+1.281` MPVPE mm; every interval excludes zero.

Use the adapter and external anchor now. Continue RGB-only only for an
internal/official-val camera-frame tradeoff diagnosis. Stop this rope recipe,
old external transfer, full-view, larger train subsets, and mixture expansion.
The corrected v2 models were not rerun on test because the one-shot v1 test had
already been observed before the metadata-only selector audit.
