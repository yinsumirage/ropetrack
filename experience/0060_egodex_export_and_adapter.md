# EgoDex export, adapter, and first real-data smoke

Date: 2026-07-16

## Why this adapter is joint-first

EgoDex does not provide native MANO parameters or GT mesh vertices. Each
episode pairs a 30 FPS 1080p MP4 with an HDF5 file containing per-frame SE(3)
transforms for the camera and body/hand joints, a fixed camera intrinsic, and
usually per-joint confidence. Calling a fitted or backend-predicted MANO row
"EgoDex GT MANO" would be incorrect.

The useful first contract for RopeTrack is therefore:

- decoded RGB frame;
- one sample per hand, with side and a projected 2D bbox;
- camera-frame 21 x 3 joints in metres, ordered wrist then
  thumb/index/middle/ring/little with four joints per finger;
- the 21 native confidence values;
- episode and frame indices preserved in a temporal-compatible sample id.

`scripts/eval.py --save-mano-cache` adds **predicted** MANO orientation, pose,
shape, and translation later. This is already the cache contract consumed by
the rope teacher/student path, so a separate GT-MANO fitting stage is not
required for the current RopeTrack experiments.

## Real schema and projection audit

Raw test root: `/data/wentao/datasets/egodex/test`.

The first audited pair, `add_remove_lid/0.{hdf5,mp4}`, has 94 aligned frames,
1920 x 1080 video at 30 FPS, a 3 x 3 intrinsic, 69 transforms, and confidence
arrays. Wrist confidence was high in this episode, while individual occluded
fingertips could be much lower (index-tip means were about 0.43).

Camera conversion follows Apple's reference implementation exactly:
`inv(transforms/camera) @ transforms/joint`, with the resulting translation
passed directly to the provided intrinsic/OpenCV projection. Do not add the
HO3D y/z flip. Apple explicitly warns that the synthesized Vision Pro RGB has
a perspective mismatch, so projected joints and projected-joint bboxes can be
slightly displaced even when the 3D annotation is valid.

## Implemented path

- `scripts/prepare_egodex.py` streams the MP4, samples frames, exports JPEGs,
  writes `<split>.jsonl`, and writes aligned `<split>_xyz.json` without loading
  a whole split into memory.
- Both hands share a `<task>__<episode>` sequence key so an episode cannot leak
  across train/validation. Right-hand frame numbers receive a fixed offset,
  which makes the two hand streams unique contiguous segments while the raw
  frame index and side remain explicit manifest fields.
- `ropetrack.datasets.hand_pose` reads the manifest and exposes side-aware GT
  bbox candidates.
- `scripts/score_predictions.py` now accepts joint-only GT and writes `-1` for
  unavailable mesh/F-score metrics instead of fabricating mesh supervision.
- `scripts/make_rope_labels.py --dataset egodex` consumes the exported joints.
- The eval and differentiable teacher decode paths treat EgoDex joints like
  the FreiHAND/OpenPose 21-joint order; the alpha-student already joins generic
  teacher directories by sample id.

Keep raw archives/extractions under `/data/wentao/datasets/egodex` and derived
project subsets under `/data/wentao/ropetrack/processed/egodex`.

## Smokes

Final CPU job `183212` exported 20 hand samples from two test episodes at
stride 15 to `/data/wentao/ropetrack/processed/egodex/test_smoke_183212`: 20
manifest rows, 20 aligned GT rows, 20 rope-label rows, and 10 shared JPEGs.
Earlier visualization job `183183` rendered six joint/rope overlays. Both hand
sides landed on the corresponding visible hands; the small residual 2D offset
is consistent with the documented RGB perspective mismatch.

GPU job `183192` first passed the 20-sample WiLoR + MANO-cache + joint-only
evaluation path with no failed rows and PA-MPJPE 11.95 mm. It also exposed an
absolute-camera protocol bug: WiLoR's default assumed focal length made raw
MPJPE an impossible 3.89 m on EgoDex. The adapter now supplies EgoDex's real
736.63 px focal length to crop-to-full camera conversion. Corrected GPU job
`183195` again had zero failures, wrote the full `(20, 3)/(20, 45)/(20, 10)`
MANO-cache fields, preserved PA-MPJPE at 11.95 mm, and reduced raw MPJPE to a
plausible 125.52 mm. Mesh and F-score fields are `-1`, as intended for missing
GT mesh. These 20 low/unfiltered-confidence rows are a pipeline smoke, not a
benchmark claim.

GPU job `183206` then consumed the corrected predicted MANO cache plus EgoDex
rope labels through `apply_rope_refinement.py --dataset egodex`. The five-step
smoke wrote every normal teacher artifact for 20 rows, reduced mean absolute
rope residual from 0.22695 to 0.14114 (37.8% closure), and improved 81% of valid
finger rows. This proves data-contract compatibility; it is not a converged
teacher result.

## Guardrails and next use

- The test split is for evaluation/protocol development. Export `part1` as
  `training`, not as another test set.
- Do not decode the entire 829-hour dataset blindly. Start with stride 10 for a
  broad image-level screen; use stride 1 only on selected episode-disjoint
  temporal subsets.
- Confidence is preserved but the current compatibility scorer is unweighted.
  Do not publish an EgoDex benchmark number until a confidence-mask policy and
  fixed episode split are frozen.
- Projected-joint bbox margin is configurable and must be visually audited.
- A future GT-like MANO fitting stage, if needed for base-model supervision,
  must be labelled fitted/pseudo MANO and validated separately. It is not a
  prerequisite for RopeTrack rope-label and correction training.
