# ARCTIC P2 validation protocol and smoke

Date: 2026-07-17

## Scope

This note freezes the local hand-only ARCTIC protocol before the primary run.
It does not report a leaderboard result. ARCTIC test GT is hidden and no online
submission was made.

## Data and split

- Archive root: `/data/wentao/datasets/arctic/downloads`.
- Extracted root: `/data/wentao/datasets/arctic/unpack/arctic_data/data`.
- Official `splits.zip` is 18,602,586,094 bytes and passed its published
  checksum in Slurm job 183803. Base extraction job 183855 completed.
- P1 and P2 use the same subject-disjoint sequences:
  train `s01,s02,s04,s06,s07,s08,s09,s10`, validation `s05`, test `s03`.
- P1 is fixed views 1-8; P2 is egocentric view 0. `raw_seqs` omits `s03`, so
  trustworthy local scoring is P2 validation only.
- `p2_val.npy` contains 34 `s05` sequences and 25,203 unique view-0 images.
  Its order exactly matches the official first/last-10-frame exclusion.
- Official validity (`is_valid * hand_valid`) yields 20,005 right-hand and
  18,916 left-hand samples before any frame stride.

## Frozen alignment

- Annotation index is `image basename - misc[sid].ioi_offset`; the observed
  subject offset is applied rather than assumed.
- Cropped P2 images are 840x600 resizes of the distorted 2800x2000 ego image.
  Intrinsics scale only their first two rows by 0.3.
- Scoring uses official camera-coordinate view-0 3D, in OpenCV axes and metres.
- Side-specific GT boxes use distorted 2D view 9 scaled by 0.3, then a 0.25
  joint-range margin. This is the coordinate system that overlays the stored
  images; undistorted view 0 remains the official projection check.
- The official 21 joints are MANO kinematic joints in native 16+5 order. They
  are reordered to the WiLoR/OpenPose order for scoring and rope labels.
  Regressing posed vertices with `J_regressor` is not equivalent on ARCTIC and
  introduced 1.7-2.6 mm mean error in the audit, so it is rejected for joints.
- Direct native MANO decode matched official GT for both hands below 0.001 mm;
  K projection matched view-0 2D below 0.001 px. GT mesh is native right/left
  MANO decoded from official pose/shape/camera rotation and translated to the
  official wrist.

Three cross-sequence overlays (`capsulemachine_use_01`, `notebook_use_01`,
`phone_use_01`) and three frames from `box_grab_01` passed visual inspection
for both sides. View-0 versus distorted view-9 differences were 0.3-2.3 px in
the cross-sequence samples.

## Owned changes

- Added the manifest-backed `arctic` adapter and `arctic_p2_val` config.
- Added `scripts/prepare_arctic.py` with deterministic counts, hashes, P2 frame
  checks, GT boxes, reordered kinematic joints, and native MANO mesh export.
- Fixed the shared batch evaluator to mirror left output geometry back after
  flipped input inference. MANO cache now records `is_right`; release decoding
  mirrors left joints/vertices while old caches default to right.
- ARCTIC release decoding uses WiLoR MANO kinematic joints. A protocol test
  covers frame offset, 0.3 intrinsics scaling, and bbox construction.
- No third-party source was patched.

The complete repo-owned `tests/` suite passed: 377 tests. A bare repository-wide
collection also enters isolated ViTPose tests and needs its separate `mmcv`
environment, so it is not the RopeTrack regression command.

## GPU smoke

Slurm job 183930 ran 12 samples (8 right, 4 left), WiLoR original GT-bbox,
MANO cache, GT-derived rope labels, and the frozen release student. It had zero
base failures. The direct WiLoR export and cache decode agree to numerical
roundoff.

All errors below are millimetres; this early edge-frame smoke is a plumbing
check, not a primary result.

| method | raw joint | PA joint | raw mesh | PA mesh |
|---|---:|---:|---:|---:|
| WiLoR original | 104.2260 | 4.7237 | 104.3692 | 4.6787 |
| release Flex15 | 103.7975 | 4.6842 | 103.9485 | 4.6193 |
| Flex15 - base | -0.4285 | -0.0395 | -0.4207 | -0.0594 |

The student used its checkpoint gate threshold 0.1. Mean absolute rope
residual fell from 0.07736 to 0.06141; closure was 0.2062 and 25% of fingers
were gated. These values only prove the end-to-end path is active.

## Remaining primary gate

The 34 `s05` archives completed first, so job 183969 rehashes/extracts that
subject and unlocks stride-10/full P2 preparation (183960/183961) without
waiting for training subjects. The independent final data gate remains job
183954 after downloader 183768: rehash and extract all 339 archives and verify
every member/size while retaining the archives.

Run stride-10 (GPU 183962, score 183963) as a screen and the full P2 validation
result only if it passes. Report aggregate and side-specific raw/PA joint and
mesh errors, signed release deltas, failures, and rope closure. Object MR-E
metrics, P1, temporal methods, training, and online test metrics remain
unavailable/out of scope.
