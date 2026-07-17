# HOT3D Aria public-GT protocol and smoke

Date: 2026-07-18

## Scope

This is an Aria-only plumbing and protocol smoke, not a primary HOT3D result.
Quest is intentionally excluded. The tested sequence is public-GT
`P0003_c701bd11`; 16 QA-filtered RGB hand samples were exported, eight per
side.

## Frozen sample contract

- Source images are raw `214-1` RGB frames decoded from `recording.vrs`.
- Camera pose is `T_world_device @ T_device_camera`; the camera extrinsic and
  native `FISHEYE624` projection come from the timestamp-specific MPS online
  calibration rather than a pinhole approximation.
- Samples must pass `mask_qa_pass`, `mask_good_exposure`, and
  `mask_headset_pose_available`. The hand-specific official visibility ratio
  must be at least 0.5.
- GT boxes come from `box2d_hands.csv` and are clamped to the 1408x1408 image.
  This matters because valid official boxes can extend past the raw frame.
- Native left/right MANO PCA pose, wrist world transform, and shared betas are
  decoded with the official provider. The 21 kinematic joints are reordered
  to the WiLoR/OpenPose convention. Vertices and joints are transformed to
  OpenCV camera coordinates in metres.
- Each manifest row also retains the source `pose_pca15`, `betas10`, and 4x4
  wrist-world transform. These are source GT parameters, not yet converted to
  the release model's canonical 45D axis-angle target.
- The processed root stores raw images, a manifest, joints, vertices, protocol
  hashes, and projection checks. WiLoR crops official boxes on demand; no
  duplicate offline crop dataset is created by this smoke.

The manifest's 3x3 intrinsic contains the focal lengths and principal point
exposed by the online calibration because the current WiLoR interface accepts
only a pinhole focal. It is not a complete representation of `FISHEYE624`.
Exact GT visualization uses the full fish-eye model. Consequently the PA
pose/mesh numbers below are useful smoke metrics, while raw global-translation
errors are diagnostic rather than benchmark-grade HOT3D claims.

The six native-fisheye projection checks place wrist and all five finger
chains on the visible hands for both sides, including near the distorted image
edge. Official boxes are sometimes loose or boundary-clipped but remain on the
correct hand. This closes the side, transform, time-sync, and joint-order gates
for the tested sequence.

## Owned changes

- Added `scripts/prepare_hot3d.py`, the `hot3d` manifest adapter/config, rope
  chain support, and release-student decode support.
- Added a small bbox/joint-order protocol test. The complete repo-owned test
  suite passes: 379 tests.
- No upstream HOT3D or WiLoR source was patched. The smoke used official HOT3D
  toolkit commit `146b34afef8c1a32adeef7e981c070109f225c87` and
  `projectaria_tools==1.5.1`.

## HPC run

- Job 185135 installed the official dependencies, then stopped on a transient
  GitHub TLS failure while cloning the toolkit. No dataset output was written.
- Job 185140 reused the locally verified official checkout and completed the
  16-sample CPU export plus six projection checks.
- Job 185158 refreshed only the manifest/protocol with source MANO parameters;
  joint and vertex SHA-256 hashes and the GPU sample order remained unchanged.
- Job 185144 completed WiLoR original, MANO cache, GT rope labels, the frozen
  four-teacher release student, aggregate/side scoring, and consistency checks.
  There were zero inference failures.
- The direct WiLoR export and cache-decoded base agree within
  `5.96e-05 mm` maximum coordinate difference for both joints and vertices.

All errors are millimetres:

| method | raw joint | PA joint | raw mesh | PA mesh |
|---|---:|---:|---:|---:|
| WiLoR original | 41.7449 | 4.6180 | 43.4261 | 4.6905 |
| frozen release student | 41.7237 | 4.6733 | 43.4016 | 4.7426 |
| release - base | -0.0211 | +0.0553 | -0.0245 | +0.0521 |

The base mean absolute rope residual was already 0.03152. The checkpoint's
0.1 gate activated only 1/80 valid fingers (one left thumb), so the student
was effectively identity: rope closure was 3.10%, while PA joint/mesh changed
slightly in the wrong direction. Right-hand scores are unchanged to numerical
roundoff; the left PA-joint change is `+0.1106 mm` because that is where the
single active thumb occurs.

## Interpretation and next gate

The end-to-end Aria path is valid, but this 16-sample static-keyboard slice is
too small and too easy to judge transfer value. Do not claim that the release
student helps or hurts HOT3D generally from it. The useful result is that raw
RGB, official crop, native MANO GT, left/right handling, fish-eye projection,
WiLoR cache, rope labels, student decode, and joint/mesh scoring now agree.

Before training, build a participant-disjoint Aria public-GT manifest and run a
larger action/visibility-diverse screen. Keep dynamic crops for evaluation;
only materialize crops once the training sampler and augmentation contract are
frozen. Root-relative pose/mesh training can use the current raw crops. Any
training or evaluation of absolute camera translation should first rectify
each crop to a pinhole camera with the timestamp-specific online calibration.
Quest remains out of scope unless Aria later exposes a specific gap.

## Durable paths

- Processed smoke: `/data/wentao/ropetrack/processed/hot3d/aria_smoke`
- CPU run: `/data/wentao/ropetrack/runs/hot3d_aria_smoke_20260718`
- GPU result: `/data/wentao/ropetrack/runs/hot3d_aria_smoke_20260718/gpu_185144`
