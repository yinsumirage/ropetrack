# Dataset Contract Matrix

This is the current cross-dataset contract for RopeTrack. It records what is
normalized by project code, what was independently checked, and which metric
layers are safe to interpret. It is a protocol boundary, not an experiment
scoreboard.

Update this file whenever a dataset adapter changes its unit, coordinate frame,
joint order, handedness normalization, camera model, split, or GT source. A new
dataset is not ready for training or scoring until every required gate in the
last section has an explicit result.

## Project-level canonical contract

| Field | RopeTrack convention |
|---|---|
| Length unit | metres in manifests, GT arrays, predictions, and caches; reports convert to millimetres |
| Joint layout | 21 joints in OpenPose order: wrist, then thumb/index/middle/ring/pinky MCP-PIP-DIP-tip |
| Camera axes | OpenCV camera frame where supported: +x right, +y down, +z forward |
| Handedness | Dataset side is retained. WiLoR may flip a left crop to its canonical-right input, but exported points are mirrored back before camera translation |
| Sample identity | All cross-artifact joins use `sample_id`; positional joins without an order/hash check are invalid |
| Local pose claim | PA and wrist-relative joint metrics; valid only after unit, side, order, and GT gates pass |
| Global claim | Camera-frame metrics require a trustworthy camera model, projection convention, and translation path; local-pose validity alone is insufficient |
| Mesh claim | Requires native MANO/mesh GT consistency in addition to the 21-joint contract |

The canonical conversions are implemented in
`ropetrack/datasets/hand_pose.py`, `ropetrack/eval/protocols.py`, and
`ropetrack/eval/pipeline.py`.

## Dataset matrix

| Dataset / supported split | Raw GT and conversion | Camera and handedness | Verified gates | Safe interpretation | Known boundary |
|---|---|---|---|---|---|
| FreiHAND evaluation | Official `evaluation_xyz.json` / `evaluation_verts.json`; metres; MANO 16 joints plus FreiHAND tip vertices `[744,320,443,555,672]`, reordered to project OpenPose order | OpenCV-style camera; no HO3D axis flip; right hand | Official row order, GT bbox/projection path, clean baseline parity, joint/mesh scorer tests | PA, wrist-relative, camera-frame joint and mesh under the official evaluation protocol | Dataset is right-hand-centric and does not validate the project left-hand round trip |
| HO3D v2 evaluation | Official evaluation GT; model vertices use HO3D tip vertices `[744,320,443,554,671]`; model frame is converted with `[x,-y,-z]` to OpenCV camera metres | Right hand; explicit OpenGL-style to OpenCV y/z flip | Evaluation order/root protocol check and benchmark parity | PA and wrist-relative joint/mesh; camera only when the source/export translation convention is known to match | Keep v2 and v3 results separate |
| HO3D v3 train/evaluation | Train export keeps the official first 16 joints and replaces raw train tips with evaluation-convention MANO vertices before OpenPose reorder and axis conversion | Same as HO3D v2 | The rejected raw-tip path differed by `2.693 mm` mean on the 32-row diagnostic; corrected prepared 21-joint error is `2.996e-5 mm`; train/eval ID overlap is zero | PA and wrist-relative joint/mesh | Current raw camera error is metre-scale and convention-dominated; do not use it as a deployment claim |
| ARCTIC P2 train / `s05` val | Official camera-coordinate kinematic joints and side-specific native MANO; metres; OpenPose reorder; annotation index applies subject `ioi_offset` | View 0 image/camera; intrinsics scaled by `0.3`; distorted view-9 joints are used only to build the GT bbox; both hands retained | Subject-disjoint split; 16-sample joint/MANO smoke; six train overlays and visible cross-sequence val overlays; full val has 38,921 hands with zero inference failures | PA and wrist-relative local articulation; native root-aligned mesh checks | Global validation is less independently instrumented than DexYCB; do not claim a universal world/camera calibration from PA alone |
| HOT3D Aria public-GT | Native left/right MANO and kinematic joints in world coordinates; `T_world_device @ T_device_camera`, then world-to-camera; metres; OpenPose reorder | Aria stream `214-1`; timestamp-specific online `FISHEYE624`; both hands retained | Six full-fisheye projection overlays close side, transform, time-sync, and order gates on the smoke sequence; direct WiLoR export vs cache decode max difference `5.96e-5 mm` | PA and wrist-relative local joint/mesh diagnostics | Manifest stores only focal/principal values for WiLoR's pinhole crop interface; raw camera error is diagnostic until crops are rectified with the full fisheye model |
| DexYCB official S1 | Per-camera `joint_3d` is already OpenCV camera-frame metres; `joint_2d` is color pixels; `pose_m` is global axis-angle + PCA45 + translation; subject betas and side-specific native MANO | No extra camera extrinsic is applied because labels are already in each camera frame; both hands retained | Final gate: 304 samples, 10 subjects, 8 cameras, 152 left/152 right; reprojection mean/max `1.266e-5/1.365e-4 px`; native-MANO camera mean/max `6.131e-5/2.486e-4 mm`; split/no-leak verifier PASS | Strongest current contract: PA, wrist-relative, camera-frame joints, and native-MANO diagnostics | `pose_m[48:51]` is MANO translation, not the wrist joint; ideal rope remains GT-derived simulation rather than a physical sensor |
| InterHand2.6M v1.0 30fps one-view | Official world joints are millimetres; camera joints use `R @ (world - campos)` then `/1000`; official 42-joint skeleton is sliced per side and reordered to project OpenPose order | Pinhole projection `[x/z*fx+cx, y/z*fy+cy]`; deterministic one-camera-per-frame selection; both single and interacting hands retained | Gate: 433 train/val samples; reprojection mean/max `2.157e-5/3.976e-4 px`; WiLoR/DirectPose left-right round-trip mean/max `2.803e-5/1.520e-4 mm`; corrected protocol verifier PASS | Joint PA, wrist-relative, and camera-frame placement diagnostics with explicit separation | NeuralAnnot MANO vs joint GT mean is `5.952 mm` with sparse large outliers; mesh/MPVPE is label-fit-sensitive and secondary |

## Metric-grade summary

| Dataset | Local joint PA/root | Camera-frame placement | Mesh |
|---|---|---|---|
| FreiHAND | validated | validated under official protocol | validated |
| HO3D v2/v3 | validated | diagnostic in the current v3 DirectPose path | validated after the v3 fingertip correction |
| ARCTIC | validated | diagnostic / medium-confidence | native root-aligned checks validated |
| HOT3D Aria | validated | diagnostic until full-fisheye rectification | validated for the public-GT smoke/local protocol |
| DexYCB S1 | validated | coordinate contract validated; model `cam_t` quality is a separate issue | native-MANO contract validated |
| InterHand2.6M | validated | coordinate contract validated; WiLoR placement remains weak | secondary, label-fit-sensitive |

Large camera-frame error is not automatically a dataset-conversion failure.
For example, DirectPose changes only the 45D local MANO hand pose and cannot
repair frozen WiLoR camera translation. Conversely, good PA does not validate
camera placement because PA removes translation, rotation, and scale.

## Required gate before using a new or changed adapter

1. Freeze raw release name, split, sample ID, episode ID, and file signatures.
2. State source units and convert exactly once to metres.
3. State source coordinate frame and prove any world-to-camera or axis transform.
4. State the native joint order, project order, fingertip convention, and left/right handling.
5. Numerically reproject 3D GT to source 2D annotations across sides, subjects,
   cameras, depths, and visibility conditions.
6. Decode native MANO where available and compare camera and wrist-relative
   joints; report label-fit disagreement rather than fitting axes by search.
7. Render representative overlays, including both hands and difficult image
   boundaries/occlusions.
8. Verify sample-order hashes, cross-split overlap, exported-prediction/cache
   parity, and historical score parity before interpreting a new result.
9. Mark PA/root, camera, and mesh claims separately as `validated`,
   `diagnostic`, or `unsupported`.

## Evidence records

- HO3D normal-data correction and three-dataset no-leak result:
  `experience/0079_normal_joint_no_leak_final.md`.
- DexYCB S1 protocol and coordinate gates:
  `experience/0081_dexycb_s1_first_round.md`.
- InterHand one-view protocol and metric boundary:
  `experience/0083_interhand26m_oneview_first_round.md`.
- Five-dataset camera/root/T/RT/Sim3 decomposition:
  `experience/0084_direct_pose_error_decomposition.md`.
- Current code and artifact ownership:
  `docs/current-code-and-artifact-map.md`.
