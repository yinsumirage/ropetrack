# AGENTS.md

This repo is `ropetrack`, the implementation repo for the hand4D wrist bench
work. Treat `E:\Desktop\hand4D\now` as the external knowledge base and
`docs/knowledge/hand4d_now.md` as the local short version.

## Rules

- Keep third-party code isolated in `third_party/`. Do not edit submodules unless
  the user explicitly asks for a patch.
- Own only the outer code: data manifests, backend wrappers, evaluation,
  visualization, hard splits, rope labels, and experiment records.
- Benchmark first. Do data protocol and coordinate checks before training.
- Use one sample schema and one prediction schema for all backends.
- Do not commit raw data, processed data, checkpoints, predictions, metrics, or
  large figures.
- Prefer stdlib and existing code. Add dependencies only when a real script needs
  them.
- Before running an experiment, read `experience/INDEX.md`. After an experiment,
  write a short note and add it to the index.

## Current Research Line

Start with FreiHAND + HO3D v2, reproduce clean HaMeR/WiLoR/AnyHand baselines,
then build hard mask/blur/crop/appearance splits. Only after clean and hard
benchmarks are credible, test fingertip-to-wrist rope distance as post-opt,
then as model input/loss.

## Third-Party Backends

- `third_party/hamer`: https://github.com/geopavlakos/hamer
- `third_party/wilor`: https://github.com/rolpotamias/WiLoR
- `third_party/anyhand`: https://github.com/chen-si-cs/AnyHand

## Expected Schemas

Core sample fields live in `src/ropetrack/schema.py`: `sample_id`, `dataset`,
`split`, `image_path`, `hand_side`, `bbox_xyxy`, optional camera and label paths.

Core prediction fields live there too: `sample_id`, `backend`,
predicted joints/vertices/MANO paths, optional projected joints, bbox, hand side,
confidence, and coordinate notes.
