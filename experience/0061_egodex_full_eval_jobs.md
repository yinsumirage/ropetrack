# EgoDex broad test evaluation jobs

Date: 2026-07-16

## Scope

This is the first broad EgoDex compatibility evaluation, not a new public
benchmark protocol. It covers all 3,243 episodes in the native `test` split,
samples every tenth 30 FPS frame, exports both hands, and keeps the native
confidence values. The expected scale is roughly 75k RGB frames and up to
150k hand rows. Dense stride-1 temporal subsets remain deferred.

Raw and derived roots:

```text
/data/wentao/datasets/egodex/test
/data/wentao/ropetrack/processed/egodex/test_eval_s10
/data/wentao/ropetrack/runs/egodex_test_s10_20260716
```

EgoDex has 3D joint GT but no native MANO/mesh GT. Commit `a48c965` therefore
adds a `--joint-only-output` mode to the base evaluator and refiner. Prediction
JSONs retain all 21 joints and use null mesh rows, while the MANO cache still
stores the backend-predicted orientation, pose, shape, and camera translation
needed by the rope path. This avoids writing tens of GB of meaningless mesh
predictions. Mesh and F-score metrics must remain `-1`.

The same commit adds a deterministic inference-time shuffled-rope control. It
permutes `input_rope_norm` and its validity mask across samples but preserves
the clean target, base pose, and base rope prediction.

## Submitted dependency chain

```text
183269       CPU export all episodes at stride 10 + rope labels
183270       GPU WiLoR AnyHand GT-bbox base + predicted MANO cache
183271       CPU base joint scoring
183272_[0-4] GPU refinement/oracle array
183273_[0-4] CPU corresponding joint-score array (aftercorr)
```

Array `183272` cells:

| Index | Cell | Purpose |
|---:|---|---|
| 0 | `student_clean` | Released four-teacher flex15 MLP, zero-shot on EgoDex |
| 1 | `student_noise005_drop010` | Same release model with noise 0.05 and dropout 0.1 |
| 2 | `student_shuffle` | Destroy the sample-to-rope pairing; causal control |
| 3 | `teacher_rope_flex15` | Frozen 400-step rope/flex15/gate010 teacher recipe |
| 4 | `oracle_tip_flex15` | 400-step fingertip-joint oracle ceiling using EgoDex joint GT |

The oracle is valid without GT MANO because its objective is the native 3D
joint target and its output remains constrained to corrections of WiLoR's
predicted MANO state. It measures the flex15/MANO correction ceiling; it does
not turn predicted MANO into ground truth.

All GPU work is in Slurm jobs. CPU scoring is separated so GPUs are not held
idle. No synthetic occlusion or new confidence-mask protocol is part of this
batch.

## Interpretation guardrail

Report raw and aligned joint errors plus signed deltas versus the same base.
Do not compare mesh metrics. Treat the current all-confidence scorer as an
engineering compatibility screen: a publishable EgoDex number still needs a
frozen confidence policy and episode protocol.

