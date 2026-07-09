# Temporal Rope Refiner Design (2026-07-10)

## Status and decision

Approved for implementation on branch `codex/temporal-ho3d-refiner`.

Training may use HO3D v3 train-split GT and `oracle_chain` targets. Deployment
inputs remain wrist RGB-derived MANO predictions plus five rope readings;
evaluation remains causal and held out. The method claim stays narrow:
rope-assisted temporal repair under occlusion, not generic hand-pose SOTA.

## Evidence that constrains the design

- The removed 45D cached MANO MLP fit its train cache but worsened held-out
  FreiHAND mask70 PA by 0.91 mm (WiLoR) and 1.19 mm (HaMeR). A larger
  unconstrained pose head is not a valid default.
- The released framewise student is a 65 -> 256 -> 256 -> 15 `flex15` MLP
  with bounded alpha, hard `gate010`, and same-decoder evaluation.
- E2 showed that `rope/pose45` only improved 0.09 mm over `rope/flex15`, while
  `oracle_chain/pose45` opened roughly 1.62 mm of additional headroom. Temporal
  evidence must unlock information; output width alone does not.
- HO3D v3 has 83,325 train frames in 55 sequences and 20,137 evaluation
  frames in 13 sequences. The official lists contain frame gaps, so a temporal
  segment is `(sequence, maximal expected-step-contiguous run)`, not merely a
  sequence name.
- The existing stride-4 HO3D v3 mask70 rope teacher contains 20,832 aligned
  frames. It is suitable for a 7.5 fps fast track, but it must not be described
  as a 30 fps experiment.

## Considered approaches

### A. Causal filters only

Apply causal EMA to rope readings or framewise alphas. This is cheap, provides
the required smoothing/lag baseline, and needs no training. It cannot learn
which motion history is useful and may trade lower jitter for delayed recovery.

### B. Causal `flex15` residual student (safe mainline)

Encode a fixed causal window of the existing 65D frame features plus rope first
differences with a small GRU. Predict a bounded residual on top of the matched
framewise alpha student. Keep the current-frame hard gate and same MANO decode.
This is the lowest-risk learned version and should improve robustness, jitter,
and recovery, but its point-error headroom is limited by the `flex15` ceiling.

### C. Causal `pose45` oracle-distilled student (high-upside escalation)

Generate train-only `oracle_chain/pose45` alpha targets on HO3D v3 and train the
same causal model family to predict bounded 45D corrections. Compare against an
identical history-length-1 model. This is the route most likely to beat both the
old 45D MLP and the framewise `flex15` ceiling, but it carries the highest
overfit and clean-frame regression risk.

## Chosen architecture

The implementation supports B and C with one model and one checkpoint format.

Per timestep:

```text
base_hand_pose 45
base_rope 5
rope_reading 5
rope_residual 5
rope_valid 5
rope_first_difference 5
```

The model input is therefore 70D. The existing 65D feature normalization and
the five difference-channel statistics are fitted on training sequences only
and stored in the checkpoint. `context_valid` is a separate boolean mask, not
a normalized feature. Valid frames are packed in chronological order before a
single-layer GRU (default hidden size 128); padding cannot update hidden state.

For each action space, first train a matched framewise student on the exact
same sequence split and target teacher, then freeze it. The temporal head is
zero initialized and predicts a residual in bounded-alpha logit space:

```text
z_base = atanh(clamp(alpha_frame / max_alpha, -1+eps, 1-eps))
alpha_temporal = max_alpha * tanh(z_base + delta_logit)
```

Thus epoch zero is numerically the matched framewise baseline. The framewise
student is never jointly fine-tuned. Training minimizes
`L1(alpha_temporal, alpha_teacher) + 1e-4 * mean(alpha_temporal**2)`; no
smoothness loss is included in v1 because it would confound learned temporal
evidence with explicit filtering. The frozen base and temporal checkpoint
identity, bounds, normalization, action space, split, and seeds are all stored
in the checkpoint.

Targets are:

- the existing `rope/flex15/gate010` teacher for the safe mainline;
- train-only `oracle_chain/pose45` teachers for the high-upside branch. Both
  gated `gate010` (deployment primary) and ungated (capacity ablation) teachers
  are generated; checkpoint gate semantics must match the target teacher.

The current frame's rope residual gate is applied after temporal inference.
History must never open a currently closed gate. Global orientation, camera,
translation, backbone code, and `third_party/` remain frozen.

## Temporal data contract

1. Parse every sample id as `sequence/frame` (teacher prefixes are allowed).
2. Sort internally by sequence and numeric frame, but restore original row
   order on output.
3. Split at sequence boundaries and every unexpected frame gap.
4. Build only backward-looking windows; missing history is zero padded with a
   false context mask.
5. Split train/validation by underlying sequence before feature statistics,
   augmentation, or window construction. Related hard effects from one source
   sequence must stay in the same split.
6. The stride-4 fast track uses exact frame step 4. Dense evaluation may emit
   every frame while selecting history at `t-4, t-8, ...`.
7. A later dense/episode track will generate deterministic clean -> mask70 ->
   recovery phases and record phase/boundary metadata only for scoring. Episode
   phase is never a model input.

The split is deterministic: strip teacher prefixes, hash
`"20260710:" + sequence` with SHA-256, sort by `(digest, sequence)`, and place
the first `ceil(0.2 * num_sequences)` sequences in validation. For the current
55-sequence HO3D v3 train set this is 11 validation and 44 training sequences.
Validation teacher-alpha L1 selects hyperparameters and early stopping. The 13
official HO3D v3 evaluation sequences are never used for model selection and
are opened only for finalist reporting.

Dense episode generation resets at every sequence or raw-frame gap. Each full
120-frame cycle is exactly 30 unchanged context frames, 60 mask70 frames, and
30 unchanged recovery frames. Incomplete trailing cycles are emitted as
`tail` for alignment but excluded from episode aggregates. Train and eval use
the same phase lengths; mask geometry remains per-frame GT-bbox mask70.

## Augmentation and controls

- Framewise Gaussian rope noise may vary per frame.
- Bias and scale calibration errors are constant within a temporal segment.
- `zero_sensor` zeros reading, residual, first difference, and sensor validity
  before any noise/dropout path.
- `shuffle_rope` breaks rope/sample correspondence within a sequence.
- `shuffle_history` preserves the current frame and destroys only causal order.
- Required comparisons: raw base, release student, matched framewise student,
  alpha EMA, rope EMA, history length 1, true-history GRU, zero sensor, shuffled
  rope/history, and a centered non-causal smoother labelled oracle-only.

## Evaluation

All point metrics use the existing same-decoder HO3D evaluator. A new
numpy-only temporal scorer resets at each sequence/gap and reports:

- all-joint PA-MPJPE and PA-MPVPE;
- occluded-tip error and rope residual closure;
- root-relative velocity and acceleration error;
- prediction acceleration/jitter;
- episode masked-phase error;
- recovery frames and lag after episode boundaries;
- per-sequence bootstrap confidence intervals;
- apply latency.

Primary pass gate versus the matched framewise model on held-out sequences:

- masked/episode all-joint PA improves by at least 0.15 mm and the
  sequence-bootstrap 95% interval excludes zero;
- occluded-tip error does not regress;
- acceleration error or jitter improves by at least 10%;
- visible/clean PA stays within 0.05 mm;
- recovery is not slower by more than one frame.

Validation sequences are used only for early stopping and choosing finalists.
The primary pass gate above is applied once to the 13 official evaluation
sequences; validation results are reported separately and cannot satisfy the
final gate.

A result that lowers jitter by adding lag but misses the point-error/recovery
gates is a smoothing result, not a successful temporal refiner.

For each full episode and method, recovery is the first recovery-frame offset
`k` whose PA error is no more than that method's pre-mask 30-frame median plus
0.5 mm for three consecutive frames; unresolved episodes receive 30. Phase lag
is the signed lag in `[-15, 15]` maximizing Pearson correlation between
predicted and GT root-relative joint-velocity magnitudes over each entire
maximal contiguous segment of at least 31 frames (positive means prediction
trails GT). If either velocity series has zero variance, lag is `null`, that
segment is excluded from lag aggregation, and the excluded count is reported.
Sequence-bootstrap intervals use 2,000 sequence resamples with seed 20260710,
never frame resampling.

## Experiment sequence

1. Reuse the aligned stride-4 flex15 teacher and prepare one common dense HO3D
   v3 mask70 evaluation export.
2. Run causal filter baselines and sequence-disjoint framewise controls.
3. Sweep `flex15` history `{1, 4, 8, 16}`, hidden `{128, 256}`, then promote the
   best configuration to three seeds and stress tests.
4. In parallel, generate the train-only `oracle_chain/pose45` teacher.
5. Sweep matched framewise/temporal `pose45` models, then three seeds for the
   best configuration.
6. Build temporally coherent train/eval episodes only after the fast track
   proves the cache/model/scorer path, then repeat the finalists on episode
   metrics.
7. Record all non-trivial runs under `/data/wentao/ropetrack/runs`, archive
   launchers under ignored `.local_checks/`, and write an indexed `experience/`
   note with both positive and negative results.

The deterministic split seed is 20260710. Model/augmentation seeds are
`{0, 1, 2}` for promoted configurations; screening uses seed 0. Shuffling
controls reuse the paired model seed. Bootstrap seed is 20260710.

## Failure policy

- Fail loudly on duplicate ids, missing rows, cross-sequence windows, unexpected
  frame gaps, split overlap, checkpoint/action mismatch, non-finite values, or
  prediction/GT length mismatch.
- Do not silently delete failed frames and then bridge the resulting gap.
- Cancel jobs that download, use the wrong data path, or hold a GPU without
  doing GPU work.
- Preserve the released framewise checkpoint path and golden regression.

## Accepted risks

- The current sensor is GT-derived and still simulated.
- `flex15` may improve temporal quality without materially improving PA.
- `pose45` may overfit sequence dynamics; history-length-1, zero-sensor, clean,
  and shuffled-history controls determine whether any gain is real.
- The stride-4 fast track measures 7.5 fps context. Claims about 30 fps wait for
  the dense episode track.
