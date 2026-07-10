# Causal Past-Residual V2 Plan

**Date:** 2026-07-10  
**Branch:** `codex/temporal-ho3d-refiner`

## Evidence and decision

The corrected HO3D v3 sequence split produced a strong causal K1 model after
fixing the binary-validity normalization bug. Validation-only selection froze
the current finalist at teacher-alpha L1 `0.00726503`, versus `0.01551176` for
its matched framewise MLP. K1 is not current-only: its 70D feature contains a
rope difference at `t-4`.

The existing joint GRU does not preserve that strong path. At lr `1e-4`, K4
and K8 were only `0.01518460` and `0.01523412`. Job `174919` repeats K4/K8 at
the K1-scale lr `1e-3`; it is diagnostic only and cannot replace the already
frozen stride-4 finalist.

V2 therefore freezes the selected K1 checkpoint and learns only a strictly
past-conditioned, zero-initialized residual. It must never make current-frame
performance worse merely because a longer window exists.

## Model contract

```text
frozen nested K1(cache, current + t-4 delta) -> base_alpha
strict past slots only -> encoder -> one-layer GRU -> zero-init logit residual
bounded(base_alpha, residual) -> final_alpha
```

- The past branch never reads the current slot.
- Rows with no valid past return K1 exactly, even after training.
- Epoch zero and a no-improvement checkpoint equal K1 exactly.
- K1 and its nested framewise model are frozen and embedded in the V2
  checkpoint; the original checkpoint files are not runtime dependencies.
- Training loss is teacher-alpha L1 plus `1e-4 * mean(logit_residual**2)`.
  It does not penalize the already-good base alpha.
- The current-frame gate remains the last operation in apply.

Use an anchor-difference bounded transform so zero residual is bitwise identity:

```python
base_logit = atanh(clamped_base)
anchor = max_alpha * tanh(base_logit)
candidate = max_alpha * tanh(base_logit + delta)
output = clamp(base_alpha + candidate - anchor, -max_alpha, max_alpha)
```

## Cadence contract

Do not conflate cache continuity with temporal lag:

```yaml
train_raw_frame_step: 4
inference_raw_frame_step: 1
history_step: 4
feature_delta_step: 4
```

`temporal_features(..., raw_frame_step=1, delta_step=4)` may find `t-4` only
inside a maximal raw-step-1 segment. A missing `t-1` resets the segment even if
an older `t-4` row exists. Existing V1 checkpoints retain their current schema
and behavior.

## Implementation tasks

1. Add separate feature-delta cadence with causality/gap tests while preserving
   the old default (`delta_step=None` means `raw_frame_step`).
2. Add `PastResidualRopeAlphaStudent` with strict-past packing, bitwise K1
   identity, and bounded output tests.
3. Add a self-contained schema-v2 checkpoint containing the frozen K1 payload;
   reject action, gate, split, teacher, dimension, and cadence mismatches.
4. Extend temporal training with a mutually exclusive
   `--base-temporal-checkpoint` path. Optimizer parameter tests must prove that
   K1 and framewise weights are frozen.
5. Dispatch schema V1/V2 in the existing temporal apply mode. Add
   `--temporal-disable-history` as the exact same-cadence K1 control.
6. Run the full local and HPC CPU suites before V2 GPU training.

## Required tests

- Zero-init and no-past output are `torch.equal` to nested K1.
- Changing the current slot cannot change the past residual; changing a past
  slot can.
- Future frames, other sequences, and pre-gap frames cannot affect the current
  result.
- Dense frame 8 uses history `[0, 4, 8]` and delta `rope[8]-rope[4]`;
  deleting any intervening raw frame resets both.
- Shuffled-past changes only the past prefix.
- V2 loads after deleting the source K1 checkpoint.
- Existing V1 roundtrip/apply tests remain unchanged.
- If validation never beats K1, saved V2 remains zero residual with
  `best_epoch=-1`.

## Minimal HPC gate

Train only flex15 at first:

```text
K4, past hidden 64, lr {3e-4, 1e-3}, seed 0
K4 shuffled-past control at the better lr
same-cadence nested K1 via --temporal-disable-history
```

Promote only if K4 improves validation alpha L1 by at least `2e-4` over the
frozen K1. Then run seeds `{0,1,2}`, K8, and the dense 30/60/30 episode track.
The final pass gate remains at least `0.15 mm` masked PA improvement with a
sequence-bootstrap interval excluding zero, no occluded-tip regression, and
at least 10% jitter/acceleration improvement. Shuffled past must lose the gain.
