# 0024 Rope Refiner Train Smoke

Date: 2026-07-05

## Purpose

Implement the first rope-conditioned refiner training path and run a small
FreiHAND hard-training smoke on the HPC.

This is intentionally cache-level training first:

- hard training images are generated from FreiHAND training samples;
- WiLoR exports base MANO pose/cache on those hard images;
- the new `RopePoseRefiner` only trains a small MLP over cached pose and rope
  features;
- the backbone, detector, camera, global orientation, and third-party code stay
  frozen.

## Code

Added:

- `ropetrack/refine/rope_refiner.py`
- `ropetrack/refine/cache.py`
- `scripts/rope_refiner/build_freihand_refiner_cache.py`
- `scripts/rope_refiner/train_cached_refiner.py`
- `scripts/rope_refiner/eval_cached_refiner.py`
- `tests/test_rope_refiner.py`
- `tests/test_refiner_data_scripts.py`
- `tests/test_refiner_mano_cache.py`

Extended:

- `scripts/make_hard_images.py` supports FreiHAND `--split training`.
- `scripts/make_rope_labels.py` supports FreiHAND `--split training`.
- `scripts/eval.py` supports `--split` and `--save-mano-cache`.
- `ropetrack/eval/pipeline.py` can save cached MANO pose from batch exports.

## Verification

Local:

- `python -m unittest tests.test_eval_pipeline tests.test_rope_refiner tests.test_refiner_data_scripts tests.test_refiner_mano_cache -v`
- Result: 23 tests passed.

Remote:

- Same focused tests passed under the `ropetrack` conda environment after sync.
- A second focused `tests.test_rope_refiner` run passed after the cache metadata
  fix.

## HPC Runs

Run root:

`/data/wentao/ropetrack/runs/rope_refiner_smoke_20260705_061038`

CPU generation job:

- Job `166136`
- Completed with exit code `0:0`
- Generated 512 FreiHAND training hard-mask samples:
  `/data/wentao/ropetrack/runs/rope_refiner_smoke_20260705_061038/hard/freihand_mask70_train512`
- Generated 512 rope labels:
  `/data/wentao/ropetrack/runs/rope_refiner_smoke_20260705_061038/labels/freihand_mask70_train512_rope.jsonl`
- Generated 8 rope visualizations under:
  `/data/wentao/ropetrack/runs/rope_refiner_smoke_20260705_061038/viz/train_rope`

GPU export attempt:

- Job `166148`
- Failed quickly because the training split hard root has `training_xyz.json`,
  while the old protocol precheck still expected `evaluation_xyz.json`.
- Fix: only run the official eval protocol precheck for `split == evaluation`.

GPU export/cache attempt:

- Job `166164`
- WiLoR export succeeded and wrote:
  `/data/wentao/ropetrack/runs/rope_refiner_smoke_20260705_061038/export/freihand_mask70_train512_wilor_original/mano_cache.npz`
- Refiner training then failed because cache validation treated `finger_order`
  metadata as a sample-level array.
- Fix: validate first dimension only for explicit sample-level cache keys.

GPU train-only run:

- Job `166177`
- Completed with exit code `0:0`
- Runtime: `00:00:26`
- Reused the successful WiLoR export and trained/evaluated the cached refiner.

## Smoke Result

Export integrity:

- `num_samples`: 512
- `num_failures`: 0
- `num_bbox_candidates`: 512
- backend: WiLoR
- mode: GT bbox

Cache shapes:

- `base_hand_pose`: `[512, 45]`
- `target_hand_pose`: `[512, 45]`
- `base_rope_norm`: `[512, 5]`
- `input_rope_norm`: `[512, 5]`
- `rope_valid`: `[512, 5]`

Cache-level pose metric:

```json
{
  "num_samples": 512,
  "base_pose_l1": 0.29466676712036133,
  "refined_pose_l1": 0.2501317262649536,
  "mean_abs_delta": 0.05695416033267975,
  "delta_pose_l1": 0.05695416033267975
}
```

Interpretation:

- The first smoke proves the data path, MANO cache path, and train/eval scripts
  are connected.
- The cached refiner can reduce hand-pose L1 on the same 512 hard-training
  samples.
- This is not yet proof of benchmark improvement, because the current smoke does
  not regenerate refined MANO vertices/joints and does not evaluate on held-out
  hard evaluation samples.

## Caveats

- The current target is FreiHAND GT MANO hand pose from `training_mano.json`.
- This run is same-cache train/eval, so it is only an overfit/sanity result.
- The next meaningful gate is held-out hard evaluation: apply the trained
  refiner to cached hard-eval WiLoR MANO outputs, convert refined pose through
  MANO, write a normal prediction file, and score MPJPE/F-score.
