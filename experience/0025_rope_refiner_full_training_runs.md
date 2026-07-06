# 0025 Rope Refiner Full Training Runs

Date: 2026-07-05

## Purpose

Run full FreiHAND hard-training cached refiner jobs for WiLoR and HaMeR after
the 512-sample smoke proved the scripts were connected.

## Stable Data

Evaluation rope labels were copied out of the throwaway run tree:

- `/data/wentao/ropetrack/rope_labels/freihand/evaluation_rope.jsonl`
- `/data/wentao/ropetrack/rope_labels/ho3d_v2/evaluation_rope.jsonl`

Full hard-training data:

- WiLoR-named root:
  `/data/wentao/ropetrack/hard/freihand/mask70_wilor_training`
- HaMeR-named root:
  `/data/wentao/ropetrack/hard/freihand/mask70_hamer_training`

Both roots contain 32560 rows in `hard_manifest.jsonl` and `rope_labels.jsonl`.

## Jobs

WiLoR data prep:

- Job `166770`
- Completed with exit code `0:0`
- Runtime: `00:33:32`

First WiLoR GPU attempt:

- Job `166773`
- Completed, but only ran 20 samples because `build_run_args` defaulted to
  `limit=20`.
- Treat this only as a smoke.

HaMeR copy attempt:

- Job `166795`
- Failed with `python: command not found`.
- Cause: script did not activate the conda environment before a metadata Python
  snippet.

Full WiLoR GPU rerun:

- Job `166837`
- Completed with exit code `0:0`
- Runtime: `00:10:04`
- `run_meta.json` reports `limit=0`, `num_samples=32560`, `num_failures=0`.

HaMeR copy rerun:

- Job `166838`
- Completed with exit code `0:0`
- Runtime: `00:03:10`

Full HaMeR GPU run:

- Job `166839`
- Completed with exit code `0:0`
- Runtime: `00:11:31`
- `run_meta.json` reports `limit=0`, `num_samples=32560`, `num_failures=0`.

## Results

WiLoR full training cache-level metric:

```json
{
  "num_samples": 32560,
  "base_pose_l1": 0.299074649810791,
  "refined_pose_l1": 0.2313011884689331,
  "mean_abs_delta": 0.08753085136413574
}
```

HaMeR full training cache-level metric:

```json
{
  "num_samples": 32560,
  "base_pose_l1": 0.29229843616485596,
  "refined_pose_l1": 0.21311041712760925,
  "mean_abs_delta": 0.10535528510808945
}
```

## Fixes

Changed the config default limit from `20` to `0`:

- full dataset is now the default;
- smoke runs must explicitly pass `--limit 20`.

Verification:

- Local focused tests: `31` passed.
- Remote `tests.test_eval_config`: `7` passed.

## Interpretation

These are still train-cache metrics, not benchmark metrics. They show the
cached refiner can fit the full hard-training pose target for both WiLoR and
HaMeR. The next required gate is held-out hard evaluation: apply the checkpoint
to hard-eval MANO caches, regenerate MANO vertices/joints, and score normal
FreiHAND metrics.
