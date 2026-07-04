# 0021 Rope Labels And Diagnostic Tools

Date: 2026-07-05

## Purpose

Start Stage 2 rope work without touching training code. This adds the first two
pieces only:

- GT rope-label export from existing FreiHAND/HO3D eval roots.
- Prediction-vs-label rope diagnostics for existing `pred.json` runs.

## Code

- `ropetrack/rope.py`
  - Dataset-specific 21-joint finger chains.
  - `rope_dist_m`, `rope_chain_m`, `rope_norm`, and `rope_valid`.
  - STMF-style normalization with `fist_ratio=0.5` by default.
- `scripts/make_rope_labels.py`
  - Writes JSONL labels.
  - Optional PNG visualization with projected finger chains and 5 rope bars.
  - When `--sample-order-file` is used, maps sample IDs back to canonical GT
    indices instead of assuming the requested order is a prefix.
- `scripts/score_rope_predictions.py`
  - Reads benchmark `pred.json`.
  - Uses `run_meta.json["sample_order"]` when provided.
  - Writes `scores.txt`, `scores.json`, and per-sample `rope_errors.jsonl`.

## Key Decision

For diagnostics, predicted rope norm is normalized using the GT label's
`rope_chain_m`, not the predicted hand's chain length. This prevents a false
good score when a prediction shortens the whole finger but preserves its own
internal normalized extension.

The raw prediction-derived distance is still recorded as `pred_rope_dist_m`.
Collapsed predictions are counted as errors when the GT finger is valid. A
zero-filled exporter fallback should not disappear from rope MAE just because
the predicted finger chain has zero length.

## Protocol Notes

- FreiHAND uses chains `[0,1,2,3,4]`, `[0,5,6,7,8]`,
  `[0,9,10,11,12]`, `[0,13,14,15,16]`, `[0,17,18,19,20]`.
- HO3D uses the hard-image/eval 21-joint fingertip convention:
  thumb tip `16`, index `17`, middle `18`, ring `19`, pinky `20`.
- The rope code does not use MANO vertex fingertip IDs. Those remain an eval
  export detail in `ropetrack.eval.protocols`.
- HO3D visualization flips Y/Z before projection, matching the hard-image
  builder.
- HO3D run-meta sample order is not assumed to match `evaluation_xyz.json`
  index order; the script builds `sample_id -> GT index` from the root.

## Verification

```powershell
python -m unittest tests.test_rope -v
```

Result: 8 rope tests passed; full local test suite also passed.

## Next

Run on HPC:

1. Generate HO3D v2 and FreiHAND clean rope JSONL labels with visual samples.
2. Score clean vs `finger_end80` baseline runs.
3. Record whether rope MAE rises on hard splits before starting refiner training.
