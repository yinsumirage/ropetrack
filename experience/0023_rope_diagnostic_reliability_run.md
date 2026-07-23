# 0023 Rope Diagnostic Reliability Run

Date: 2026-07-05

## Purpose

Add and run report-oriented analysis for rope diagnostic reliability before any
training work.

## Code

Added:

- `scripts/rope_diagnostics/analyze_rope_errors.py`
- `tests/test_rope_diagnostics.py`

The script reads multiple `score_rope_predictions.py` output folders and writes:

- `run_summary.tsv`
- `hard_clean_delta.tsv`
- `per_finger.tsv`
- `gt_bin_summary.tsv`
- `worst_cases.tsv`
- scatter and worst-case figures under `figures/`

## Verification

Local:

- `python -m unittest tests.test_rope_diagnostics -v`
- synthetic plot smoke under `.local_checks/rope_diag_synthetic`

Remote:

- Synced the new script and test to `~/project/ropetrack`.
- `python -m py_compile scripts/rope_diagnostics/analyze_rope_errors.py`
- `python -m unittest tests.test_rope_diagnostics -v`

## HPC Run

Slurm job:

- `165869`
- CPU partition
- Completed with exit code `0:0`
- Runtime: `00:00:30`

Command shape:

```bash
python scripts/rope_diagnostics/analyze_rope_errors.py \
  /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/scores \
  /data/wentao/ropetrack/runs/rope_phase12_20260705_031056/diagnostics \
  --top-k 20
```

Output root:

`/data/wentao/ropetrack/runs/rope_phase12_20260705_031056/diagnostics`

Local copied output:

`.local_checks/rope_phase12_20260705_031056/diagnostics`

## Key Findings

- Largest clean-to-hard rope degradation is on FreiHAND `finger_end80` and
  `mask70`.
- HO3D v2 degradation is weaker; `mask70` is the clearest HO3D v2 hard split.
- Closed fingers show strong positive bias, meaning the model often predicts
  them as too open under hard occlusion.
- HO3D v2 closed-bin counts are tiny (`55` finger instances), so those bin
  numbers are caveats rather than main claims.

Stable summary:

The legacy standalone report was removed; this note is the retained result.
