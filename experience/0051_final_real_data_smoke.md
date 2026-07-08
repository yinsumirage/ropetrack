# 0051 Final Real-Data Smoke After Consolidation

Date: 2026-07-08

## Purpose

After the consolidation pass, local tests covered 214 unit tests with fake MANO
paths. This smoke ran the real WiLoR/MANO/export/student/analysis paths once on
HPC data before closing P2 and the release cleanup.

Run root:

```text
/data/wentao/ropetrack/runs/rope_final_smoke_20260708_191033
```

Jobs:

| Job | State | Note |
|---|---|---|
| `171715` | failed after S2 | Script used stale expected constants from the older 0042 single-student row. The actual S2 output matched the 0044 release row exactly. |
| `171797` | completed | Continued S2 recheck plus S3-S6; log ended with `ALL_FINAL_SMOKE_CHECKS_PASSED`. |

## Checks

### S1: WiLoR Export Smoke

Command shape:

```bash
python scripts/eval.py --dataset freihand_mask70 --method wilor_original \
  --limit 16 --save-mano-cache --run-eval --out-dir <run>/s1_export
```

Result:

- Loaded the real WiLoR checkpoint.
- Wrote `mano_cache.npz`, `run_meta.json`, `failures.json`, `eval_input/pred.json`.
- Ran the score subprocess on 16 samples.
- `run_meta.json` and `failures.json` were strict-JSON parseable.

### S2: Golden Release Student Regression

Release checkpoint:

```text
/data/wentao/ropetrack/releases/p2_four_teacher_student/student.pt
```

Inputs were the same mask70 WiLoR export used by 0044:

```text
/data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932/exports/mask70_wilor
```

Observed values:

| Metric | Smoke S2 | 0044 original `multi_mask70` |
|---|---:|---:|
| `xyz_procrustes_al_mean3d` | 0.843216059176498 | 0.843216059176498 |
| `mesh_al_mean3d` | 0.852896298133312 | 0.852896298133312 |
| `f_al_score_5` | 0.6252338560908393 | 0.6252338560908393 |
| `closure_frac` | 0.43981700010308744 | 0.43981700010308744 |
| `alpha.mean_abs` | 0.06708694994449615 | 0.06708694994449615 |
| samples | 3960 | 3960 |

The release-copy checkpoint and original run checkpoint also had identical
SHA256:

```text
5b279ef1dfca04eddaa15de0933e992eb9f8b1be0a3c19cfb1c29f2113da49fa
```

Note: the originally requested hard constants (`0.843164`, `0.437833`) are from
the older 0042 single-student row, not the 0044 four-teacher release model.

### S3: Default Optimize Mode

Ran `apply_rope_refinement.py` without `--mode` on the 16-sample S1 export.

Observed summary:

```text
mode=optimize
action_space=mult5
closure_frac=0.3586650317948814
alpha.mean_abs=0.07829778641462326
num_samples=16
```

This confirms the new naked-command default is `optimize` and that alpha plus
rope-residual reporting still writes on real MANO data.

### S4: Hard Root And Rope Label Generators

Ran:

```bash
python scripts/make_hard_images.py --dataset freihand --input-root <FreiHAND> \
  --output-root <run>/s4_hard --effect mask --severity 0.70 --limit 8
python scripts/make_rope_labels.py --dataset freihand --input-root <FreiHAND> \
  --output <run>/s4_rope.jsonl --limit 8
```

Result:

- Wrote `<run>/s4_hard/hard_manifest.jsonl`.
- Wrote exactly 8 rope labels to `<run>/s4_rope.jsonl`.

### S5: Analysis Tools

Ran:

- `summarize_runs.py` on S2 and S3, producing `<run>/s5_summary/runs_summary.tsv`.
- `make_qualitative_panels.py` on S2, producing two panels and
  `<run>/s5_panels/panels_manifest.json`.
- `score_rope_predictions.py` on S1 with the normal `run_meta.json`, producing
  `<run>/s5_rope_score/scores.json`.

### S6: Loud Failure Guard

Ran `score_rope_predictions.py` with `--run-meta /does/not/exist.json`.

Result: non-zero exit with:

```text
FileNotFoundError: run_meta does not exist: /does/not/exist.json
```

This confirms the post-consolidation sample-order loader fails loudly instead
of silently falling back.

## Conclusion

The consolidation and release hygiene changes are now validated on real data:
WiLoR export, MANO cache, release student apply, default optimize apply, hard
root generation, rope-label generation, analysis tools, qualitative panels, and
the loud run-meta guard all executed on HPC.

No new scientific numbers are introduced here; S2 is a regression check against
the already-recorded 0044 release row.
