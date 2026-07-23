# Repository Package And Script Consolidation

Date: 2026-07-22

## Scope

After closing the current progress report, the repository was reorganized without
changing model equations, losses, dataset protocols, metrics, or checkpoints.
Stable reusable implementations now live under `ropetrack/`; `scripts/` contains
thin entrypoints and workflow-specific tools.

## Durable Dataset Contract

[`docs/dataset-contract-matrix.md`](../docs/dataset-contract-matrix.md) records the
shared sample/prediction conventions and the evidence boundary for FreiHAND,
HO3D v2/v3, ARCTIC, HOT3D, DexYCB S1, and InterHand2.6M. Any future adapter,
coordinate, joint-order, projection, or metric change must update that matrix and
rerun its relevant gates.

## Ownership Boundary

- Shared dataset logic: `ropetrack/datasets/`
- Shared scoring and metric logic: `ropetrack/eval/`
- Shared refiner implementations: `ropetrack/refine/`
- Dataset preparation/audits: `scripts/datasets/`
- Evaluation and analysis entrypoints: `scripts/evaluation/`
- Active refiner workflows: `scripts/rope_refiner/`
- Stopped temporal/state reproductions: `scripts/legacy/temporal/`

The moved command files remain thin wrappers, so current workflows have a clear
CLI while Python callers import package modules. Historical experience commands
remain evidence for their pinned commits; use `scripts/README.md` and
`docs/current-code-and-artifact-map.md` for current paths.

## Verification

Local checks:

- `python -m compileall -q ropetrack scripts tests`: pass.
- CLI help checks for the unified eval, InterHand preparation, generic scorer,
  DirectPose apply/head, and legacy temporal entrypoints: pass.
- `python -m pytest tests -q`: **451 passed**, with only pre-existing warning
  classes.
- No active script imports another script implementation; legacy temporal tools
  only import within their explicitly preserved legacy family.
- `git diff --check`: pass (one existing line-ending warning only).

Remote CPU-only parity smoke:

- Slurm job `191559`, account `engram`: **COMPLETED**, no GPU and no training.
- Re-scored the existing 16-sample HOT3D Aria smoke using the reorganized scorer.
- Exact historical parity was retained: raw joint **41.744855 mm**, PA joint
  **4.618033 mm**, raw mesh **43.426131 mm**, PA mesh **4.690451 mm**.
- Verification artifact:
  `/data/wentao/ropetrack/runs/repo_layout_smoke_20260722/output_191559/verification.json`.

Additional multi-dataset scorer parity smoke:

- Slurm job `191575`, account `engram`: **COMPLETED** in 24 seconds, CPU-only,
  no training and no artifact mutation.
- Re-scored existing 256-sample DexYCB and 256-sample InterHand2.6M smokes
  through the new `scripts/evaluation/` wrappers and package implementations.
- Both sample counts and sample-ID SHA-256 hashes matched the historical score
  files exactly.
- DexYCB PA joint results were exactly retained for WiLoR/RGB-only/RGB+rope:
  **5.160133 / 4.976567 / 5.016399 mm**. The checked root-relative and camera
  metrics also matched to the `1e-9` tolerance.
- InterHand PA joint results were exactly retained for WiLoR/RGB-only/RGB+rope:
  **7.158766 / 7.087882 / 7.135901 mm**. The checked root-relative, camera,
  and MPVPE metrics also matched to the `1e-9` tolerance.
- Verification artifact:
  `/data/wentao/ropetrack/runs/repo_layout_smoke_20260722/output_multidataset_191575/verification.json`.

## Boundaries

- `third_party/` was not modified.
- No model was trained and no benchmark result was promoted.
- The stopped temporal/state tools were preserved for reproduction, not restored
  as active research directions.
- The worktree already contained report/document cleanup and dataset notes; this
  consolidation was not staged, committed, or pushed as part of the check.
