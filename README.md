# ropetrack

Bench-first code repo for wrist RGB + fingertip-to-wrist rope distance hand
tracking: 5 rope scalars repair occlusion errors of RGB hand-pose backends.

This repo owns data protocols, hard-occlusion splits, rope labels, teacher
optimization, student distillation, evaluation, and experiment records, with
thin wrappers around external backends:

- HaMeR original
- WiLoR original
- AnyHand fine-tuned checkpoints through the local `HandPredictor`

## Layout

```text
configs/          Dataset and experiment yaml entry points.
data/             Local dataset links. No real data in git.
docs/             Current maps/protocols, retained P0-P2 report, and ignored report HTML.
experience/       Flat numbered evidence chain + timeline/topic INDEX.md.
scripts/          Categorized thin CLIs: datasets, evaluation, refinement, diagnostics, legacy.
ropetrack/        Core package: io, rope geometry, dataset adapters, eval, refine, viz.
tests/            Unit + toy-integration suite (the all-green invariant).
third_party/      Git submodules for HaMeR and WiLoR.
```

## First Commands

```powershell
git submodule update --init --recursive
python -m unittest discover -s tests
```

## Data Links

Keep datasets outside git and link them into stable paths:

```powershell
New-Item -ItemType Junction -Path data\raw\freihand -Target D:\datasets\FreiHAND
New-Item -ItemType Junction -Path data\raw\ho3d_v2 -Target D:\datasets\HO3D_v2
```

## Current State

Phases P0-P2 are closed: clean baselines reproduced, hard splits quantified,
rope test-time optimization diagnosed and frozen, and the release
alpha-student distilled and verified (train split -> held-out eval).

- Current code/artifact/branch/temporal map:
  `docs/current-code-and-artifact-map.md`.
- Documentation ownership and supported-file catalog: `docs/README.md`.
- Cross-dataset coordinate and metric contract: `docs/dataset-contract-matrix.md`.
- Release model identity and provenance: `RELEASE.md`.
- Retained P0-P2 report: `docs/2026-07-08-progress-report.md`.
- Cross-conversation working rules: `CLAUDE.md` (Claude) / `AGENTS.md` (Codex).
- The bootstrap milestone (tiny manifests, first 20+20 smoke) completed in
  week one; see `experience/0000`-`0013`.

The active post-release experiment is `DirectPoseHead`: a 45D local MANO
hand-pose residual head over frozen localized WiLoR tokens plus normalized
rope. `experience/0079_normal_joint_no_leak_final.md` is the normal-mixture
boundary; DexYCB, InterHand, and the five-dataset error decomposition are in
`0081`, `0083_interhand26m`, and `0084`. The product scope now assumes an
external VR/tracker supplies wrist 6DoF, while RopeTrack focuses on wrist-frame
local articulation. GT-derived rope still does not prove a physical sensor.

Generated predictions, metrics, and figures stay out of git. Current full-run
outputs live under the HPC data root, usually `/data/wentao/ropetrack/runs`.
