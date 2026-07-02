# ropetrack

Bench-first code repo for wrist RGB + fingertip-to-wrist rope distance hand tracking.

The current goal is not a new training framework. This repo owns data manifests,
evaluation, visualization, experiment records, and thin wrappers around external
backends:

- HaMeR original
- WiLoR original
- AnyHand checkpoints / predictor style

## Layout

```text
configs/          Dataset and experiment config entry points.
data/             Local dataset links and generated manifests. No real data in git.
docs/knowledge/   Short local copy of the hand4D/now decisions.
experience/       Experiment notes, failures, and an index to avoid repeating work.
scripts/          CLI entry points once loaders/runners exist.
src/ropetrack/    Core schemas, IO, metrics, wrappers, eval, viz, rope code.
tests/            Small smoke checks.
third_party/      Git submodules for HaMeR, WiLoR, AnyHand.
```

## First Commands

```powershell
git submodule update --init --recursive
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Data Links

Keep datasets outside git and link them into stable paths:

```powershell
New-Item -ItemType Junction -Path data\raw\freihand -Target D:\datasets\FreiHAND
New-Item -ItemType Junction -Path data\raw\ho3d_v2 -Target D:\datasets\HO3D_v2
```

## First Milestone

1. Audit FreiHAND and HO3D v2 raw data.
2. Export small unified manifests.
3. Run HaMeR/WiLoR/AnyHand wrappers on 20 + 20 samples.
4. Save predictions in one schema.
5. Compute MPJPE and fingertip error.
6. Write findings under `experience/` before scaling up.

Generated predictions, metrics, and figures stay out of git. Current full-run
outputs live under the HPC data root, usually `/data/wentao/ropetrack/runs`.
