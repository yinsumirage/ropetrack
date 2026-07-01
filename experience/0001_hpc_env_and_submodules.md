# 0001 HPC Env And Submodules

Date: 2026-07-01

## Current Paths

- Repo on HPC: `~/project/ropetrack`
- Data root on HPC: `/data/wentao/ropetrack`
- Current downloaded but unprocessed data: `freihand`, `ho3dv3`

Do not record transient compute node names in repo docs. Slurm node names can
change; the stable fact is the path under the account.

## Environment Decision

Use two paths if needed:

- AnyHand/WiLoR first for smoke inference and baseline wiring.
- HaMeR separately if its `detectron2`, `mmcv`, `ViTPose`, or `chumpy` stack
  fights the WiLoR environment.

Current PyTorch is `torch==2.5.*`, so PyTorch 2.6 checkpoint loading behavior is
not the active blocker.

## Submodule Lesson

Running AnyHand helper scripts inside `third_party/anyhand` can modify AnyHand's
own `.gitmodules` or nested `third_party/hamer` state. Do not commit those
changes into upstream submodules unless intentionally making a fork patch.

Useful cleanup after accidental AnyHand submodule changes:

```bash
cd ~/project/ropetrack/third_party/anyhand
git restore --staged .
git restore .
```

This keeps untracked downloaded assets such as `mano_data/` and
`pretrained_models/`. The outer repo ignores untracked submodule downloads via
`.gitmodules`.

## Next

Run AnyHand/WiLoR smoke inference on a GPU allocation, then add only the minimal
outer wrapper needed to emit the ropetrack prediction schema.
