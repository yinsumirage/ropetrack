# 0000 Repo Bootstrap

Date: 2026-07-01

## Decision

The repo starts as an outer benchmark/wrapper repo:

- core package: `src/ropetrack`
- experiment workspace: `experiments/`
- repo memory: `experience/`
- third-party code: `third_party/` submodules

## Why

The hand4D/now notes say the fastest credible path is clean data protocol,
baseline reproduction, hard benchmark, then rope post-opt/model work. A unified
training framework would delay the first trustworthy table.

## Skipped

- Placeholder training framework.
- Dataset conversion scripts without real data paths.
- Large dependencies before a script needs them.
- Editing HaMeR/WiLoR/AnyHand internals.

## Next

Link FreiHAND and HO3D v2, then write the smallest raw audit scripts.
