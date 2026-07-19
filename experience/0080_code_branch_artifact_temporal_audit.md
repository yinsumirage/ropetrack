# Code, Branch, Artifact, And Temporal Audit

Date: 2026-07-19

## Scope

Evidence-based audit of the current tracked entrypoints, the P2 release and
DirectPose artifact contracts, `main` plus both temporal branches, the detached
Codex worktree, and the old temporal/trusted-state route. No training or data
generation was run. Remote `/data` checks were read-only.

The stable result is `docs/current-code-and-artifact-map.md`.

## Code Findings And Cleanup

- `DirectPoseHead` is a real train/apply CLI with MANO-loss training, strict
  cache/sample alignment, episode-disjoint internal splitting, protocol/sample
  hashes, token/no-token controls, sensor perturbation controls, and a tested
  `weights_only=True` checkpoint load. It trains over frozen cached WiLoR
  tokens; it is not an end-to-end WiLoR backbone trainer.
- The current normal-data flow reuses tracked exporters, eval/MANO caches,
  token extraction, refiner cache construction, DirectPose train/apply, and the
  project scorer. Dataset merge, fold protocol, stitching, and Slurm submission
  scripts remain ignored run-local glue under
  `.local_checks/normal_joint_20260719`; exact copies and machine-readable
  protocols live under the remote run root.
- The P0-P2 release path remains independent and supported through
  `apply_rope_refinement.py --mode optimize|student`,
  `train_alpha_student.py`, and `RELEASE.md`. DirectPose does not replace its
  checkpoint.
- No tracked code was deleted. Every suspected old temporal file has at least
  one of: a current tracked importer, a dedicated test, checkpoint compatibility
  value, or unique experiment reproduction value. Deleting it would be a
  cosmetic cleanup that breaks evidence or old checkpoints.
- The earliest temporal schema remains reachable through the tested legacy
  loader in `ropetrack/refine/temporal.py`. It is compatibility code, not a
  recommendation to train another K16/K96 model.
- Markdown relative-link audit found no missing tracked links. Stable entry
  documents were updated instead of rewriting dated evidence.

## Branch And Worktree Evidence

Audited local and remote refs were synchronized before edits:

- `codex/temporal-oracle-state`: `997da919e6a146a571c89ce1cb22ab6defbbe852`;
- `codex/temporal-ho3d-refiner`: `ba31824f8828edb3d588372e010e2c78052481e3`;
- `main`: `bcb81dd2fbc82a207194376a54f1edd1e1ef45b5`.

Merge-base and left/right counts:

- `main...temporal-ho3d-refiner`: merge base `bcb81dd`, counts `0/22` in
  main/refiner order;
- `temporal-ho3d-refiner...temporal-oracle-state`: merge base `ba31824`,
  counts `0/50` in refiner/oracle order;
- `main...temporal-oracle-state`: merge base `bcb81dd`, counts `0/72` in
  main/oracle order.

Therefore `temporal-ho3d-refiner` has no code or record absent from the active
branch and needs no cherry-pick or merge. Keep the active branch; treat the old
refiner branch as a historical marker and delete it only after user approval.
`main` is the frozen release baseline and has no unique commit to recover.

The merge-base file diffs agree with the commit graph:

- `main...temporal-ho3d-refiner`: 26 files, 7,895 insertions and 49
  deletions, covering temporal configs/plans, `refine/temporal.py`, temporal
  train/apply/score code, and their tests;
- `temporal-ho3d-refiner...temporal-oracle-state`: 94 files, 9,832 insertions
  and 76 deletions, covering explicit-state follow-ups, new dataset adapters,
  DirectPose/orientation, the normal HO3D exporter, 0056-0079 records, and
  their tests.

The detached worktree at
`C:/Users/gwt/.codex/worktrees/f2fc/ropetrack` is at `e2f6d31` and is dirty:
`experience/INDEX.md`, `ropetrack/eval/protocols.py`, and
`tests/test_protocols.py` are modified, with untracked
`experience/0019_hamer_tip_policy_and_root_relative_loss.md`. Its tip-policy
code and conclusion are incorporated on the active branch, but its exact dirty
patch is not committed. Removing the worktree would discard that patch; this
audit did not edit or delete it.

## Temporal Decision

- **Validated on controlled synthetic occlusion:** explicit last-trusted visual
  state plus current rope/K1 helps through about 60 masked frames. State age
  crosses over near 120 and is harmful by 240.
- **Stopped:** dense K16/K96 versus K1, larger GRU/Transformer capacity,
  velocity grids, rope-residual arbitration, the mixed universal gate, and the
  tested simple natural-HOT3D visibility/usefulness gates.
- **Continue only once:** a sequence-disjoint localized DirectPose-state oracle
  and minimal gate, with whole subject/participant/sequence splits, an age-60
  cap, frozen clean/occlusion/dropout controls, and the numeric stop gates in
  `docs/current-code-and-artifact-map.md`. If its oracle does not clear the
  gate, stop temporal work rather than changing model class.

## Read-Only Remote Artifact Inventory

Important retained roots (exact apparent bytes from `du -sb`):

- P2 release copy: `367926` bytes at
  `/data/wentao/ropetrack/releases/p2_four_teacher_student`;
- DirectPose normal final run: `4913467946` bytes at
  `/data/wentao/ropetrack/runs/direct_pose_normal_joint_20260719`;
- DirectPose scale run: `135860552259` bytes;
- temporal state follow-ups: `42216671898` bytes;
- dense history: `21735072976` bytes;
- temporal oracle state: `13896676379` bytes.

The release directory contains only `manifest.json` (1545 bytes), `student.pt`
(350600 bytes), and `train_log.json` (11685 bytes); keep all three.

Large exact cleanup candidates, not deleted:

| Target | Bytes | Recovery and recommendation |
|---|---:|---|
| `/data/wentao/ropetrack/runs/direct_pose_scale_20260718/lora_cache` | 42833862316 | Recomputable activation cache; retain LoRA scores/checkpoints first, then delete only with approval. |
| `/data/wentao/ropetrack/runs/direct_pose_scale_20260718/lora_prepared` | 31237546265 | Recomputable sequential cache; same approval boundary. |
| `/data/wentao/ropetrack/runs/direct_pose_scale_20260718/shards_train` | 21935691786 | Recomputable training shards; expensive to regenerate, so delete only if storage pressure exceeds rerun cost. |
| `/data/wentao/ropetrack/runs/temporal_state_followups_20260715/oracle` | 32061674881 | Recomputable prediction/oracle arrays; preserve `scores/`, `scripts/`, checkpoints, and manifests first. |
| `/data/wentao/ropetrack/runs/temporal_oracle_state_20260715/oracle` | 13896506767 | Recomputable oracle arrays; preserve the small scores/scripts/manifests. |
| `/data/wentao/ropetrack/runs/dense_history_v1_20260711/methods` | 12836429731 | Recomputable stopped-model predictions; preserve scores and checkpoints needed for the negative result. |
| `/data/wentao/ropetrack/runs/dense_history_v1_20260711/teacher` | 8891520114 | Recomputable teacher cache, but costly; delete only after deciding no temporal reproduction is needed. |

HPC deletion is not immediately recoverable; "recomputable" means rerunning
from pinned code/data, not restoring from trash. Wait for explicit user
authorization before removing any target.

## Documentation Changes

- added `docs/current-code-and-artifact-map.md`;
- updated `AGENTS.md`, `CLAUDE.md`, `README.md`, `RELEASE.md`, and
  `scripts/README.md` to point at the current status and separate release from
  experiments;
- corrected the stale blanket reading of the 0026 rope-only 45D failure so it
  does not contradict the localized-token DirectPose result;
- added this note and the index row.

## Verification

- 18 current/frozen/temporal CLI help paths returned exit code 0, including
  DirectPose `train` and `apply`, P2 train/apply, temporal train/apply/score,
  oracle/state/gate CLIs, the HO3D normal exporter, and the scorer;
- `python -m pytest tests -q`: `404 passed`, with four known non-fatal warnings;
- Markdown relative-link audit: no missing tracked relative links;
- `git diff --check`: passed before recording this note and is required again
  before commit;
- no `third_party/`, data, prediction, checkpoint, metric, or large-figure file
  is part of the tracked diff.

## Do Not Repeat

- Do not infer activity from a filename or delete tested compatibility code for
  aesthetics.
- Do not cite the contaminated 0076 `HOT3D all` cell.
- Do not describe the reused 0077 HOT3D participant CV as an untouched test.
- Do not tune another mixture on the 0079 final scores.
- Do not equate GT-derived normalized rope with a physical sensor.
- Do not remove the dirty detached worktree or any remote artifact without
  explicit approval.
