# 0053 Temporal Refiner Asset and Teacher Jobs

Date: 2026-07-10

## Scope

Launch the reusable HO3D v3 assets needed by the causal temporal-refiner plan,
while starting the expensive oracle teacher jobs in parallel. The first half
of this note preserves the submission state; later sections record completed
teacher upper bounds and the pinned retry of the evaluation export. Final
official/sliced retry scores are not claimed here.

The remote repo was fast-forwarded on branch
`codex/temporal-ho3d-refiner` from `9361c33` to plan commit
`4aeb3eb4c986e3979b4cc0917e410ffab81dfa0c`. The existing dirty
`third_party/wilor` submodule remained exactly as `m third_party/wilor`; it was
not edited, reset, cleaned, stashed, or updated.

## Four-Way Train-Asset Alignment

The login-node preflight used the `ropetrack` conda environment and compared:

- `/data/wentao/ropetrack/hard/ho3d_v3/mask70_train/train.txt`;
- `/data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl`;
- `run_meta.json` `sample_order` in the stride-4 export;
- `mano_cache.npz` `sample_id` in the same export.

The executed check was:

```python
import json
from pathlib import Path
import numpy as np

root = Path('/data/wentao/ropetrack')
train_ids = (root/'hard/ho3d_v3/mask70_train/train.txt').read_text().splitlines()
rope_ids = [json.loads(x)['sample_id'] for x in (root/'rope_labels/ho3d_v3/training_rope_s4.jsonl').read_text().splitlines()]
export = root/'runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor'
meta = json.loads((export/'run_meta.json').read_text())['sample_order']
with np.load(export/'mano_cache.npz') as z:
    cache_ids = [str(x) for x in z['sample_id']]
assert len(train_ids) == 20832
assert train_ids == rope_ids == meta == cache_ids
print('temporal-assets-ok', len(train_ids))
```

Exact output:

```text
temporal-assets-ok 20832
```

The same output is archived at `<run_root>/alignment.txt`.

## Run Root and Archived Scripts

The requested base path did not exist, so no timestamp suffix was needed:

```text
/data/wentao/ropetrack/runs/temporal_assets_20260710
```

The reusable ignored launcher remains at:

```text
~/project/ropetrack/.local_checks/submit_temporal_assets_v0.sh
```

The exact submitted scripts were copied under the run root:

```text
scripts/teacher_gate010.sbatch
scripts/teacher_ungated.sbatch
scripts/make_eval_assets.sbatch
scripts/export_apply_eval.sbatch
scripts/score_eval.sbatch
scripts/submit_temporal_assets_v0.sh
```

Submission output, the initial queue snapshot, job dependencies, and source
commit are also archived in `submissions.txt`, `squeue_initial.txt`,
`jobs.tsv`, and `manifest.tsv`.

## Submitted Jobs

| Kind | Job | Dependency | Partition/resources | Output |
|---|---:|---:|---|---|
| `oracle_chain/pose45/gate010` teacher | 174311 | - | gpu, 1 GPU, 16 CPU, 128G, 1 day | `teacher/oracle_chain_pose45_gate010` |
| `oracle_chain/pose45/ungated` teacher | 174312 | - | gpu, 1 GPU, 16 CPU, 128G, 1 day | `teacher/oracle_chain_pose45_ungated` |
| HO3D v3 eval hard root + rope labels | 174313 | - | cpu, 8 CPU, 64G, 12 hours | `/data/wentao/ropetrack/hard/ho3d_v3/mask70` and `evaluation_rope.jsonl` |
| WiLoR export + release-student apply | 174314 | `afterok:174313` | gpu, 1 GPU, 16 CPU, 128G, 1 day | `eval_export`, `eval/release` |
| official + sliced scoring | 174315 | `afterok:174314` | cpu, 8 CPU, 64G, 8 hours | `eval/release/scores`, `eval/release/sliced` |

Both teachers use the aligned 20,832-row stride-4 export and the fixed strong
recipe:

```text
--mode optimize --objective oracle_chain --action-space pose45
--opt-steps 400 --opt-lr 32 --opt-alpha-l2 0.001
--batch-size 512 --device cuda
```

Job `174311` additionally uses `--gate-residual-threshold 0.1`; job `174312`
has no gate. Both read
`hard/ho3d_v3/mask70_train/training_xyz.json`, the stride-4 rope labels,
`run_meta.json`, and `mano_cache.npz` from the known export.

Job `174313` runs the current `make_hard_images.py` and
`make_rope_labels.py` on raw `/data/wentao/ropetrack/HO3D_v3`, split
`evaluation`, with mask severity `0.70` and no sample limit. Job `174314`
runs:

```text
scripts/eval.py --dataset ho3d_v3_mask70 --method wilor_original
  --split evaluation --save-mano-cache --batch-size 64 --num-workers 0
scripts/rope_refiner/apply_rope_refinement.py --mode student
  --checkpoint /data/wentao/ropetrack/releases/p2_four_teacher_student/student.pt
  --dataset ho3d --device cuda --batch-size 512
```

Job `174315` runs `score_predictions.py` on the release `pred.json` and
`score_sliced_predictions.py` with the generated hard manifest and
`refiner_eval_cache.npz`.

Exact `sbatch` output:

```text
Submitted batch job 174311
Submitted batch job 174312
Submitted batch job 174313
Submitted batch job 174314
Submitted batch job 174315
```

Immediate post-submit `squeue` capture:

```text
             JOBID             NAME      STATE         NODELIST(REASON)                     DEPENDENCY
            174315       tmp_escore    PENDING             (Dependency)    afterok:174314(unfulfilled)
            174313       tmp_easset    PENDING               (Priority)                         (null)
            174314       tmp_eapply    PENDING             (Dependency)    afterok:174313(unfulfilled)
            174312         tmp_tung    PENDING               (Priority)                         (null)
            174311        tmp_tg010    PENDING               (Priority)                         (null)
```

A later spot check showed `174313` running on a CPU node while the two teacher
GPU jobs remained pending on priority and `174314`/`174315` remained pending
on their dependencies. None of these jobs had completed when this note was
written.

## Code-Review Hardening

A post-submission review found a critical provenance risk: every submitted
sbatch script used the live path `~/project/ropetrack`, so later branch updates
could change the code seen when a pending job finally starts. The submission
manifest correctly recorded source commit
`4aeb3eb4c986e3979b4cc0917e410ffab81dfa0c`; the first documentation-only
commit was `5d4d4642a652d28ee5756ca18b782f8e0cd7dc19`. This check proved that the
intervening commit did not change runtime code:

```text
git diff --exit-code 4aeb3eb..5d4d464 -- ropetrack scripts tests
# exit 0, no output
```

The HPC layout is now pinned as follows:

```text
/public/home/guowt2512/project/ropetrack               4aeb3eb (detached HEAD)
/public/home/guowt2512/project/ropetrack_temporal_dev  5d4d464 [codex/temporal-ho3d-refiner]
```

The primary worktree was switched non-destructively to detached `4aeb3eb` and
still reports exactly `m third_party/wilor`. Existing jobs `174311`-`174315`
continue to use that pinned primary path. A marker at
`~/project/ropetrack/.local_checks/PINNED_TEMPORAL_ASSETS_20260710.md` forbids
moving it until those jobs and validator `174393` are terminal. Future code
sync and new runs use `~/project/ropetrack_temporal_dev`; its submodules were
not initialized or edited. After this hardening record is pushed, only that
development worktree advances to the new branch tip; the primary remains at
`4aeb3eb`.

The review also found that successful file generation alone was not a strong
enough dependency gate. CPU validator `174393` was therefore submitted with
`afterok:174313`, using 4 CPUs, 32G, and a two-hour limit. It asserts exactly
20,137 rows and identical order across raw `evaluation.txt`, generated hard
`evaluation.txt`, hard-manifest `sample_id`, and evaluation-rope `sample_id`,
plus 20,137 rows in raw and generated xyz/verts GT JSON files. It completed:

```text
174393 tmp_evalid COMPLETED 0:0 00:00:55
temporal-eval-assets-ok 20137
```

To remove the race while changing a live pending dependency, `174314` was
temporarily held, updated to `afterok:174393`, verified while held, then
released and verified again. It was never started or cancelled. Once `174393`
completed, Slurm consumed the satisfied dependency and returned `174314` to
normal priority with `Dependency=(null)`. Job `174315` remains
`afterok:174314`.

Current state at the hardening check:

| Job | State | Meaning |
|---:|---|---|
| 174313 | `COMPLETED 0:0` | hard/rope asset generation finished |
| 174393 | `COMPLETED 0:0` | exact 20,137-row semantic gate passed |
| 174311 | `PENDING (Priority)` | gated oracle teacher not complete |
| 174312 | `PENDING (Priority)` | ungated oracle teacher not complete |
| 174314 | `PENDING (Priority)` | export/release apply not complete |
| 174315 | `PENDING (Dependency)` | scoring waits for 174314 |

The ignored future launcher was hardened but not rerun. It now creates a
per-run detached source worktree at a captured `EXPECTED_COMMIT`, pins WiLoR,
and makes every sbatch source a shared HEAD assertion. It uses
`sbatch --parsable`, collects IDs in the parent shell, and has an error trap
that cancels only IDs submitted by that invocation until job metadata is
complete. Eval assets are generated under a run-local staging root while an
atomic `mkdir` lock is held; the same 20,137-row/order gate runs before
individual atomic renames promote the hard root and rope file. Existing fixed
targets are never overwritten. Local and HPC-dev copies passed `bash -n`; only
static review was performed.

Updated run evidence is archived in `manifest.tsv`, `jobs.tsv`,
`submissions.txt`, `review_hardening_evidence.md`, and
`scripts/validate_eval_assets.sbatch` under the same run root.

## Completed Pose45 Oracle Teachers

Both independent GPU teachers completed successfully against the aligned
20,832-row HO3D v3 stride-4 training export:

| Teacher | Job | State / elapsed | Mean abs pose delta | Rope mean abs, base -> refined |
|---|---:|---|---:|---:|
| gated `oracle_chain/pose45/gate010` | 174311 | `COMPLETED 0:0`, `00:06:45` | 0.042826 | 0.087543 -> 0.047090 |
| ungated `oracle_chain/pose45` | 174312 | `COMPLETED 0:0`, `00:07:05` | 0.120744 | 0.087543 -> 0.048523 |

The complete optimizer summaries remain under
`teacher/oracle_chain_pose45_gate010/summary.json` and
`teacher/oracle_chain_pose45_ungated/summary.json`. The gated summary reports
`frac_samples_any_gated=0.6634`; the ungated result is deliberately a less
constrained ceiling, not a deployment policy.

CPU validation job `174436` independently decoded the generated MANO poses and
completed `0:0` in 22 seconds with exact output prefix
`teacher-joint-diagnostics-ok 20832`. Its archived
`teacher_joint_diagnostics.json` reports:

| Metric (mm; lower is better) | Base | Gated | Gated delta | Ungated | Ungated delta |
|---|---:|---:|---:|---:|---:|
| PA-MPJPE, all 21 joints | 7.4616 | 6.8944 | -0.5672 | 6.1705 | -1.2911 |
| PA tip MPJPE | 11.3916 | 9.5990 | -1.7926 | 6.9428 | -4.4487 |
| root-relative MPJPE, non-wrist 20 | 19.1429 | 15.6549 | -3.4880 | 8.7778 | -10.3651 |

These are GT-driven `oracle_chain` upper bounds on the training export. They
establish useful target headroom for temporal distillation but are not held-out
causal model scores.

## Evaluation Export Protocol Failure

GPU export/apply job `174314` failed `1:0` after 63 seconds, before model
initialization or inference. The generated hard root itself was not corrupt:
raw and hard `evaluation.txt` both had the same 20,137 IDs, and their
`evaluation_xyz.json` plus the first copied meta were identical.

The exact first sample was `SM1/0000`. Its evaluation meta stores
`handJoints3D` as a flat `(3,)` root vector:

```text
[0.025131747, -0.043128736, -0.527298212]
```

That is exactly `evaluation_xyz[0][0]`, so the correct root distance is zero.
The old check used `np.asarray(meta["handJoints3D"])[0]`, producing the scalar
`0.025131747`; NumPy broadcast that scalar against the three-coordinate GT and
created the false `0.556631m` mismatch printed by `174314`. A y/z axis flip
instead gives `1.058118m`, confirming this was a shape bug rather than a unit,
axis, root-selection, or sample-order error.

Commit `a62219bb79b8a470b3a64c4b1d45760b25bd5204` fixes only the root extraction
with `.reshape(-1, 3)[0]` and adds a real flat-meta regression. TDD captured the
old false failure at `1.414214m`; after the fix, 8 dataset-adapter tests and 24
related eval/config/hard-image tests passed, with `py_compile` and diff checks
clean. Both specification and code-quality reviews approved the two-file
change. The protocol check remains enabled.

## Pinned Evaluation Retry

The retry is isolated from both live repo worktrees at:

```text
/data/wentao/ropetrack/runs/temporal_assets_20260710/source_eval_retry_a62219b
```

It is a detached worktree at full commit
`a62219bb79b8a470b3a64c4b1d45760b25bd5204`. WiLoR was cloned from the pinned
primary runtime copy and checked out at gitlink
`fcb911312a38fa8badd30d9656a167485d61b8f9`; `mano_data` and
`pretrained_models` are symlinked from the primary runtime. Both newly
submitted retry jobs source `scripts/assert_source_eval_retry_a62219b.sh`,
which asserts both SHAs before doing work. Neither the detached primary
`4aeb3eb` nor the development worktree HEAD was moved to create this source.

The original targets `eval_export` and `eval/release` were absent before
submission, and the GPU script rechecks both at runtime before writing. The
exact archived retry scripts are:

```text
scripts/assert_source_eval_retry_a62219b.sh
scripts/eval_protocol_preflight_a62219b.sbatch
scripts/eval_export_apply_retry_a62219b.sbatch
scripts/submit_eval_protocol_retry_a62219b.sh
```

CPU preflight `174447` used 4 CPUs and 32G on `cpu`, imported from the pinned
source, built the real `ho3d_v3_mask70` adapter, and ran the active protocol
check on the first 32 samples. It completed `0:0` in 30 seconds with:

```text
ho3d-protocol-ok 32
```

GPU retry `174448` requests one GPU, 16 CPUs, 128G, and one day. It runs the
same original export and release-student apply commands as `174314`, without
`--protocol-check-samples 0`, under `afterok:174447`. Existing score job
`174315` was preserved and atomically changed from failed dependency
`afterok:174314(failed)` to `afterok:174448(unfulfilled)`; `scontrol show job`
confirmed it remained pending and cannot run before the retry succeeds.

| Job | Dependency | State at latest capture |
|---:|---|---|
| 174447 protocol preflight | - | `COMPLETED 0:0`, `00:00:30` |
| 174448 export + release apply retry | `afterok:174447` satisfied | `PENDING (Priority)` |
| 174315 official + sliced score | `afterok:174448` | `PENDING (Dependency)` |

Submission outputs, exact dependencies, source SHAs, before/after score-job
state, and initial job descriptions are archived as
`eval_retry_a62219b_{submissions.txt,jobs.tsv,manifest.tsv,score_before.txt,score_after.txt,state_initial.txt}`.
The completed CPU output and the post-preflight Slurm snapshot are additionally
archived in `eval_retry_a62219b_preflight_result.txt` and
`eval_retry_a62219b_state_after_preflight.txt`.

### Post-review retry hardening

Before GPU job `174448` started, an operations review found two provenance and
rollback gaps in the retry launcher. Exact-HEAD checks alone would not reject a
tracked or staged edit at the detached main checkout or inside WiLoR. The live
`assert_source_eval_retry_a62219b.sh` and all launcher copies now run
`git diff --quiet HEAD --` in both repositories after checking their expected
SHAs, with explicit errors on a dirty tree. This deliberately ignores the
required untracked/ignored asset symlinks. The live check passed for main
`a62219bb79b8a470b3a64c4b1d45760b25bd5204` and WiLoR
`fcb911312a38fa8badd30d9656a167485d61b8f9` while `174448` was still
`PENDING (Priority)`.

The original rollback could cancel only jobs submitted by that launcher, but
after changing score job `174315` it did not restore the old score dependency
if a later metadata step failed. The hardened launcher captures the original
`afterok:174314` dependency, sets `score_dependency_updated=true` immediately
after a successful `scontrol update`, and on a later error first restores that
dependency if `174315` is still pending, then cancels only the newly submitted
IDs. The launcher was not rerun; no live dependency or source HEAD was changed
by this review fix.

The local ignored launcher, HPC development copy, and archived launcher are
byte-identical with SHA-256
`f113ed65251e2ef70458a60d5413185041c0e0c30c8e1d8ed067453a8b74551f`.
The live assert script SHA-256 is
`ebd75236bac6a825b3a5477fef4c81879eeb94c9cd7e19aa272fc710d17a36b1`.
All copies passed `bash -n`. At the same verification snapshot, `174448`
remained `PENDING (Priority)` and `174315` remained pending with
`Dependency=afterok:174448(unfulfilled)`.

Automatic cleanup of a partially created pre-submission retry worktree remains
deferred: the launcher still fails closed on existing retry-owned paths so
partial setup is preserved for inspection. No current run path was deleted or
rewritten to address this minor review item.

## Next

After `174448` and dependent score job `174315` finish, verify `sacct` exit
codes, exact 20,137-row export order, official/sliced scores, and output
provenance before using the eval products. Keep the primary HPC worktree
detached at `4aeb3eb`; use the temporal development worktree only for branch
sync and the detached retry source for this run.
