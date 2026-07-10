# 0054 Temporal Stride-4 Results and Strict-Past V2 Jobs

Date: 2026-07-10

## Scope

Close the corrected HO3D v3 stride-4 temporal comparison, state the failure
mode honestly, and launch the next strict-past and dense-episode tracks from
pinned source commits. This note does not claim results for jobs that were
still pending at the final snapshot.

## Frozen validation selection

The validation-only search selected the causal K1 checkpoint at:

```text
/data/wentao/ropetrack/runs/temporal_grid_v2_20260710/
temporal_peak_tuning/flex15_gate010/dropout/
h1_d128_lr15e4_seed0/temporal_student.pt
```

Its teacher-alpha validation L1 was `0.00726503`, versus `0.01551176` for
the matched framewise MLP, a reduction of `0.00824673` or 53.2%. K1 still
contains a causal rope difference at `t-4`; it is not a purely current-frame
model.

The joint K4/K8 GRU did not beat K1. At lr `1e-3`, its best validation L1 was
`0.00944816` for K4 and `0.00973770` for K8. This is why the next model freezes
K1 and learns only a zero-initialized residual from strictly past slots.

## Corrected official stride-4 result

The protocol-safe evaluation subset has 5,066 rows in 70 contiguous sequence
segments. The exact score payload is:

```text
/data/wentao/ropetrack/runs/temporal_eval_stride4_20260710/
scores/temporal_scores.json
```

Lower is better except rope closure.

| Method | PA-MPJPE mm | PA-MPVPE mm | Velocity | Acceleration | Jitter | Rope closure |
|---|---:|---:|---:|---:|---:|---:|
| Base WiLoR | 9.939596 | - | - | - | - | - |
| Matched framewise | 9.114775 | 8.567327 | 50.85434 | 553.04379 | 544.88012 | 0.392953 |
| Released four-teacher MLP | 9.044195 | 8.505082 | - | - | - | 0.434539 |
| Selected causal K1 | **8.855625** | **8.253867** | 51.91108 | 568.07797 | 561.36285 | **0.569911** |

Against the matched framewise model, causal K1 improves PA-MPJPE by
`0.259150 mm`, with sequence-bootstrap 95% CI
`[-0.369864, -0.168930]`. PA-MPVPE improves by `0.313460 mm`, CI
`[-0.424876, -0.226237]`. It also beats the released MLP by `0.188570 mm`
PA-MPJPE and `0.251215 mm` PA-MPVPE.

The current result is an accuracy win, not a smoothing win. Velocity error is
`1.05674` worse than matched framewise, CI `[0.71768, 1.46642]`, and
acceleration is `15.034` worse, CI `[10.7849, 20.0211]`; jitter is about 3.0%
worse. Alpha/rope EMA controls did not recover the point score. Do not claim
the temporal smoothness gate has passed.

Fingertips did not regress. Occluded-tip PA error is `12.758731 mm`, versus
`13.398687 mm` matched framewise and `13.280732 mm` release. The improvements
are `0.639957 mm` and `0.522001 mm`, respectively. All-tip error is
`12.583614 mm` versus `13.292162 mm` matched; clean-tip error is
`12.343442 mm` versus `13.146063 mm`.

## Strict-past V2

Commit `879ff88fcd8fb4376d267624821eeb0dfa0a19dd` adds the schema-v2
strict-past residual model. Two independent reviews approved it and local plus
HPC full suites each passed 331 tests. Its fixed contract is:

```text
frozen selected K1 + strictly-past GRU residual + zero-init bounded head
train raw step 4; dense inference raw step 1; history/delta step 4
```

The pinned HPC worktree is:

```text
~/project/ropetrack_temporal_879ff88
```

Run root and jobs:

```text
/data/wentao/ropetrack/runs/temporal_past_v2_20260710
```

| Job | Partition | Purpose | Snapshot state |
|---:|---|---|---|
| 175942 | cpu | full tests plus frozen-K1 provenance | `COMPLETED 0:0`, 49 s |
| 175943_[0-1] | gpu | K4 hidden64, lr 3e-4/1e-3 | `COMPLETED 0:0`, 132/79 s |
| 175944 | gpu | shuffled-past at selected lr | `COMPLETED 0:0`, 100 s |
| 175945 | cpu | validation ranking and promotion gate | `COMPLETED 0:0`, 4 s |

Promotion requires at least `2e-4` validation-L1 improvement over frozen K1
and less than half that gain in the shuffled-past control. Only a passing
screen should expand to seeds 1/2 and K8.

The screen did not pass. At lr `3e-4`, the best real-history validation L1
was `0.00726160`, only `3.43e-6` better than frozen K1. Lr `1e-3` reached
`0.00726350`. Shuffled past reached `0.00726059`, a larger `4.44e-6` gain.
The recorded decision is therefore `history_specific=false` and
`promote=false`; seeds 1/2 and K8 were deliberately not submitted. On the
current stride-4 teacher, the additional past branch is fitting noise rather
than useful history. This does not replace the dense-episode test, where the
disturbance itself has temporal structure.

## Dense 30/60/30 episode track

Commit `1524f98e73e637aee5d34a788daf2549da118f25` adds the two dataset
configs. The roots contain 83,325 training frames and 20,137 evaluation
frames. Evaluation labels passed exact order/phase validation: 70 segments
and 153 complete episodes. The training-label array row was still running at
the snapshot.

Pinned worktree and output root:

```text
~/project/ropetrack_temporal_1524f98
/data/wentao/ropetrack/runs/temporal_episode_dense_wilor_20260710
```

| Job | Partition | Purpose | Dependency / snapshot |
|---:|---|---|---|
| 175908_[0-1] | cpu | dense rope labels and asset validation | eval complete; train running |
| 175949 | cpu | exact config/order/GT/protocol preflight | `afterok:175908` |
| 175950_[0-1] | gpu | dense train/eval WiLoR + MANO-cache exports | `afterok:175949` |
| 175951 | cpu | export integrity summary | `afterok:175950` |

The export keeps `gt_bbox`, `mano_vertices`, meters, and the active 32-sample
evaluation protocol check. Training bboxes use the existing projected-joint
fallback because HO3D v3 training metadata has no `handBoundingBox`.

## Do not repeat

- The flex15 teacher directory contains `alpha.npy` and
  `refiner_eval_cache.npz`, not `teacher.npz`.
- `load_temporal_checkpoint` returns four values: temporal model/config and
  nested framewise model/config.
- A local-path submodule clone on HPC needs command-scoped
  `-c protocol.file.allow=always`; do not change the global Git setting.
- Never use official test scores to retune the already frozen K1 selection.

## Next

1. Do not expand the failed stride-4 strict-past grid.
2. After dense exports pass, generate dense train teachers, train on the dense
   raw-step-1 episode cache, and evaluate recovery/masked/context phases.
3. Require both point accuracy and the original 10% jitter/acceleration gate;
   the current K1 result alone does not close the temporal objective.
