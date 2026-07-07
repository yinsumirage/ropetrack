# 0048 P3 v0 And Report Close Jobs

Date: 2026-07-07

## Context

After `0047`, the release model stays the four-teacher augmented multi student
from `0044`. This batch does two small report closeouts and starts the first
P3 v0 image-feature head experiment.

## Code Synced

Commit synced to HPC:

```text
cae6c12 Add P3 image feature student path
```

Main changes:

- `RopeAlphaStudent` now has configurable `in_dim`; old 65-d checkpoints still
  load through the default config fallback.
- `train_alpha_student.py --feature-cache` appends frozen image features to the
  existing 65-d rope/pose feature vector.
- `apply_rope_refinement.py --mode student --feature-cache` joins eval feature
  caches by `sample_id`.
- Guardrails reject missing features for image-feature checkpoints and reject
  image features for rope-only checkpoints.

Verification:

```text
python -m unittest tests.test_alpha_student
Ran 22 tests - OK

python -m unittest discover -s tests
Ran 225 tests - OK
```

## Submitted Batch

Run root:

```text
/data/wentao/ropetrack/runs/rope_p3_v0_and_report_close_20260707_222251
```

Submitted jobs:

| Job | Name | Dependency | Purpose |
|---:|---|---|---|
| 170579 | p3v0_rep | - | Report close: release model on HO3D v2 finger_end80; apply-level teacher/student latency timing on FreiHAND mask70 |
| 170580 | p3v0_te | - | P3 v0 train/eval: full image+rope head and rope-shuffled image-only control on FreiHAND mask70 |
| 170581 | p3v0_sc | 170579,170580 | Score/slice/deadzone and summarize the report-close and P3 eval cells |

Initial Slurm state:

```text
170579 p3v0_rep PENDING
170580 p3v0_te  PENDING
170581 p3v0_sc  PENDING (Dependency)
```

## Cells

Report close:

- `report_close/release_ho3d_finger_end80`
- `latency/teacher_mask70_apply.time`
- `latency/release_student_mask70_apply.time`

P3 v0:

- `p3_train/p3_head_v0`
- `p3_train/p3_head_v0_imageonly` (`--shuffle-rope`; image+pose-only control)
- `p3_eval/p3_head_v0_mask70`
- `p3_eval/p3_head_v0_imageonly_mask70`

## Decision Rule

The P3 v0 table is preliminary/next-plan material, not a blocker for the P2
report. Compare:

- rope-only release student: about `-1.64 mm` on FreiHAND mask70;
- P3 image-only shuffled control: image feature repair without rope;
- P3 full: image plus rope.

If P3 full approaches the strong oracle gap (`~ -1.8 mm` or better), pooled
features are useful enough for a short preliminary result. If full is roughly
rope-only, keep P3 as next-stage work and try token-grid features later.
