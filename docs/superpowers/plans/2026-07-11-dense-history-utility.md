# Dense History Utility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Determine whether clean HO3D v3 context causally improves correction during later mask70 occlusion.

**Architecture:** Reuse the registered 30/60/30 episode data and existing flex15 teacher, matched framewise student, dense K1, and schema-v2 strict-past residual. Gate the pipeline on a phase sanity score, compare K16/K96 with same-cadence K1, and require shuffled history to lose the gain.

**Tech Stack:** Python, PyTorch, NumPy, existing RopeTrack scripts, Slurm, one GPU per training/apply job.

---

## File map

- Create ignored HPC scripts under `.local_checks/dense_history_*.sbatch` and
  `.local_checks/setup_submit_dense_history.sh`.
- Reuse `scripts/rope_refiner/apply_rope_refinement.py`,
  `scripts/rope_refiner/train_alpha_student.py`,
  `scripts/rope_refiner/train_temporal_student.py`, and
  `scripts/score_temporal_predictions.py` unchanged.
- Record terminal results in `experience/0055_dense_history_utility.md` only
  after jobs finish.

### Task 1: Phase sanity and source gate

**Files:**
- Create: `.local_checks/dense_history_phase_gate.sbatch`

- [ ] Run the temporal/refinement scorer tests from pinned source
  `~/project/ropetrack_temporal_1524f98`.
- [ ] Score base WiLoR with:

```bash
python scripts/score_temporal_predictions.py \
  --method base=/data/wentao/ropetrack/runs/temporal_episode_dense_wilor_20260710/export/ho3d_v3_temporal_episode_eval_wilor_original/eval_input \
  --reference base \
  --gt-dir /data/wentao/ropetrack/hard/ho3d_v3/temporal_episode_30_60_30_eval \
  --run-meta /data/wentao/ropetrack/runs/temporal_episode_dense_wilor_20260710/export/ho3d_v3_temporal_episode_eval_wilor_original/run_meta.json \
  --hard-manifest /data/wentao/ropetrack/hard/ho3d_v3/temporal_episode_30_60_30_eval/hard_manifest.jsonl \
  --fps 30 --raw-frame-step 1 \
  --output /data/wentao/ropetrack/runs/dense_history_v1_20260711/scores/base_episode.json
```

- [ ] Assert `masked_pa_mpjpe_mm` exceeds both context and recovery by at
  least `0.15 mm`; a failure stops all dependent GPU jobs.

### Task 2: Dense flex15 teacher

**Files:**
- Create: `.local_checks/dense_history_teacher.sbatch`

- [ ] Optimize the existing dense train export with the frozen winner recipe:

```bash
python scripts/rope_refiner/apply_rope_refinement.py \
  --mode optimize --objective rope --action-space flex15 \
  --gate-residual-threshold 0.1 \
  --opt-steps 400 --opt-lr 32 --opt-alpha-l2 0.001 \
  --dataset ho3d \
  --rope-labels /data/wentao/ropetrack/rope_labels/ho3d_v3/temporal_episode_30_60_30_train.jsonl \
  --pred-dir /data/wentao/ropetrack/runs/temporal_episode_dense_wilor_20260710/export/ho3d_v3_temporal_episode_train_wilor_original/eval_input \
  --run-meta /data/wentao/ropetrack/runs/temporal_episode_dense_wilor_20260710/export/ho3d_v3_temporal_episode_train_wilor_original/run_meta.json \
  --mano-cache /data/wentao/ropetrack/runs/temporal_episode_dense_wilor_20260710/export/ho3d_v3_temporal_episode_train_wilor_original/mano_cache.npz \
  --out-dir /data/wentao/ropetrack/runs/dense_history_v1_20260711/teacher/rope_flex15_gate010 \
  --device cuda --batch-size 512
```

- [ ] Verify 83,325 aligned `sample_id`, alpha, and cache rows.

### Task 3: Matched framewise and dense K1

**Files:**
- Create: `.local_checks/dense_history_base_train.sbatch`

- [ ] Train the matched framewise checkpoint with sequence split seed
  `20260710`, validation fraction `0.2`, hidden size 256, lr `1e-3`, and the
  existing noise/dropout recipe.
- [ ] Train dense K1 from that checkpoint:

```bash
python scripts/rope_refiner/train_temporal_student.py \
  --teacher-dir /data/wentao/ropetrack/runs/dense_history_v1_20260711/teacher/rope_flex15_gate010 \
  --framewise-checkpoint /data/wentao/ropetrack/runs/dense_history_v1_20260711/framewise/seed0/student.pt \
  --action-space flex15 \
  --out-dir /data/wentao/ropetrack/runs/dense_history_v1_20260711/k1/seed0 \
  --history-length 1 --raw-frame-step 1 --history-step 1 \
  --hidden-dim 128 --lr 0.0015 --batch-size 512 \
  --max-epochs 600 --patience 150 --val-frac 0.2 \
  --split-seed 20260710 --seed 0 --sensor-mode normal \
  --aug-noise-std 0 --aug-dropout 0.1 \
  --aug-bias-std 0 --aug-scale-range 0 --device cuda
```

- [ ] Reject non-finite metrics or a K1 checkpoint that does not beat its
  matched framewise validation L1.

### Task 4: Strict-past memory screen

**Files:**
- Create: `.local_checks/dense_history_k96_smoke.sbatch`
- Create: `.local_checks/dense_history_v2_grid.sbatch`
- Create: `.local_checks/dense_history_v2_control.sbatch`

- [ ] Run K96 for one epoch with batch size 256 to prove GPU memory safety.
- [ ] Submit K16/K96 x lr `3e-4/1e-3`, hidden size 64, seed 0:

```bash
python scripts/rope_refiner/train_temporal_student.py \
  --teacher-dir /data/wentao/ropetrack/runs/dense_history_v1_20260711/teacher/rope_flex15_gate010 \
  --base-temporal-checkpoint /data/wentao/ropetrack/runs/dense_history_v1_20260711/k1/seed0/temporal_student.pt \
  --action-space flex15 --out-dir "$OUT" \
  --history-length "$HISTORY" --raw-frame-step 1 \
  --inference-raw-frame-step 1 --history-step 1 --feature-delta-step 1 \
  --hidden-dim 64 --lr "$LR" --batch-size 256 \
  --max-epochs 400 --patience 80 --val-frac 0.2 \
  --split-seed 20260710 --seed 0 --sensor-mode normal \
  --aug-noise-std 0 --aug-dropout 0.1 \
  --aug-bias-std 0 --aug-scale-range 0 --device cuda
```

- [ ] Select the best K96 validation lr and train the same configuration with
  `--shuffle-history`.
- [ ] Do not expand seeds if K96 improves K1 by less than `2e-4` or shuffled
  history retains at least half of the gain.

### Task 5: Dense apply and phase scoring

**Files:**
- Create: `.local_checks/dense_history_apply.sbatch`
- Create: `.local_checks/dense_history_score.sbatch`

- [ ] Apply matched framewise, dense K1, best K16, best K96, K96 with
  `--temporal-disable-history`, and shuffled K96 to the dense evaluation cache.
- [ ] Run `score_temporal_predictions.py` with same-cadence K1 as reference,
  the episode manifest, fps 30, and raw step 1.
- [ ] Write `/data/wentao/ropetrack/runs/dense_history_v1_20260711/scores/history_episode.json`.
- [ ] Promote only if masked PA improves by `0.15 mm` with CI excluding zero,
  tips do not regress, acceleration/jitter improve 10%, and shuffled history
  loses at least half the real gain.

### Task 6: Verification and record

**Files:**
- Create after terminal results: `experience/0055_dense_history_utility.md`
- Modify after terminal results: `experience/INDEX.md`

- [ ] Verify every Slurm job exit code and exact pinned source SHA.
- [ ] Run `python -m unittest discover -s tests` and `git diff --check`.
- [ ] Record positive and negative results without committing data,
  checkpoints, predictions, or metrics payloads.
- [ ] Commit and push the closing record.
