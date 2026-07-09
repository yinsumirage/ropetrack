# HO3D Temporal Rope Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** Build and evaluate a causal HO3D v3 temporal rope refiner that beats matched framewise correction without sequence leakage, while preserving the released same-decoder and hard-gate protocol.

**Architecture:** A frozen matched framewise alpha student provides `alpha_frame`. A small GRU consumes causal windows of the existing 65D pose/rope feature plus five rope-difference channels and predicts a zero-initialized bounded logit residual. One implementation supports the safe `flex15` rope-teacher path and the high-upside train-only `oracle_chain/pose45` path.

**Tech Stack:** Python 3.11, NumPy, PyTorch, stdlib `unittest`, existing RopeTrack MANO/eval code, Slurm on the `engram` account.

**Design:** `docs/2026-07-10-temporal-refiner-design.md`

---

## File map

- Create `ropetrack/refine/temporal.py`: sequence/gap parsing, deterministic split, causal windows, rope differences, EMA, temporal model, checkpoint I/O, inference.
- Modify `ropetrack/refine/alpha_student.py`: expose reconstruction from an in-memory framewise checkpoint payload without changing legacy checkpoint behavior.
- Modify `scripts/rope_refiner/train_alpha_student.py`: optional sequence-disjoint split for matched framewise baselines.
- Create `scripts/rope_refiner/train_temporal_student.py`: frozen-framewise temporal training and controls.
- Modify `scripts/rope_refiner/apply_rope_refinement.py`: temporal checkpoint dispatch plus rope/alpha EMA baselines.
- Create `scripts/score_temporal_predictions.py`: gap-safe velocity, acceleration, jitter, lag, episode recovery, and sequence bootstrap.
- Modify `scripts/make_hard_images.py`: deterministic HO3D temporal episode roots.
- Modify `tests/test_alpha_student.py`, `tests/test_apply_rope_refinement.py`, `tests/test_make_hard_images.py`.
- Create `tests/test_temporal_refiner.py`, `tests/test_score_temporal_predictions.py`.
- Create ignored `.local_checks/submit_temporal_assets_v0.sh` and later grid launchers; never commit them.
- Create `experience/0053_temporal_refiner_jobs.md` and a result note after runs close; update `experience/INDEX.md`.

---

### Task 1: Launch reusable HPC assets and oracle teachers

**Files:**
- Create (ignored): `.local_checks/submit_temporal_assets_v0.sh`
- Read: `scripts/make_hard_images.py`
- Read: `scripts/make_rope_labels.py`
- Read: `scripts/eval.py`
- Read: `scripts/rope_refiner/apply_rope_refinement.py`

- [ ] **Step 1: Verify immutable train assets before submission**

Run a lightweight login-node check that exits nonzero unless all four sources
have 20,832 aligned rows:

```bash
source /public/home/guowt2512/miniforge3/etc/profile.d/conda.sh
conda activate ropetrack
python - <<'PY'
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
PY
```

Expected: `temporal-assets-ok 20832`.

- [ ] **Step 2: Write two independent GPU teacher scripts**

Both scripts use the existing stride-4 export. The deployment teacher adds
`--gate-residual-threshold 0.1`; the capacity teacher omits it. Each header is:

```bash
#!/usr/bin/env bash
#SBATCH -A engram
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o /data/wentao/ropetrack/runs/temporal_assets_20260710/logs/%x-%j.out
#SBATCH -e /data/wentao/ropetrack/runs/temporal_assets_20260710/logs/%x-%j.err
set -euo pipefail
source /public/home/guowt2512/miniforge3/etc/profile.d/conda.sh
conda activate ropetrack
cd ~/project/ropetrack
python scripts/rope_refiner/apply_rope_refinement.py \
  --mode optimize --objective oracle_chain --action-space pose45 \
  --opt-steps 400 --opt-lr 32 --opt-alpha-l2 0.001 \
  --dataset ho3d \
  --gt-xyz /data/wentao/ropetrack/hard/ho3d_v3/mask70_train/training_xyz.json \
  --rope-labels /data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl \
  --pred-dir /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor/eval_input \
  --run-meta /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor/run_meta.json \
  --mano-cache /data/wentao/ropetrack/runs/rope_p2_ho3d_v3_train_teacher_20260707_145307/export/ho3d_v3_mask70_train_wilor/mano_cache.npz \
  --out-dir /data/wentao/ropetrack/runs/temporal_assets_20260710/teacher/oracle_chain_pose45_gate010 \
  --device cuda --batch-size 512 --gate-residual-threshold 0.1
```

The ungated script is an exact separate file whose final two lines are:

```bash
  --out-dir /data/wentao/ropetrack/runs/temporal_assets_20260710/teacher/oracle_chain_pose45_ungated \
  --device cuda --batch-size 512
```

- [ ] **Step 3: Write the missing HO3D v3 eval dependency chain**

The CPU job creates `/data/wentao/ropetrack/hard/ho3d_v3/mask70` and aligned
rope labels. The dependent GPU job exports WiLoR MANO cache and applies the
released framewise student. The final CPU job scores both official and sliced
metrics. Use `afterok`, never poll inside a GPU allocation.

CPU generation body:

```bash
python scripts/make_hard_images.py --dataset ho3d --split evaluation \
  --input-root /data/wentao/ropetrack/HO3D_v3 \
  --output-root /data/wentao/ropetrack/hard/ho3d_v3/mask70 \
  --effect mask --severity 0.70 --limit 0
python scripts/make_rope_labels.py --dataset ho3d --split evaluation \
  --input-root /data/wentao/ropetrack/HO3D_v3 \
  --output /data/wentao/ropetrack/rope_labels/ho3d_v3/evaluation_rope.jsonl \
  --limit 0
```

GPU export/apply body:

```bash
python scripts/eval.py --dataset ho3d_v3_mask70 --method wilor_original \
  --split evaluation --save-mano-cache --batch-size 64 --num-workers 0 \
  --out-dir /data/wentao/ropetrack/runs/temporal_assets_20260710/eval_export
python scripts/rope_refiner/apply_rope_refinement.py --mode student \
  --checkpoint /data/wentao/ropetrack/releases/p2_four_teacher_student/student.pt \
  --dataset ho3d \
  --rope-labels /data/wentao/ropetrack/rope_labels/ho3d_v3/evaluation_rope.jsonl \
  --pred-dir /data/wentao/ropetrack/runs/temporal_assets_20260710/eval_export/eval_input \
  --run-meta /data/wentao/ropetrack/runs/temporal_assets_20260710/eval_export/run_meta.json \
  --mano-cache /data/wentao/ropetrack/runs/temporal_assets_20260710/eval_export/mano_cache.npz \
  --out-dir /data/wentao/ropetrack/runs/temporal_assets_20260710/eval/release \
  --device cuda --batch-size 512
```

- [ ] **Step 4: Submit and record job ids**

Run `sbatch` for the two oracle teachers and the CPU→GPU→CPU eval chain. Record
the exact ids and paths in `experience/0053_temporal_refiner_jobs.md` only
after submission output is captured.

---

### Task 2: Temporal sequence and window primitives

**Files:**
- Create: `ropetrack/refine/temporal.py`
- Create: `tests/test_temporal_refiner.py`

- [ ] **Step 1: Write failing parsing, split, gap, and causality tests**

```python
def test_causal_windows_reset_on_sequence_and_gap():
    ids = np.array(['A/0000', 'A/0004', 'A/0012', 'B/0000'])
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    windows, valid, lengths = build_causal_windows(
        ids, x, history_length=3, raw_frame_step=4, history_step=4)
    np.testing.assert_array_equal(valid[1], [True, True, False])
    np.testing.assert_array_equal(valid[2], [True, False, False])
    np.testing.assert_array_equal(valid[3], [True, False, False])
    np.testing.assert_array_equal(windows[1, :2], x[:2])

def test_future_changes_do_not_change_current_window():
    ids = np.array(['A/0000', 'A/0004', 'A/0008'])
    x = np.arange(6, dtype=np.float32).reshape(3, 2)
    before = build_causal_windows(ids, x, 3, 4, 4)[0][1].copy()
    x[2] = 999
    after = build_causal_windows(ids, x, 3, 4, 4)[0][1]
    np.testing.assert_array_equal(before, after)

def test_dense_rows_can_use_sparse_history_without_becoming_gaps():
    ids = np.asarray([f'A/{i:04d}' for i in range(6)])
    x = np.arange(6, dtype=np.float32)[:, None]
    windows, valid, _ = build_causal_windows(ids, x, 3, raw_frame_step=1, history_step=2)
    np.testing.assert_array_equal(windows[4, :3, 0], [0, 2, 4])
    np.testing.assert_array_equal(valid[4], [True, True, True])

def test_sequence_split_is_disjoint_and_stable():
    ids = np.array([f'{seq}/{frame:04d}' for seq in ('A','B','C','D','E') for frame in range(2)])
    a = deterministic_sequence_split(ids, val_fraction=0.2, seed=20260710)
    b = deterministic_sequence_split(ids, val_fraction=0.2, seed=20260710)
    assert a.train_sequences == b.train_sequences
    assert set(a.train_sequences).isdisjoint(a.val_sequences)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_temporal_refiner -v`

Expected: import failure because `ropetrack.refine.temporal` does not exist.

- [ ] **Step 3: Implement parsing and deterministic split**

```python
@dataclass(frozen=True)
class SequenceSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_sequences: tuple[str, ...]
    val_sequences: tuple[str, ...]

def sequence_frame(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).replace('\\', '/').split('/')
    if len(parts) < 2 or not parts[-1].isdigit():
        raise ValueError(f'invalid temporal sample_id: {sample_id}')
    return parts[-2], int(parts[-1])

def deterministic_sequence_split(sample_ids, val_fraction: float, seed: int) -> SequenceSplit:
    seq = np.asarray([sequence_frame(x)[0] for x in sample_ids])
    names = sorted(set(seq), key=lambda x: (hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest(), x))
    n_val = max(1, math.ceil(len(names) * val_fraction))
    val = tuple(names[:n_val])
    train = tuple(names[n_val:])
    return SequenceSplit(np.flatnonzero(np.isin(seq, train)), np.flatnonzero(np.isin(seq, val)), train, val)
```

- [ ] **Step 4: Implement left-aligned causal windows and rope differences**

```python
def build_causal_windows(sample_ids, features, history_length: int,
                         raw_frame_step: int, history_step: int):
    # group by sequence and split only when adjacent raw delta != raw_frame_step
    # within each segment, look up exact frames t-n*history_step
    # left-align chronological matches, pad with zeros, restore original rows
    return windows, valid, valid.sum(axis=1).astype(np.int64)

def temporal_features(cache):
    base = features_from_cache(cache)
    delta = np.zeros((len(base), 5), np.float32)
    # within each exact contiguous segment only, delta[t] = input[t] - input[t-1]
    return np.concatenate([base, delta], axis=1)
```

Reject duplicate `(sequence, frame)`, nonpositive steps, history steps not
divisible by raw steps, and feature/sample length mismatch. `temporal_features`
uses raw-frame adjacency for first differences; it never differences across a
gap.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m unittest tests.test_temporal_refiner -v`

Expected: all Task 2 tests pass.

Commit: `git commit -m "Add causal temporal window protocol"`

---

### Task 3: Sequence-disjoint matched framewise trainer

**Files:**
- Modify: `scripts/rope_refiner/train_alpha_student.py`
- Modify: `tests/test_alpha_student.py`

- [ ] **Step 1: Write a failing sequence-split trainer test**

```python
def test_framewise_sequence_split_is_recorded_and_disjoint(self):
    cache = toy_cache(20)
    cache['sample_id'] = np.asarray([f'S{i // 4}/{i % 4:04d}' for i in range(20)])
    with tempfile.TemporaryDirectory() as tmp:
        summary = train_student(cache, np.zeros((20, 15), np.float32), 'flex15', Path(tmp),
            gate_threshold=0.1, max_epochs=1, patience=1, split_by='sequence', split_seed=20260710)
        cfg = summary['config']
        self.assertTrue(set(cfg['train_sequences']).isdisjoint(cfg['val_sequences']))
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_alpha_student.AlphaStudentTrainingTest.test_framewise_sequence_split_is_recorded_and_disjoint -v`

Expected: `train_student() got an unexpected keyword argument 'split_by'`.

- [ ] **Step 3: Add optional split arguments without changing the default**

Add `split_by: str = 'frame'` and `split_seed: int | None = None`. Keep the
legacy permutation path byte-for-byte for `frame`; for `sequence`, call
`deterministic_sequence_split(cache['sample_id'], val_frac, split_seed or seed)`.
Store `split_by`, `split_seed`, `train_sequences`, and `val_sequences` in the
checkpoint config. Add CLI choices `--split-by frame|sequence` and
`--split-seed`.

- [ ] **Step 4: Verify legacy and sequence paths**

Run:

```bash
python -m unittest tests.test_alpha_student -v
python -m unittest tests.test_apply_rope_refinement -v
```

Expected: both modules pass; legacy checkpoint tests remain unchanged.

- [ ] **Step 5: Commit**

Commit: `git commit -m "Add sequence-disjoint alpha training split"`

---

### Task 4: Bounded temporal model and checkpoint

**Files:**
- Modify: `ropetrack/refine/alpha_student.py`
- Modify: `ropetrack/refine/temporal.py`
- Modify: `tests/test_temporal_refiner.py`

- [ ] **Step 1: Write failing zero-residual and bounds tests**

```python
def test_temporal_zero_head_equals_framewise_alpha():
    model = TemporalRopeAlphaStudent(in_dim=70, out_dim=15, hidden_dim=16, max_alpha=0.5)
    windows = torch.randn(3, 4, 70)
    lengths = torch.tensor([4, 2, 1])
    base = torch.tensor([[0.1] * 15, [-0.2] * 15, [0.0] * 15])
    torch.testing.assert_close(model(windows, lengths, base), base)

def test_temporal_output_is_bounded():
    model = TemporalRopeAlphaStudent(70, 15, 8, 0.5)
    with torch.no_grad():
        model.head.bias.fill_(100)
    out = model(torch.zeros(2, 1, 70), torch.ones(2, dtype=torch.long), torch.zeros(2, 15))
    assert float(out.abs().max()) <= 0.5
```

- [ ] **Step 2: Run tests and verify RED**

Expected: `TemporalRopeAlphaStudent` is not defined.

- [ ] **Step 3: Implement the packed GRU and logit residual**

```python
class TemporalRopeAlphaStudent(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, max_alpha=0.5):
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, windows, lengths, base_alpha):
        encoded = self.encoder(windows)
        packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        delta = self.head(hidden[-1])
        unit = (base_alpha / self.max_alpha).clamp(-1 + 1e-6, 1 - 1e-6)
        return self.max_alpha * torch.tanh(torch.atanh(unit) + delta)
```

- [ ] **Step 4: Add in-memory framewise reconstruction and temporal checkpoint I/O**

`alpha_student.py` gets `student_from_payload(payload, device)`; `load_student`
calls it so old checkpoints remain compatible. Temporal checkpoints store:

```python
{
  'model_state': temporal.state_dict(),
  'config': temporal_config,
  'framewise': {'model_state': framewise_payload['model_state'], 'config': framewise_payload['config']},
}
```

Load with `torch.load(..., weights_only=True)` only.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m unittest tests.test_temporal_refiner tests.test_alpha_student -v`

Commit: `git commit -m "Add bounded temporal alpha student"`

---

### Task 5: Temporal training and causal controls

**Files:**
- Create: `scripts/rope_refiner/train_temporal_student.py`
- Modify: `ropetrack/refine/temporal.py`
- Modify: `tests/test_temporal_refiner.py`

- [ ] **Step 1: Write failing tests for frozen base, zero sensor, and history shuffle**

```python
def test_zero_sensor_clears_all_sensor_channels():
    out = prepare_temporal_cache(toy_cache(6), sensor_mode='zero', seed=0)
    assert not out['rope_valid'].any()
    np.testing.assert_array_equal(out['input_rope_norm'], 0)

def test_shuffle_history_never_changes_current_slot():
    windows = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    valid = np.asarray([[True, True, True], [True, True, False]])
    out = shuffle_history(windows, valid, seed=3)
    for row, length in enumerate(valid.sum(1)):
        np.testing.assert_array_equal(out[row, length - 1], windows[row, length - 1])

def test_temporal_stats_ignore_validation_extremes():
    x = np.zeros((6, 70), np.float32)
    x[:4, 0] = [0, 1, 2, 3]
    x[4:, 0] = 9999
    mean, std = temporal_feature_stats(x, np.asarray([0,1,2,3]))
    self.assertAlmostEqual(float(mean[0]), 1.5)
    self.assertLess(float(std[0]), 2.0)
```

- [ ] **Step 2: Run and verify RED**

Expected: helper functions are missing.

- [ ] **Step 3: Implement segment-consistent augmentation**

For each epoch, draw Gaussian noise/dropout per frame. If bias or scale is
enabled, draw one `[5]` bias and one scalar scale per contiguous segment, apply
them to every frame in that segment, clip to `[0,1]`, then rebuild residual and
difference features. In `zero` mode return before every noise/dropout branch.

Implement and use the exact statistics contract:

```python
def temporal_feature_stats(features70, train_idx):
    train = np.asarray(features70, np.float32)[np.asarray(train_idx)]
    return train.mean(0).astype(np.float32), np.maximum(train.std(0), 1e-4).astype(np.float32)
```

Fit it once from clean 70D rows belonging to training sequences only. Store
`temporal_feature_mean` and `temporal_feature_std` in the checkpoint. Every
augmented train window, clean validation window, and inference window uses
`(x - mean) / std` from those stored arrays; no other statistics are fitted.

- [ ] **Step 4: Implement the training entrypoint**

Required CLI:

```text
--teacher-dir --framewise-checkpoint --action-space --out-dir
--history-length --raw-frame-step --history-step --hidden-dim --lr --batch-size
--max-epochs --patience --val-frac --split-seed --seed
--sensor-mode normal|zero --shuffle-rope --shuffle-history
--aug-noise-std --aug-dropout --aug-bias-std --aug-scale-range
--device
```

At startup assert single teacher dir, checkpoint action/out dim match, and
framewise train/val sequence lists equal the deterministic split. Freeze all
framewise parameters (`requires_grad=False`). Each epoch rebuilds augmented
windows, computes frozen framewise alpha under `torch.no_grad()`, trains only
the temporal model with the specified L1+L2 loss, and early-stops on clean
validation teacher-alpha L1. Assert the checkpoint mean/std shapes are `(70,)`
before saving and loading.

- [ ] **Step 5: Add a two-sequence CPU train smoke**

Run one epoch with `hidden_dim=8`, `history_length=2`, and assert:

```python
assert summary['config']['model_type'] == 'causal_gru'
assert summary['config']['framewise_frozen'] is True
assert set(summary['config']['train_sequences']).isdisjoint(summary['config']['val_sequences'])
assert (out_dir/'temporal_student.pt').exists()
```

- [ ] **Step 6: Run tests and commit**

Run: `python -m unittest tests.test_temporal_refiner -v`

Commit: `git commit -m "Train causal temporal alpha students"`

---

### Task 6: Apply integration and causal EMA baselines

**Files:**
- Modify: `scripts/rope_refiner/apply_rope_refinement.py`
- Modify: `ropetrack/refine/temporal.py`
- Modify: `tests/test_apply_rope_refinement.py`
- Modify: `tests/test_temporal_refiner.py`

- [ ] **Step 1: Write failing EMA boundary and temporal dispatch tests**

```python
def test_causal_ema_resets_at_gap_and_sequence():
    ids = np.array(['A/0000','A/0001','A/0003','B/0000'])
    x = np.array([[0.],[1.],[10.],[20.]], np.float32)
    out = causal_ema(ids, x, decay=0.5, raw_frame_step=1)
    np.testing.assert_allclose(out[:, 0], [0., 0.5, 10., 20.])

def test_temporal_mode_keeps_closed_current_gate_zero(self):
    cache = {
        'base_rope_norm': np.full((2, 5), 0.5, np.float32),
        'input_rope_norm': np.full((2, 5), 0.55, np.float32),
        'rope_valid': np.ones((2, 5), bool),
    }
    temporal_alpha = np.ones((2, 15), np.float32)
    gate = gate_from_cache(cache, 0.1)
    gated = temporal_alpha * expand_gate_to_alpha(gate, 'flex15')
    np.testing.assert_array_equal(gated, 0.0)

def test_cli_accepts_temporal_mode(self):
    args = parse_args([
        '--mode', 'temporal', '--checkpoint', 'temporal.pt',
        '--rope-labels', 'rope.jsonl', '--pred-dir', 'pred',
        '--run-meta', 'run_meta.json', '--mano-cache', 'mano.npz',
        '--out-dir', 'out',
    ])
    self.assertEqual(args.mode, 'temporal')
```

- [ ] **Step 2: Run and verify RED**

Expected: `causal_ema`/temporal mode unavailable.

- [ ] **Step 3: Add inference and filter arguments**

Extend mode choices with `temporal`. Add mutually exclusive
`--rope-ema-decay` and `--alpha-ema-decay` in `[0,1)`, plus
`--ema-raw-frame-step`. Rope EMA runs before feature building; alpha EMA runs after
student/temporal inference and before the current-frame gate. Temporal
checkpoint config is authoritative for action space, gate, history,
`raw_frame_step`, `history_step`, and stored 70D normalization. Measure wall
time with `time.perf_counter()` around alpha inference plus action/decode and
write `timing.wall_seconds` and `timing.per_sample_ms` to `summary.json`.

- [ ] **Step 4: Preserve legacy numerical path**

When mode is `student` and no EMA flags are passed, execute the same statements
as before. Re-run the local FakeMano tests and later the HPC release golden
regression before calling the implementation complete.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m unittest tests.test_apply_rope_refinement tests.test_temporal_refiner -v
```

Commit: `git commit -m "Apply temporal students and causal EMA"`

---

### Task 7: Gap-safe temporal scorer

**Files:**
- Create: `scripts/score_temporal_predictions.py`
- Create: `tests/test_score_temporal_predictions.py`

- [ ] **Step 1: Write failing metric tests**

Use a toy sequence with constant velocity GT, identical prediction, a one-frame
lagged prediction, a gap, and a second sequence:

```python
def test_motion_metrics_reset_at_gap_and_detect_lag():
    ids = np.asarray(['A/0000','A/0001','A/0002','A/0004','A/0005','B/0000','B/0001'])
    gt = np.zeros((7, 21, 3), np.float32)
    gt[:, :, 0] = np.asarray([0, 1, 2, 10, 11, 20, 21])[:, None] * 0.001
    same = temporal_motion_metrics(ids, gt, gt, fps=30.0, raw_frame_step=1)
    self.assertAlmostEqual(same['velocity_error_mm_s'], 0.0)
    self.assertAlmostEqual(same['acceleration_error_mm_s2'], 0.0)
    self.assertEqual(same['num_velocity_edges'], 4)  # 2 in A0, 1 in A1, 1 in B

def test_phase_lag_positive_means_prediction_trails():
    gt = np.sin(np.linspace(0, 4 * np.pi, 80))
    pred = np.concatenate([np.repeat(gt[0], 2), gt[:-2]])
    self.assertEqual(phase_lag(gt, pred, max_lag=15), 2)

def test_recovery_requires_three_consecutive_frames():
    error = np.array([1.0] * 30 + [5.0] * 60 + [2.0, 1.4, 1.4, 1.4] + [1.0] * 26)
    assert recovery_frames(error, context=30, masked=60, recovery=30, margin_mm=0.5, stable=3) == 1
```

- [ ] **Step 2: Run and verify RED**

Expected: scorer module missing.

- [ ] **Step 3: Implement metrics**

CLI accepts repeated `--method NAME=DIR`, `--reference NAME`, `--gt-dir`,
`--run-meta`, optional `--hard-manifest`, `--fps`, `--raw-frame-step`, and
`--output`. Each method dir must contain `pred.json`, `rope_residuals.npz`, and
`summary.json`; `base` may point at the common eval export. Load GT xyz and
verts plus sample order. Reuse `per_joint_pa_distances` for PA-MPJPE and
`align_w_scale`/`point_distances` for per-frame PA-MPVPE. Root-relativize joints
at joint 0 for motion metrics; multiply first/second differences by fps/fps
squared and convert meters to millimeters. Split every sequence at raw-frame
gaps before differences.

If a hard manifest is present, aggregate context/masked/recovery PA separately
and compute recovery. Load rope closure from each method's residual NPZ and
latency from `summary.json`. Fail if any method's sample ids or row count differ
from the reference.

Lag searches integer offsets `[-15,15]` on each maximal segment with at least
31 frames. Zero-variance correlations return `None` and increment
`num_lag_segments_excluded`.

- [ ] **Step 4: Implement sequence bootstrap**

Resample sequence keys 2,000 times with `np.random.default_rng(20260710)` and
report percentile `[2.5,97.5]` for every method-minus-reference PA-MPJPE,
PA-MPVPE, masked/recovery PA, velocity error, and acceleration error. Never
sample frames independently. The JSON top level contains `methods`,
`reference`, `paired_deltas`, `bootstrap_ci`, `rope_closure`, and `timing` so
the pre-registered gate can be evaluated without joining another table.

- [ ] **Step 5: Run tests and commit**

Run: `python -m unittest tests.test_score_temporal_predictions -v`

Commit: `git commit -m "Score causal temporal predictions"`

---

### Task 8: Deterministic temporal episode roots

**Files:**
- Modify: `scripts/make_hard_images.py`
- Modify: `ropetrack/refine/temporal.py`
- Modify: `tests/test_make_hard_images.py`
- Modify: `tests/test_temporal_refiner.py`

- [ ] **Step 1: Write failing schedule and image tests**

```python
def test_episode_schedule_resets_at_gap():
    ids = [f'A/{i:04d}' for i in range(120)] + ['A/0200', 'A/0201']
    phase = episode_schedule(ids, context=30, masked=60, recovery=30, raw_frame_step=1)
    assert phase[0].phase == 'context'
    assert phase[30].phase == 'masked'
    assert phase[90].phase == 'recovery'
    assert phase[120].phase == 'tail'

def test_episode_builder_copies_clean_and_masks_only_masked_phase():
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / 'src', Path(tmp) / 'out'
        rgb = src / 'evaluation' / 'AP10' / 'rgb'
        meta_dir = src / 'evaluation' / 'AP10' / 'meta'
        rgb.mkdir(parents=True); meta_dir.mkdir(parents=True)
        K = np.eye(3, dtype=np.float32)
        joints = np.zeros((21, 3), np.float32); joints[:, 2] = -1.0
        for frame in ('0000', '0001', '0002'):
            Image.new('RGB', (32, 32), 'white').save(rgb / f'{frame}.png')
            with (meta_dir / f'{frame}.pkl').open('wb') as f:
                pickle.dump({'handBoundingBox': [4,4,28,28], 'handJoints3D': joints, 'camMat': K}, f)
        (src/'evaluation.txt').write_text('AP10/0000\nAP10/0001\nAP10/0002\n')
        (src/'evaluation_xyz.json').write_text(json.dumps([joints.tolist()] * 3))
        (src/'evaluation_verts.json').write_text(json.dumps([[[0,0,0]]] * 3))
        hard.build_ho3d_hard_root(src, out, 'mask', 0.7, None, 7,
            episode_context=1, episode_mask=1, episode_recovery=1)
        rows = [json.loads(x) for x in (out/'hard_manifest.jsonl').read_text().splitlines()]
        self.assertEqual([x['episode_phase'] for x in rows], ['context','masked','recovery'])
        self.assertEqual(Image.open(out/'evaluation/AP10/rgb/0000.png').getpixel((16,16)), (255,255,255))
        self.assertEqual(Image.open(out/'evaluation/AP10/rgb/0001.png').getpixel((16,16)), (0,0,0))
        self.assertEqual(Image.open(out/'evaluation/AP10/rgb/0002.png').getpixel((16,16)), (255,255,255))
```

- [ ] **Step 2: Run and verify RED**

Expected: episode arguments/helpers missing.

- [ ] **Step 3: Add explicit CLI and manifest fields**

Add `--episode-context`, `--episode-mask`, `--episode-recovery`, all default
zero. Require all three together and HO3D only. For context/recovery use
`shutil.copy2`; masked frames call existing `save_hard_image`. Manifest rows
add `episode_id`, `episode_phase`, `episode_offset`, and `segment_id`. Tails are
copied clean and labelled `tail`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
python -m unittest tests.test_make_hard_images tests.test_temporal_refiner -v
```

Commit: `git commit -m "Generate temporal occlusion episodes"`

---

### Task 9: Full local verification, review, push, and HPC grid

**Files:**
- Modify: `scripts/README.md`
- Create: `experience/0053_temporal_refiner_jobs.md`
- Modify: `experience/INDEX.md`
- Create (ignored): `.local_checks/submit_temporal_grid_v0.sh`

- [ ] **Step 1: Run the complete local suite**

Run: `python -m unittest discover -s tests`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Request adversarial code review**

Review the full branch diff against this plan. Fix every Critical/Important
finding, then rerun the complete suite.

- [ ] **Step 3: Push and sync HPC without touching the dirty submodule**

Push `codex/temporal-ho3d-refiner`, fetch/fast-forward the same branch in
`~/project/ropetrack`, and verify the only allowed remote status entry is
` m third_party/wilor`.

- [ ] **Step 4: Run HPC light tests**

Submit a CPU test job (not login-node training) for the full suite. Gate all
new GPU training jobs on its success.

- [ ] **Step 5: Submit screening grid**

Using the existing 20,832-frame flex15 teacher and the two completed pose45
teachers, train matched framewise checkpoints with sequence split seed
20260710. Train the following screening cells, but do not apply them to the
official evaluation export:

```text
flex15: history {1,4,8,16} x hidden {128,256}, seed 0
flex15 controls: zero_sensor, shuffle_rope, shuffle_history at best K
flex15 filters: rope EMA {0.5,0.8}, alpha EMA {0.5,0.8}
pose45 gate010: history {1,4,8,16} x hidden {128,256}, seed 0
pose45 ungated: history {1,8}, hidden 128, seed 0 (capacity ablation only)
```

Bundle multiple short training cells per GPU job so each allocation performs
continuous GPU work. Select candidates only from the 11-sequence clean
validation teacher-alpha L1 stored in `train_log.json`; record the complete
ranking before any official apply job is submitted.

- [ ] **Step 6: Promote finalists**

Choose by validation teacher-alpha L1 only. Promote one flex15 and one gated
pose45 configuration to seeds `{0,1,2}` plus noise `0.05`, dropout `0.1`, and
bias/scale stress. Only after the ranking is frozen, submit official apply for
the promoted finalists, their matched framewise references, EMA baselines, and
the pre-registered zero/shuffle/history controls. A single dependent CPU scorer
compares all method dirs with `--reference matched_framewise`. No rejected grid
checkpoint is evaluated on the official 13 sequences.

- [ ] **Step 7: Run dense episode track if fast-track plumbing is healthy**

Generate stride-1 train/eval episode roots with `30/60/30`, export caches,
generate matching teachers, retrain finalists, and score episode PA, recovery,
jitter, and lag. This is a new Slurm dependency chain; do not reuse stride-4
metrics as 30 fps evidence.

- [ ] **Step 8: Record jobs and commit**

Write job ids, run roots, exact cells, and initial health checks to
`experience/0053_temporal_refiner_jobs.md`; add the index row and commit.

---

### Task 10: Result-driven iteration and closure

**Files:**
- Create: `experience/0054_temporal_refiner_results.md`
- Modify: `experience/INDEX.md`
- Modify: `docs/2026-07-08-progress-report.md` only if a pass gate is met

- [ ] **Step 1: Verify every Slurm terminal state and output count**

Require `COMPLETED 0:0`, exact expected prediction count, finite JSON metrics,
and no missing sequence rows. Cancel and diagnose any job violating GPU/data
rules.

- [ ] **Step 2: Apply the pre-registered gates**

Compare each finalist against its matched framewise model on official 13-seq
eval. Do not select by official eval. Classify each as point-accuracy win,
smoothing-only, neutral, or regression.

- [ ] **Step 3: Iterate only on diagnosed failure modes**

- Underfit with clean train/val curves: raise hidden 128→256 or K 4→8.
- Good jitter but excessive lag: lower EMA decay or reduce K.
- Pose45 clean regression: use gate010 teacher/model, lower max alpha, or stop
  pose45; do not hide it by averaging.
- Controls retain gain: reject rope-causal attribution and inspect leakage.

Each new cell needs one hypothesis and one changed variable.

- [ ] **Step 4: Write the durable result note and verify release regression**

Record positive and negative cells, sequence-bootstrap intervals, recovery,
lag, latency, and exact run paths. Re-run the RELEASE.md golden regression for
the unchanged framewise checkpoint before claiming the branch safe.

- [ ] **Step 5: Commit and push the result closure**

Commit only code/docs/experience and small report figures. Do not commit
checkpoints, predictions, metrics directories, caches, logs, or large images.

---

## Final verification checklist

- [ ] `python -m unittest discover -s tests` passes locally and in an HPC CPU job.
- [ ] Legacy release `student` mode reproduces RELEASE.md golden constants.
- [ ] Window tests prove causality, gap reset, original-row restoration, and split disjointness.
- [ ] Zero/shuffle/history controls are clean and independently scored.
- [ ] All official metrics use the same eval export and same MANO decoder.
- [ ] No official evaluation sequence influenced hyperparameter selection.
- [ ] GPU jobs perform real GPU work and all output/log paths live under `/data/wentao/ropetrack/runs`.
- [ ] `experience/INDEX.md` points to job and result notes.
