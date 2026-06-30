# 10-Day Benchmark Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $superpower-subagents (recommended) or $superpower-executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking via update_plan.

**Goal:** By 2026-07-10, prepare a teacher-facing progress report with verified environment setup, aligned FreiHAND + HO3D v2 data protocol, clean baseline scores for HaMeR/WiLoR/AnyHand, hard benchmark evidence, and first rope-distance labels/results.

**Architecture:** Keep this repo as the outer benchmark repo. Third-party repos stay in `third_party/`; `ropetrack` owns data audit, manifests, prediction schema, wrappers, metrics, visual checks, hard splits, rope labels, and experiment notes.

**Tech Stack:** Python 3.10+, PyTorch per backend requirements, stdlib-first repo tools, NumPy/OpenCV/Pillow only when the first loader or overlay actually needs them.

---

## Ground Rules

- Full training is out of scope until clean + hard benchmark numbers are credible.
- Every dataset/backend must pass visual overlay and coordinate sanity checks before its score is trusted.
- Every experiment gets a dated note under `experience/` and a runnable folder under `experiments/`.
- MANO is a licensed manual download; dataset/model download can be blocked by accounts or mirrors. If blocked, record it immediately and switch to the smallest runnable demo path.

## Files To Create First

- `scripts/check_env.py`: print Python, CUDA, PyTorch, submodule, and MANO/model/data presence.
- `scripts/audit_data_links.py`: verify `data/raw/freihand` and `data/raw/ho3d_v2` exist and summarize expected files.
- `scripts/prepare_freihand.py`: export a small manifest first, then full manifest.
- `scripts/prepare_ho3d_v2.py`: export a small manifest first, then full manifest.
- `scripts/viz_overlay.py`: draw image + bbox + 2D joints + projected 3D joints for audit.
- `scripts/run_hamer.py`: call unmodified HaMeR and write the unified prediction schema.
- `scripts/run_wilor.py`: call unmodified WiLoR and write the unified prediction schema.
- `scripts/run_anyhand.py`: call AnyHand predictor/checkpoints and write the same schema.
- `scripts/eval_predictions.py`: compute MPJPE and fingertip error from schema files.
- `scripts/make_hard_split.py`: create fixed hard manifests for mask/blur/crop/appearance.
- `scripts/make_rope_labels.py`: compute 5 fingertip-to-wrist distances from GT joints/MANO.

## Day 1: 2026-07-01, Environment + Data Alignment + Tiny Clean Bench

- [ ] **Step 1: Check backend requirements**

Run:

```powershell
python --version
git submodule status --recursive
Get-Content third_party\hamer\README.md
Get-Content third_party\wilor\README.md
Get-Content third_party\anyhand\README.md
```

Expected: exact PyTorch/MANO/checkpoint requirements are recorded in `experience/2026-07-01_env.md`.

- [ ] **Step 2: Install one environment, not three**

Use AnyHand's constraint first because it wraps both HaMeR and WiLoR and pins `torch<2.6`. If that fails, split into backend-specific envs.

Expected: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` succeeds.

- [ ] **Step 3: Download manual assets**

Required manual assets:

```text
MANO_RIGHT.pkl
WiLoR detector.pt
WiLoR wilor_final.ckpt
HaMeR demo/eval checkpoints via fetch_demo_data.sh or equivalent
AnyHand HaMeR/WiLoR checkpoints via its setup scripts
```

Expected: `scripts/check_env.py` reports present/missing assets.

- [ ] **Step 4: Link datasets**

Create or verify:

```text
data/raw/freihand
data/raw/ho3d_v2
```

Expected: `scripts/audit_data_links.py` prints image/annotation/split availability.

- [ ] **Step 5: Export tiny manifests**

Export 20 FreiHAND validation samples and 20 HO3D v2 eval/validation samples.

Expected:

```text
data/manifests/freihand_val_tiny.jsonl
data/manifests/ho3d_v2_eval_tiny.jsonl
```

- [ ] **Step 6: Coordinate audit before scores**

For each dataset, check:

```text
unit = mm
bbox = original image xyxy
root = wrist/root definition recorded
joint_order = 21-point order recorded
hand_side = left/right policy recorded
camera = OpenCV camera coordinates recorded
```

Expected: 20 overlay images per dataset under `outputs/figures/audit_2026-07-01/`.

- [ ] **Step 7: Run tiny clean inference**

Run HaMeR, WiLoR, then AnyHand if checkpoints are ready.

Expected: prediction JSONL files under `outputs/predictions/*_tiny.jsonl`.

- [ ] **Step 8: Compute tiny metrics**

Run:

```powershell
$env:PYTHONPATH='src'
python scripts\eval_predictions.py --manifest data\manifests\freihand_val_tiny.jsonl --pred outputs\predictions\hamer_tiny.jsonl
```

Expected: a first metrics table plus a warning list of any unresolved coordinate assumptions.

## Day 2: 2026-07-02, Hard Benchmark v0 + Rope Labels

- [ ] Create deterministic hard split manifests: `mask`, `blur`, `crop`, `appearance`, `mixed`.
- [ ] Tune severity on tiny data until baseline degradation is visible but not nonsense.
- [ ] Generate rope labels from GT joints/MANO:

```text
rope_dist_mm[N,5]
rope_dist_norm[N,5]
rope_valid[N,5]
rope_source
```

- [ ] Run tiny clean vs hard for at least one working backend.
- [ ] Write `experience/2026-07-02_hard_rope_v0.md`.

Gate: if hard split does not reduce fingertip/occluded subset performance, severity is too weak; if overlays become invalid, severity is too strong.

## Day 3: 2026-07-03, Full Clean Bench Candidate

- [ ] Expand manifests beyond tiny once Day 1 alignment is accepted.
- [ ] Run HaMeR/WiLoR/AnyHand on a practical subset or full validation depending on runtime.
- [ ] Compare against official paper/repo expectations only after matching protocol.
- [ ] Produce clean table:

```text
Method | FreiHAND clean | HO3D v2 clean | Notes
```

Gate: if scores are far from expected, stop and debug data/ROI/joint/root before adding rope or training.

## Day 4: 2026-07-04, Hard Bench Candidate

- [ ] Run hard splits for the same backend set.
- [ ] Report overall and fingertip-specific drops.
- [ ] Save paired examples: original, hard image, prediction overlay.

Gate: hard benchmark must expose a meaningful weakness, especially fingertip or occluded-fingertip error.

## Day 5: 2026-07-05, Rope Post-Opt Minimal Proof

- [ ] Implement the smallest rope residual evaluator first.
- [ ] Test simulated noisy rope: 0 mm, 2 mm, 5 mm, 10 mm, dropout.
- [ ] Do post-opt only if backend outputs enough MANO/joints to optimize or re-score.

Gate: success is not lower rope residual; success is lower 3D fingertip/occluded fingertip error without breaking clean samples.

## Day 6: 2026-07-06, Failure Analysis

- [ ] Cluster failures by dataset, hand side, crop, visibility, hard type, and backend.
- [ ] Create 20-panel failure figure for teacher report.
- [ ] Decide whether the next code should be wrapper fixes, data fixes, or rope post-opt.

## Day 7: 2026-07-07, First Ablation Table

- [ ] Table rows: RGB/backend only, hard-only, rope oracle, rope 5 mm noise, rope dropout.
- [ ] Columns: MPJPE, fingertip error, occluded fingertip, rope residual.
- [ ] Keep all runs on the same manifest.

## Day 8: 2026-07-08, Report Figures

- [ ] Prepare method diagram: outer repo, backends, unified schema, evaluator, rope labels.
- [ ] Prepare data audit figure: bbox/keypoint/projection overlays.
- [ ] Prepare hard benchmark figure: mask/blur/crop/appearance examples.
- [ ] Prepare rope figure: five fingertip-to-wrist distances and what they constrain.

## Day 9: 2026-07-09, Teacher Report Draft

- [ ] Write concise progress report:

```text
1. What is built
2. What data is aligned
3. Which baselines ran
4. What scores/plots say
5. What failed
6. Next 2-week plan
```

- [ ] Include explicit risks: MANO license, dataset access, coordinate mismatch, runtime, hard split fairness.

## Day 10: 2026-07-10, Freeze Results

- [ ] Re-run the exact report commands.
- [ ] Save configs, command logs, metrics CSV, and figures.
- [ ] Make one final experience note with reproducibility commands.
- [ ] Do not add new features on report day.

## Verification

Daily minimum:

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
git status --short --branch
```

Bench verification:

```powershell
python scripts\check_env.py
python scripts\audit_data_links.py
python scripts\eval_predictions.py --help
```

## Next Skill

Use `$superpower-executing-plans` for inline execution. Subagents are useful later, but Day 1 depends on local GPU, dataset paths, and licensed assets, so inline execution is simpler.
