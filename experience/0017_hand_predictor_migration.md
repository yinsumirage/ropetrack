# 0017 Hand Predictor Migration

Date: 2026-07-03

## Purpose

Start conservative migration away from using `third_party/anyhand` as the runtime
wrapper source.

## Change

- Added outer-repo `ropetrack.backends.hand_predictor.HandPredictor`.
- Default assets now resolve from repo root:
  - `mano_data/MANO_RIGHT.pkl`
  - `pretrained_models/anyhand_wilor.ckpt`
  - `pretrained_models/model_config_wilor.yaml`
  - `pretrained_models/detector.pt`
  - `pretrained_models/hamer_ckpts/checkpoints/anyhand_hamer.ckpt`
- `scripts/bench_freihand.py` and `scripts/bench_ho3d.py` now import
  `HandPredictor` instead of `third_party/anyhand/scripts/rgb_predictor.py`.
- `third_party/anyhand` is still present for parity review and rollback.

## Asset Rule

Do not commit MANO assets or checkpoints. On HPC, place or link them under the
repo root:

```text
~/project/ropetrack/mano_data
~/project/ropetrack/pretrained_models -> /data/wentao/ropetrack/pretrained_models
```

## Verification

Local checks only:

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests
```

Result: 39 tests passed.

HPC GPU smoke:

```text
job: 162776
script: .local_checks/hpc_hand_predictor_wilor_smoke.sbatch
logs:
  logs/hand_predictor_wilor_smoke-162776.out
  logs/hand_predictor_wilor_smoke-162776.err
state: COMPLETED 0:0
elapsed: 00:04:09
gpu: NVIDIA H200
```

Observed output:

```text
torch 2.5.1+cu118
cuda True
num_hands 2
first wilor (778, 3) (21, 3) [265.0, 156.0, 423.0, 287.0] 0.742
```

HPC HO3D v2 AnyHand-HaMeR parity:

```text
run: /data/wentao/ropetrack/runs/ho3d_v2_hamer_anyhand_handpredictor_gtbbox_20260703
prediction job: 162905, COMPLETED 0:0, elapsed 00:12:31
eval job: 162934
```

Result matched `experience/0009_ho3d_hamer_gtbbox_jobs.md`:

```text
xyz_procrustes_al_mean3d: 0.744107
xyz_procrustes_al_auc3d: 0.851269
xyz_scale_trans_al_mean3d: 1.503280
xyz_scale_trans_al_auc3d: 0.703502
mesh_al_mean3d: 0.768454
mesh_al_auc3d: 0.846367
f_al_score_5: 0.643820
f_al_score_15: 0.984041
```

## Next

Remove `third_party/anyhand` after preserving `mano_data` outside the submodule.
