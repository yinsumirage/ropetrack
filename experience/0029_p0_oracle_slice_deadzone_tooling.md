# 0029 P0 Oracle/Slice/Deadzone Tooling And HO3D Chain Fix

Date: 2026-07-07

## Purpose

Implement the P0 diagnostic tooling from
`docs/2026-07-06-rope-refinement-next-plan.md` locally (no HPC runs yet):

- oracle ceiling objectives for the test-time optimizer;
- `mult15` / `flex15` action spaces next to the original `mult5`;
- rope residual closure reporting on every apply run;
- occluded-finger sliced scoring (H1);
- multiplicative dead-zone analysis (H2).

## Code

New package modules:

- `ropetrack/refine/actions.py`: `mult5` (reference-equal to the 0027
  implementation), `mult15`, `flex15` action spaces; per-finger alpha and
  pose-magnitude helpers.
- `ropetrack/refine/oracle.py`: differentiable torch twins of
  `eval_points_from_model` / `joints_from_vertices` (parity-tested against
  the numpy protocol), `oracle_tip` / `oracle_chain` joint sets,
  wrist-relative cm^2 loss.
- `ropetrack/refine/analysis.py`: residual closure summary, pearson/spearman,
  quantile buckets, `json_sanitize`.

Extended `scripts/rope_refiner/apply_rope_refinement.py`:

- `--objective rope|oracle_tip|oracle_chain` (`oracle_*` needs `--gt-xyz`,
  rows in `run_meta.json` `sample_order` order);
- `--action-space mult5|mult15|flex15`; flex directions are frozen per-sample
  rope-distance gradients per joint (saved to `flex_directions.npy`);
- every run writes `rope_residuals.npz` and a summary with alpha stats and
  residual closure, same-decoder on both sides;
- optimizer defaults changed to the published 0027 aggressive recipe
  (`steps=120 lr=2.0 alpha_l2=0.001 max_alpha=0.5`); the old conservative
  defaults were the do-nothing recipe.

New analysis scripts:

- `scripts/score_sliced_predictions.py`: PA per-joint errors sliced by
  occluded/clean fingers (from `hard_manifest.jsonl`), fingertip subsets,
  rope-residual buckets, residual-vs-improvement correlations. The
  `all_joints` slice reproduces `xyz_procrustes_al_mean3d`.
- `scripts/rope_refiner/analyze_alpha_deadzone.py`: base finger curl vs
  correction size vs residual closure, per finger and per curl bucket.

## HO3D Chain Defect (Affects Published 0028 Numbers)

Adversarial review of this change found that the WiLoR MANO wrapper always
outputs OpenPose/FreiHAND-ordered joints (`mano_wrapper.py` `joint_map`),
but the optimizer indexed `out.joints` with `FINGER_CHAINS[dataset]`. For
`--dataset ho3d` the thumb slot therefore measured wrist->ring-tip and the
index/middle/ring slots measured proximal pinky joints; only pinky was
correct. The rope loss for HO3D drove 4/5 fingers with nonsense residuals.

- This defect was inherited from the code that produced the HO3D v2 `mask70`
  optimization row in `experience/0028`; those HO3D numbers must be re-run
  and may improve. FreiHAND rows are unaffected
  (`FINGER_CHAINS["freihand"]` == wrapper order).
- Fix: wrapper joints are now always indexed with the OpenPose chains; the
  dataset-specific chains apply only to eval-decoded joints. Oracle
  objectives decode joints from vertices via the per-dataset protocol and
  were never affected.
- Regression test: `test_ho3d_dataset_uses_openpose_wrapper_chains`.

## Other Review Fixes

- `sliced_scores.json` / `summary.json` / `alpha_deadzone.json` sanitize
  non-finite floats to `null` (tip_*/finger_end manifests made every clean
  slice NaN, which serialized as invalid JSON).
- `global_orient/betas/cam_t` are now joined to the refiner cache by
  `sample_id` (`load_mano_globals`), not positionally; a reordered
  `--mano-cache` previously paired poses with the wrong extrinsics.
- Reported rope residuals are unclamped, matching the optimizer objective;
  labels stay clamped.
- Mask occlusion test matches PIL's end-inclusive rectangle painting
  (`[rx1, rx2 + 1)`).
- Dead-zone script cross-checks `sample_id` between cache and residuals.

## Verification

- `python -m pytest tests`: 148 passed locally (torch CPU,
  no MANO files needed — optimization loop is exercised through an
  injectable two-segment toy hand, `FakeMano`).
- Review: 3-dimension adversarial review, 14 findings, 10 confirmed after
  verification, all fixed; 4 refuted.

## Next (HPC)

P0 run matrix on FreiHAND `mask70` + `finger_end80`, WiLoR first:

```bash
# oracle ceiling / action-space grid (one GPU job, 6 cells)
for OBJ in rope oracle_tip; do
  for ACT in mult5 mult15 flex15; do
    python scripts/rope_refiner/apply_rope_refinement.py \
      --mode optimize --objective $OBJ --action-space $ACT \
      --dataset freihand \
      --rope-labels /data/wentao/ropetrack/rope_labels/freihand/evaluation_rope.jsonl \
      --pred-dir <export>/eval_input --run-meta <export>/run_meta.json \
      --mano-cache <export>/mano_cache.npz \
      --gt-xyz <hard_root>/evaluation_xyz.json \
      --out-dir <run_root>/${OBJ}_${ACT}
  done
done
# then per cell (CPU):
python scripts/score_predictions.py <out_dir> <out_dir>/scores --gt-dir <hard_root> --pred_file_name pred.json
python scripts/score_sliced_predictions.py <out_dir> <out_dir>/sliced --dataset freihand \
  --gt-dir <hard_root> --hard-manifest <hard_root>/hard_manifest.jsonl --cache <out_dir>/refiner_eval_cache.npz
python scripts/rope_refiner/analyze_alpha_deadzone.py --cache <out_dir>/refiner_eval_cache.npz \
  --alpha <out_dir>/alpha.npy --residuals <out_dir>/rope_residuals.npz --action-space <act> --output-dir <out_dir>/deadzone
```

Also re-run HO3D v2 `mask70` optimize (0028 row) with the chain fix before
citing it in the report.
