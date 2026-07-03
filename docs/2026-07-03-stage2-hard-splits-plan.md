# Stage 2 Hard Splits And Rope Signals Plan

Date: 2026-07-03

## Goal

Show that clean GT-bbox hand mesh baselines degrade under controlled image
corruptions, then test whether fingertip-to-wrist rope signals recover hard
case performance without damaging clean performance.

## Current Position

Stage 1 is complete enough for the next step:

- FreiHAND clean GT-bbox baselines are reproduced.
- HO3D v2 clean GT-bbox baselines are reproduced.
- HO3D v3 clean GT-bbox baselines are reproduced and reported separately.
- `scripts/eval.py` and `scripts/eval_parallel.py` are the current benchmark
  path.

## Stage 2A: Hard Image Smoke

Build a small offline hard-image generator first. It must not modify source
datasets. It writes a new mini dataset root under the HPC data area:

```text
/data/wentao/ropetrack/hard/<dataset>/<split_name>/
  evaluation/
  evaluation_xyz.json
  evaluation_verts.json
  evaluation_K.json          # FreiHAND only
  evaluation.txt             # HO3D-style roots
  hard_manifest.jsonl
```

The first smoke should use GT bboxes and one stable backend:

```text
backend: WiLoR w/ AnyHand
mode: gt_bbox
samples: 32 first, then 256 or 512
datasets: FreiHAND and HO3D v2 first
```

Hard effects to start with:

- `mask`: black rectangle inside the GT bbox.
- `blur`: Gaussian blur inside the GT bbox.
- `crop`: overwrite one bbox-side strip with black pixels.
- `mixed`: deterministic per-sample choice among the above.

The perturbation must happen inside the GT bbox image path. The bbox itself
stays unchanged, so any drop is model robustness, not detector failure.

## Stage 2B: Hard Split Candidate

Once the 32-sample smoke works:

1. Generate a 256 or 512 sample hard root for FreiHAND and HO3D v2.
2. Run clean subset and hard subset with the same backend and sample count.
3. Compare paper-style metrics:
   - FreiHAND: PA-MPJPE, PA-MPVPE, F@5, F@15.
   - HO3D: AUCj, PA-MPJPE, AUCv, PA-MPVPE, F@5, F@15.
4. Tune severity only until the split is meaningfully worse but still visually
   plausible.

## Stage 2C: Rope Labels

Only after hard degradation is established, add rope labels:

```text
data/rope/<dataset>_rope.jsonl
```

Each row should be keyed by `sample_id` and store five fingertip-to-wrist
distances:

```json
{
  "sample_id": "AP10/0000",
  "rope_dist_mm": [0, 0, 0, 0, 0],
  "rope_valid": [true, true, true, true, true],
  "source": "gt_joints"
}
```

For the first pass, this can be a plain JSONL file. Do not build a database or
large converted dataset until training code proves it needs one.

## Stage 2D: Training Direction

Training should wait until hard split evidence is clear. The likely training
path is:

1. Keep clean and hard images as offline files.
2. Add rope labels as an auxiliary per-sample input or loss target.
3. Run ablations on the same manifest:
   - RGB only.
   - RGB + hard.
   - RGB + rope.
   - RGB + noisy rope.
   - RGB + rope dropout.
4. Report both recovery on hard cases and no major regression on clean cases.

## HPC Rules

Use Slurm for all remote generation and benchmark work:

- CPU job: hard image generation and metadata copying.
- GPU job: model inference.
- CPU job: evaluation.

If syncing through Git on the cluster, set:

```bash
export http_proxy=http://hkuhpc.com:7999
export https_proxy=http://hkuhpc.com:7999
```

Keep login-node work to copying code, checking small files, and submitting jobs.

## Immediate Commands

First local checks:

```powershell
python -m unittest discover -s tests
python -m py_compile scripts\make_hard_images.py
```

First remote smoke shape:

```bash
python scripts/make_hard_images.py \
  --dataset ho3d \
  --input-root /data/wentao/ropetrack/HO3D_v2_eval \
  --output-root /data/wentao/ropetrack/hard/ho3d_v2/mask_s1_32 \
  --effect mask \
  --severity 0.45 \
  --limit 32 \
  --seed 7
```

Then run `scripts/eval.py` on the generated root with `--dataset ho3d_v2`,
`--ho3d-root /data/wentao/ropetrack/hard/ho3d_v2/mask_s1_32`, and
`--mode gt_bbox`.
