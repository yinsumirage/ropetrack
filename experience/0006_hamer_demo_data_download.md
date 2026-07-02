# 0006 HaMeR Demo Data Download

Date: 2026-07-02

## Purpose

Replace the truncated `/data/annie/hamer_demo_data.tar.gz` copy with a fresh
download from Hugging Face.

## Job

Successful CPU job:

```text
Slurm: 161843
partition: cpu
node: server23
time: 00:46:34
```

The failed launch-only retry was `161842`; it failed because the sbatch script
was stored under login-node `/tmp`. Put sbatch scripts on shared storage.

## Command Shape

The job used the `ropetrack` conda env and the cluster proxy:

```bash
export http_proxy=http://hkuhpc.com:7999
export https_proxy=http://hkuhpc.com:7999
conda activate ropetrack
hf download hngan/hamer hamer_demo_data.tar.gz --local-dir /data/wentao/ropetrack
```

## Output

Final extracted root:

```text
/data/wentao/ropetrack/hamer_demo_data
```

Observed contents:

```text
_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth 3807742341 bytes
_DATA/hamer_ckpts/checkpoints/hamer.ckpt        2689536166 bytes
_DATA/hamer_ckpts/model_config.yaml             2072 bytes
_DATA/hamer_ckpts/dataset_config.yaml           1369 bytes
_DATA/data/mano_mean_params.npz                 1178 bytes
```

The archive passed `gzip -tv`. The intermediate `hamer_demo_data.tar.gz` and
`hamer_demo_data.tar` were removed after extraction.

## Asset Placement

CPU job `161936` moved the extracted assets into the shared model root:

```text
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/checkpoints/hamer.ckpt
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/model_config.yaml
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/dataset_config.yaml
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/data/mano_mean_params.npz
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/vitpose_ckpts/vitpose+_huge/wholebody.pth
```

The stale 0-byte
`pretrained_models/hamer_ckpts/checkpoints/model_config.yaml` was replaced with
a symlink to `../model_config.yaml`, because the current AnyHand HaMeR loader
looks for `model_config.yaml` next to the selected checkpoint.

After verification, `/data/wentao/ropetrack/hamer_demo_data/_DATA` was removed.

## AnyHand-HaMeR Smoke

The current AnyHand wrapper does not call HaMeR's upstream `load_hamer()`.
Instead, `third_party/anyhand/scripts/rgb_predictor.py` loads
`model_config.yaml` from the selected checkpoint directory and passes it
directly to `HAMER.load_from_checkpoint(...)`.

Observed compatibility fixes for this layout:

```text
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/checkpoints/model_config.yaml -> ../model_config.yaml
/public/home/guowt2512/project/ropetrack/third_party/anyhand/third_party/hamer -> ../HaMeR
```

The root config was backed up as:

```text
/data/wentao/ropetrack/pretrained_models/hamer_ckpts/model_config.original.yaml
```

Then the active config was adjusted for AnyHand:

```text
MODEL.BBOX_SHAPE: [192, 256]
MODEL.BACKBONE.PRETRAINED_WEIGHTS removed
MANO.DATA_DIR: mano_data
MANO.MODEL_PATH: mano_data
MANO.MEAN_PARAMS: mano_data/mano_mean_params.npz
```

Why this was needed:

- `PRETRAINED_WEIGHTS` pointed to `hamer_training_data/vitpose_backbone.pth`,
  which was not present and is not needed when loading the fine-tuned checkpoint.
- `BBOX_SHAPE` is normally inserted by upstream `load_hamer()` for ViT crops,
  but AnyHand bypasses that helper.
- The downloaded HaMeR config pointed at its local `_DATA/data` layout; AnyHand
  already provides `mano_data`, so the duplicate `hamer_ckpts/data` copy is not
  required once the config points at AnyHand's path.

GPU smoke chain:

```text
161945 failed: AnyHand expected third_party/hamer but the submodule is HaMeR.
161950 failed: PRETRAINED_WEIGHTS referenced a missing vitpose backbone file.
161957 failed: config still pointed at removed hamer_ckpts/data MANO files.
161968 passed: AnyHandPredictor(backend='hamer') loaded on H200 and returned 2 hands.
```
