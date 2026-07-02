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
