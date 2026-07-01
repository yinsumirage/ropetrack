# 0003 Data Storage And Extraction

Date: 2026-07-02

## Jobs

CPU jobs used:

- `161188`: moved AnyHand checkpoints to shared data storage; extracted outer
  FreiHAND tar and HO3D_v3 segmentation zip.
- `161196`: extracted nested `FreiHand.tar`.
- `161199`: extracted nested `FreiHAND_pub_v2.zip`.

## Shared Models

Shared checkpoint root:

```text
/data/wentao/ropetrack/pretrained_models
```

HPC AnyHand path is now a symlink:

```text
~/project/ropetrack/third_party/anyhand/pretrained_models
  -> /data/wentao/ropetrack/pretrained_models
```

Observed files:

```text
anyhand_wilor.ckpt
detector.pt
model_config_wilor.yaml
hamer_ckpts/checkpoints/anyhand_hamer.ckpt
hamer_ckpts/checkpoints/model_config.yaml
```

Caveat: `hamer_ckpts/checkpoints/model_config.yaml` was observed as 0 bytes.
Re-check HaMeR assets before running HaMeR.

## Extracted Data

FreiHAND usable root:

```text
/data/wentao/ropetrack/FreiHAND/FreiHAND/FreiHand
```

Observed FreiHAND counts:

```text
training/rgb: 130240
training/mask: 32560
evaluation/rgb: 3960
```

Key FreiHAND files:

```text
training_K.json
training_mano.json
training_scale.json
training_verts.json
training_xyz.json
evaluation_K.json
evaluation_scale.json
annotations/freihand_train.json
annotations/freihand_val.json
annotations/freihand_test.json
```

HO3D v2 eval remains:

```text
/data/wentao/ropetrack/HO3D_v2_eval/evaluation
/data/wentao/ropetrack/HO3D_v2_eval/evaluation_xyz.json
/data/wentao/ropetrack/HO3D_v2_eval/evaluation_verts.json
```

Observed HO3D v2 eval RGB count: `11524`.

HO3D v3 current useful pieces:

```text
/data/wentao/ropetrack/HO3D_v3/sample/image
/data/wentao/ropetrack/HO3D_v3/train/*/seg
```

Observed counts:

```text
sample/image: 98
train/*/seg files: 90469
```

Known archives were removed after successful extraction.

## Next

For the first benchmark, use FreiHAND and `HO3D_v2_eval`. Treat HO3D v3 as
later training/data material until its full RGB/annotation protocol is audited.
