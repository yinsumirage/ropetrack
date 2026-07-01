# 0003 Data Storage And Extraction

Date: 2026-07-02

## Jobs

CPU jobs used:

- `161188`: moved AnyHand checkpoints to shared data storage; extracted outer
  FreiHAND tar and HO3D_v3 segmentation zip.
- `161196`: extracted nested `FreiHand.tar`.
- `161199`: extracted nested `FreiHAND_pub_v2.zip`.
- `161214`: downloaded full HO3D v3 LFS object to `raw/HO3D_v3.zip`.
- `161218`: extracted full HO3D v3 zip and removed `raw/HO3D_v3.zip`.

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
/data/wentao/ropetrack/FreiHAND
```

The initial extraction produced nested `FreiHAND/FreiHAND/FreiHand`; the useful
contents were moved up to `/data/wentao/ropetrack/FreiHAND`, and the download
shell (`.git`, `raw`, `README.md`, `metafile.yaml`, `quickstart.md`,
`.gitattributes`) was removed.

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
/data/wentao/ropetrack/HO3D_v3/train.txt
/data/wentao/ropetrack/HO3D_v3/train/<sequence>/rgb
/data/wentao/ropetrack/HO3D_v3/train/<sequence>/depth
/data/wentao/ropetrack/HO3D_v3/train/<sequence>/meta
/data/wentao/ropetrack/HO3D_v3/train/<sequence>/seg
/data/wentao/ropetrack/HO3D_v3/evaluation/<sequence>/rgb
/data/wentao/ropetrack/HO3D_v3/evaluation/<sequence>/depth
/data/wentao/ropetrack/HO3D_v3/evaluation/<sequence>/meta
/data/wentao/ropetrack/HO3D_v3/sample/image
```

Observed lightweight counts:

```text
train.txt lines: 83325
train sequences: 55
evaluation sequences: 13
sample/image: 98
```

The full HO3D v3 LFS object was downloaded on CPU job `161214`:

```text
raw/HO3D_v3.zip size: 34255013515
```

It was extracted on CPU job `161218`, then removed. The segmentation archive
`raw/HO3D_v3_segmentations_rendered.zip` was already extracted and removed.

Known extraction archives were removed after successful extraction.

## Next

For the first benchmark, use FreiHAND and `HO3D_v2_eval`. HO3D v3 is now
available for later training/data work, but audit its meta/keypoint protocol
before mixing it into any benchmark table.
