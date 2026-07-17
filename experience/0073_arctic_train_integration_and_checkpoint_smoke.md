# ARCTIC training integration and checkpoint smoke

Date: 2026-07-18

## Layout and protocol

The raw root remains
`/data/wentao/datasets/arctic/unpack/arctic_data/data`. This is the official
download script's extracted `unpack/arctic_data/data` layout. RopeTrack receives
the innermost `data` directory as `--arctic-root`; no adapter depends on the
outer `unpack` directory name.

Official P2 train inspection (job 185164) found 267 sequences and 187,050
view-0 images from subjects `s01,s02,s04,s06,s07,s08,s09,s10`. Validation is
the subject-disjoint `s05/view0`; hidden test is `s03/view0`.

`scripts/prepare_arctic.py` now supports `--split train|val`. Both use:

- annotation index = image basename minus subject `ioi_offset`;
- official first/last 10-frame exclusion inherited from the split file;
- 0.3-scaled view-0 image/intrinsics in OpenCV camera metres;
- distorted view-9 joints scaled by 0.3 only for GT bbox construction;
- official native MANO kinematic joints reordered to WiLoR/OpenPose order;
- side-specific native MANO mesh decoding when explicitly requested.

Train defaults to joint-only export because the existing teacher/student path
does not consume GT meshes. `--write-verts` remains available for small MANO
checks. `--skip-image-check` avoids 310k redundant NFS stat calls only after the
separate 2.19M-image integrity audit in experience 0071.

## Mapping and training smoke

CPU job 185169 exported 16 P2-train hand samples with both joints and 778-vertex
MANO meshes. All image paths existed; manifest/xyz/verts/rope rows matched;
sample ids, sides, subject membership, and shapes passed. Six projected-joint
visualizations were generated. The first frames were strongly object-occluded,
so the already saved visible-hand cross-sequence overlays from P2 val were also
rechecked: left/right skeletons and bboxes aligned on capsule machine, notebook,
and phone sequences.

GPU job 185170 ran WiLoR original, MANO-cache reconstruction, and the frozen
release Flex15 student on the 16 training samples with zero failures. This is a
connectivity smoke, not a representative benchmark: the early `box_grab`
frames hide the hands behind the object. Base/Flex15 errors in mm were:

| method | raw joint | PA joint | raw mesh | PA mesh |
|---|---:|---:|---:|---:|
| WiLoR | 326.839 | 7.452 | 323.139 | 7.391 |
| Flex15 | 326.683 | 7.111 | 322.951 | 7.047 |
| signed delta | -0.157 | -0.341 | -0.188 | -0.344 |

Rope mean absolute residual fell `0.09357 -> 0.07195` (closure `0.2310`). Do
not use these 16 cherry-picked/occluded samples to claim model quality.

The primary full P2-val result remains 38,921 samples with zero failures:
WiLoR raw/PA joint `51.8973/6.7032` mm and raw/PA mesh `51.6889/6.5707` mm;
Flex15 `51.7974/6.7265` and `51.5846/6.5876` mm. Signed deltas are
`-0.0999/+0.0233/-0.1042/+0.0169` mm. Rope closure is strong, but geometry is
side-dependent and effectively flat.

## Full training assets

CPU job 185179 completed the durable P2-train export at
`/data/wentao/ropetrack/processed/arctic/p2_train`:

- 310,078 manifest rows and GT joint rows;
- 157,017 right and 153,061 left hands;
- 310,078 rope-label rows;
- 267 sequences across the eight official training subjects;
- no training mesh JSON, by design.

Job 185172 was cancelled because repeated NFS image stats were unnecessary
after the full acquisition audit. Job 185177 exposed and closed one integration
bug: the legacy validation-count assertion lacked a `split == val` guard.

ARCTIC is now ready for existing WiLoR teacher export, Flex15 optimization, and
multi-dataset student distillation. A full new ARCTIC teacher/retraining run was
not started; it is a separate experiment, not an integration requirement.
