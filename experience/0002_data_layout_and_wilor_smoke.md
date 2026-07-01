# 0002 Data Layout And WiLoR Smoke

Date: 2026-07-02

## Remote Data Layout

Base path: `/data/wentao/ropetrack`

- `FreiHAND/`
  - `raw/FreiHAND.tar.gz` exists.
  - Not extracted yet; do extraction on a CPU allocation, not the login node.
- `HO3D_v2_eval/`
  - Expanded evaluation data.
  - Images: `evaluation/<sequence>/rgb/*.png`
  - Metadata: `evaluation/<sequence>/meta/*.pkl`
  - Ground truth: `evaluation_xyz.json`, `evaluation_verts.json`
  - Sequences observed: `AP10`, `AP11`, `AP12`, `AP13`, `AP14`, `MPM10`,
    `MPM11`, `MPM12`, `MPM13`, `MPM14`, `SB11`, `SB13`, `SM1`
- `HO3D_v3/`
  - Renamed from `HO-3D`.
  - `raw/HO3D_v3_segmentations_rendered.zip` exists.
  - Treat as later training/data work, not the first clean v2 benchmark.

## Smoke Result

AnyHand-WiLoR ran on GPU with `torch==2.5.1+cu118` and detected two hands on
`third_party/anyhand/WiLoR/demo_img/test1.jpg`.

Command shape:

```bash
cd ~/project/ropetrack/third_party/anyhand
python - <<'PY'
from scripts.rgb_predictor import AnyHandPredictor
p = AnyHandPredictor(backend="wilor", batch_size=1)
hands = p.predict("WiLoR/demo_img/test1.jpg")
print("num predictions:", len(hands))
for h in hands[:3]:
    print(h.backend, h.keypoints_3d.shape, h.vertices.shape, h.bbox, h.score)
PY
```

Observed output shape:

```text
num predictions: 2
wilor (21, 3) (778, 3) [265 156 423 287] 0.742...
wilor (21, 3) (778, 3) [50 214 149 360] 0.700...
```

## Lesson

Use `scripts.rgb_predictor.AnyHandPredictor` as the primary integration path.
WiLoR's original `demo.py` can run with local symlinks, but it dirties the
submodule and is not the path to wire into ropetrack.

For future extraction or manifest generation, request a CPU allocation first.
