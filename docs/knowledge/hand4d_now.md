# hand4D/now Knowledge Summary

Source: `E:\Desktop\hand4D\now`.

## Current Direction

Build a separate implementation repo for wrist RGB + fingertip-to-wrist rope
distance + MANO/keypoints. Start with FreiHAND and HO3D v2, reproduce clean
HaMeR/WiLoR baselines, then add hard benchmarks and rope experiments.

## Repo Strategy

- Keep outer repo clean: data, manifests, eval, visualization, wrappers, outputs,
  and experiment records.
- Put HaMeR and WiLoR under `third_party/`.
- Do not start by refactoring HaMeR or filling in WiLoR training code.
- Treat AnyHand as a checkpoint/provenance source, not as a runtime submodule.

## Data Strategy

```text
raw dataset
  -> one preprocessing pipeline
  -> one processed sample format
  -> all backends read the same manifest
  -> all predictions use one schema
  -> one evaluator
```

Important protocol decisions:

- Internal 3D units are meters (metrics reported in mm); the original
  hand4D-era "millimeters" decision was superseded by the implementation.
- Bboxes are original-image pixel `xyxy`.
- Every sample needs a stable `sample_id`.
- Distinguish camera-space and root-relative coordinates.
- Do visual coordinate audits before training.

## First Week Scope

1. Stable links for `data/raw/freihand` and `data/raw/ho3d_v2`.
2. Small manifests for FreiHAND and HO3D v2.
3. Overlay checks for 20 images from each dataset.
4. HaMeR/WiLoR checkpoints running through one local predictor path.
5. Minimal MPJPE and fingertip error evaluator.

## Do Not Do Yet (status as of 2026-07-08)

- Do not build one large unified training framework. (still holds)
- Do not start with DexYCB, HOT3D, EgoExo4D, or custom wrist data.
  (HO3D v3 has since become a first-class dataset — baselines + P2 teacher
  source; DexYCB is a next-phase candidate, advisor-steered)
- Do not train rope models before coordinates/units/root/joint order are
  audited. (done: audits passed, rope models trained and released — RELEASE.md)
- Do not start temporal models before single-frame rope value is shown.
  (single-frame value is now shown; temporal remains future work)
