# Third Party

This folder is for unmodified upstream repos:

- `hamer`: https://github.com/geopavlakos/hamer
- `wilor`: https://github.com/rolpotamias/WiLoR

Use submodules:

```powershell
git submodule update --init --recursive
```

Patch upstream code only when the wrapper path is proven insufficient.

AnyHand is no longer a runtime submodule. Use its fine-tuned checkpoints from
the ignored repo-root `pretrained_models/` path through
`ropetrack.backends.hand_predictor.HandPredictor`.
