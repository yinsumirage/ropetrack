# Scripts

Current benchmark entrypoints:

- `eval.py`: config-driven benchmark export and eval entrypoint.
- `eval_parallel.py`: local evaluator used by benchmark exports.
- `make_hard_images.py`: hard-image split generator.

Typical usage:

```powershell
python scripts\eval.py --dataset ho3d_v2 --method wilor_anyhand --run-eval
```

Dataset roots live in `configs/datasets/*.yaml`. Method/backend checkpoint
settings live in `configs/experiments/clean_baseline.yaml`.

Do not add empty script files. Add each script when it can run against local
data or a tiny fixture.
