# Scripts

Current benchmark entrypoints:

- `eval.py`: config-driven benchmark export and eval entrypoint.
- `eval_parallel.py`: local evaluator used by benchmark exports.
- `make_hard_images.py`: hard-image split generator.

Typical usage:

```powershell
python scripts\eval.py --dataset ho3d_v2 --method wilor_anyhand --run-eval
```

Hard split roots are generated as normal dataset roots and then selected by
dataset config name:

```bash
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/mask70 --effect mask --severity 0.70 --limit 0
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/tip_square80 --effect tip_square --severity 0.80 --limit 0
python scripts/make_hard_images.py --dataset ho3d --input-root /data/wentao/ropetrack/HO3D_v2_eval --output-root /data/wentao/ropetrack/hard/ho3d_v2/finger_end80 --effect finger_end --severity 0.80 --limit 0
python scripts/eval.py --dataset ho3d_v2_mask70 --method wilor_anyhand --run-eval
```

Dataset roots live in `configs/datasets/*.yaml`. Method/backend checkpoint
settings live in `configs/experiments/clean_baseline.yaml`.

Do not add empty script files. Add each script when it can run against local
data or a tiny fixture.
