# 0046 P3 Feature Cache Jobs

Date: 2026-07-07

## Context

P2 multi-teacher experiments are still running, but the next P3 asset can be
queued independently: frozen WiLoR backbone feature caches for the FreiHAND
mask70 evaluation and training splits. These caches are intended for the later
rope-conditioned head experiments, without modifying the frozen benchmark
pipeline.

## Code Synced

Local commit synced to HPC:

- `f77a701 Add P3 feature cache extraction`

Main additions:

- `scripts/rope_head/extract_feature_cache.py`
  - Uses the same dataset adapter and `CrossImageBBoxDataset` crop path as the
    existing benchmark/export flow.
  - Captures `model.backbone` features via a forward hook during the normal full
    model forward pass.
  - Writes `feature_cache.npz` with sample ids, pooled fp32 features, metadata,
    and optional fp16 token grids.
- `tests/test_extract_feature_cache.py`
  - Covers candidate deduplication, pooling layouts, hook failure, shuffled
    batch row placement, optional token saving, missing rows, and npz roundtrip.
- `docs/2026-07-07-report-results-pack.md`
  - Adds the Act 4b multi-teacher summary section and pending slots.
- `scripts/README.md`
  - Adds the P3 feature cache utility entry.

Local checks before commit:

```text
python -m unittest tests.test_extract_feature_cache
Ran 13 tests in 0.027s - OK

python -m py_compile scripts\rope_head\extract_feature_cache.py
OK

rg -n "[^\x00-\x7F]" scripts\rope_head\extract_feature_cache.py tests\test_extract_feature_cache.py
no matches
```

## First Submission Failure And Fix

The first eval job failed quickly:

```text
170391 p3feat_eval FAILED 00:01:01 ExitCode 1:0
```

Root cause:

- WiLoR's ViT backbone returns `(pred_mano_params, pred_cam, pred_mano_feats,
  img_feat)`.
- The original feature extraction helper unwrapped tuple/list outputs by taking
  item `0`, so it selected the MANO parameter dict instead of the fourth
  `img_feat` tensor.
- The resulting symptom was:

```text
AttributeError: 'dict' object has no attribute 'dim'
```

The still-running first train job was cancelled to avoid wasting GPU time:

```text
170392 p3feat_train CANCELLED
```

Fix before resubmission:

- Select the image-feature tensor from common backbone output containers,
  searching tuple/list outputs from the end so WiLoR's `img_feat` wins over
  parameter dicts.

Follow-up checks after the fix:

```text
python -m unittest tests.test_extract_feature_cache
Ran 15 tests in 0.186s - OK

python -m py_compile scripts\rope_head\extract_feature_cache.py
OK

rg -n "[^\x00-\x7F]" scripts\rope_head\extract_feature_cache.py tests\test_extract_feature_cache.py
no matches
```

## Submitted Jobs

Run root:

```text
/data/wentao/ropetrack/runs/rope_p3_feature_cache_20260707_195700
```

Submitted from:

```text
~/project/ropetrack/.local_checks/submit_p3_feature_cache.sh
```

Jobs:

| Job | Split | Output |
|---:|---|---|
| 170391 | FreiHAND mask70 evaluation | `/data/wentao/ropetrack/features/freihand_mask70_eval_wilor.npz` |
| 170392 | FreiHAND mask70 training | `/data/wentao/ropetrack/features/freihand_mask70_train_wilor.npz` |

Both jobs were running immediately after submission:

```text
170391 gpu p3feat_e R server44
170392 gpu p3feat_t R server63
```

## Notes

- The training cache uses the existing hard root
  `/data/wentao/ropetrack/hard/freihand/mask70_wilor_training`.
- These are data-asset jobs only. They do not block the P2 multi-v2 result
  analysis, and they should not be interpreted as P3 training results.
