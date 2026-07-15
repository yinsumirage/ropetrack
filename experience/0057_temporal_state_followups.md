# Temporal state follow-ups and gate probes

Date: 2026-07-15

## Question

After the clean-prefix oracle passed, which details are necessary for a useful
tracker: state freshness, safe state updates, image-side visibility, or
rope-side arbitration between current and trusted fingers?

## Protocol and jobs

- Follow-up oracle implementation: `ac87790b31b61b06bb5aea98d1acc0a0609f593a`.
- Cached visibility probe: `31e25fe54e46422d8cd6a87145697805e9947f9b`.
- Pooled image visibility probe: `9694fbca9b4e8af3582de1e29faa40465148502e`.
- Rope state arbitration: `b3219dbb9430a719e52e99c7ca58007a7ac83ef6`.
- Run root: `/data/wentao/ropetrack/runs/temporal_state_followups_20260715`.
- Evaluation remains the same 153 complete HO3D v3 episodes and 2,000 paired
  sequence bootstrap samples as `0056`.
- All pose methods reuse dense WiLoR, MANO, K1, and rope caches. Only the
  image-visibility probe exports new frozen pooled WiLoR backbone features.

Jobs `181099` and `181100` generated and scored the 30 deterministic follow-up
methods. Arrays `181101_[0-4]` and `181173_[0-14]` produced direct rope-control
and last-clean paired comparisons. Job `181218` ran the cached-feature gate.
Jobs `181264` and `181265` generated and scored rope-residual arbitration;
`181282` is its direct shuffled-gate comparison. Jobs `181225_[0-1]` exported
the train/eval pooled image features and `181226` trained and evaluated image
visibility gates. The first generation attempt
`181096` failed before writing predictions because its pinned worktree had a
bad MANO symlink. Correcting the symlink was the only retry change.

## State freshness and aggregation

The canonical state is the last clean frame. Positive deltas below are worse.

| State source | Masked PA | Delta vs last-clean + K1 | 95% paired CI |
|---|---:|---:|---:|
| last clean | 7.2003 mm | 0 | - |
| five frames old | 7.3764 mm | +0.1761 mm | `[+0.0431, +0.3154]` |
| 15 frames old | 7.6341 mm | +0.4337 mm | `[+0.2920, +0.5732]` |
| 30 frames old | 7.8106 mm | +0.6103 mm | `[+0.3953, +0.8229]` |
| rotation mean of last five | 7.2592 mm | +0.0588 mm | `[-0.0127, +0.1328]` |
| rotation medoid of last five | 7.2901 mm | +0.0897 mm | `[+0.0098, +0.1725]` |

State freshness matters monotonically. Prefix averaging does not recover an
upper bound: the mean is statistically tied/slightly worse, and the medoid is
significantly worse. Keep exactly the last trusted state.

## False-update sensitivity

| Pollution pattern | Masked PA | Delta vs last-clean + K1 | 95% paired CI |
|---|---:|---:|---:|
| trust first three masked frames, then freeze | 8.8710 mm | +1.6706 mm | `[+1.3150, +2.0789]` |
| one false update at masked frame 1 | 8.8347 mm | +1.6344 mm | `[+1.3080, +2.0115]` |
| one false update at masked frame 15 | 8.5505 mm | +1.3502 mm | `[+1.0848, +1.6706]` |
| one false update at masked frame 30 | 8.0260 mm | +0.8256 mm | `[+0.6307, +1.0489]` |

Even a single false-clean decision at mask onset removes almost the entire
state gain. The first learned gate therefore needs extremely low masked
false-clean rate and must detect the first masked frame; ordinary average
classification accuracy is not a sufficient promotion metric.

## Selector ceilings

| Selector | Masked PA | Delta vs last-clean + K1 | Interpretation |
|---|---:|---:|---|
| known geometric occluded fingers | 7.8836 mm | +0.6832 mm | local mask geometry misses global visual corruption |
| ideal GT frame selector | 6.8753 mm | -0.3250 mm | useful additional frame-quality ceiling |
| ideal GT per-finger selector | 6.5845 mm | -0.6158 mm | largest remaining state-fusion ceiling |
| last-clean + 400-step rope TTO | 7.0846 mm | -0.1157 mm | real but below the 0.15 mm expansion bar |
| last-clean + GT tip objective | 8.1004 mm | +0.9001 mm | smoother but worse PA |
| last-clean + GT chain objective | 8.0208 mm | +0.8205 mm | smoother but worse PA |

The ideal per-finger selector improves masked tips by 1.1253 mm over canonical
last-clean, but gives back most of the canonical acceleration/jitter benefit.
This is an accuracy ceiling for learned quality arbitration, not evidence for
hard switching at every frame.

## Cached pose/rope visibility probe

Nine causal features from pose velocity/acceleration, beta drift, EMA
deviation, and rope residual were trained with sequence-disjoint train/val
splits. Phase labels were targets only, never inputs.

| Model | Val AUC | Eval AUC | Eval masked false-clean | Eval clean false-freeze |
|---|---:|---:|---:|---:|
| linear, default 0.5 | 0.922 | 0.720 | 5.73% | 77.91% |
| MLP16, default 0.5 | 0.948 | 0.722 | 2.87% | 77.42% |
| MLP16, val 0.1% false-clean threshold | - | - | 0.0109% | 97.87% |

The low-dimensional gate has severe train-to-eval shift. A threshold safe
enough for state integrity freezes almost every clean frame and cannot refresh
the state. Do not add capacity or tune another threshold grid on these inputs.

## Rope-residual arbitration

The current and last-clean K1 poses were compared per finger by their residual
to the current true rope reading. A shuffled-state-residual control preserves
the same 50.7% state-selection rate.

| Rule | State fraction | Masked PA | Masked acceleration | Masked jitter |
|---|---:|---:|---:|---:|
| last-clean + K1 | 100% | 7.2003 mm | 5842 | 5736 |
| residual margin 0 | 50.7% | 7.8942 mm | 6929 | 6847 |
| residual margin 0.02 | 26.8% | 8.2842 mm | 6926 | 6848 |
| residual margin 0.05 | 9.0% | 8.6166 mm | 6719 | 6634 |
| margin 0, shuffled gate | 50.7% | 7.9475 mm | 7915 | 7839 |

The best residual selector is 0.6939 mm worse than canonical last-clean, with
paired CI `[+0.5048, +0.8840]` mm, and hard switching damages dynamics. Rope
fit does matter: the true margin-0 gate beats its selection-rate-matched
shuffled control by 0.0534 mm, CI `[-0.0908, -0.0145]` mm, and by 0.1646 mm
on masked tips. That real signal is still far too small to offset the large
loss versus always holding last-clean. Rope is useful for correcting flexion
after state selection; it is not a reliable proxy for visual-state quality.
Stop this selector family.

## Pooled image visibility probe

The frozen WiLoR backbone caches have shapes `(83325, 1280)` for train and
`(20137, 1280)` for eval. Linear and MLP16 classifiers use sequence-disjoint
train/validation episodes; phase labels are targets only, not inputs.

| Gate | Val AUC | Eval AUC | Threshold | Masked false-clean | First-mask miss | Clean false-freeze |
|---|---:|---:|---|---:|---:|---:|
| image linear | 0.9999 | 0.9982 | default 0.5 | 0.0109% | 0 | 6.97% |
| image linear | 0.9999 | 0.9982 | val zero false-clean | 0 | 0 | 9.39% |
| image + cached linear | 0.9999 | 0.9982 | default 0.5 | **0** | **0** | 6.94% |
| image + cached linear | 0.9999 | 0.9982 | val zero false-clean | **0** | **0** | 9.89% |
| image MLP16 | 0.9994 | 0.9979 | default 0.5 | 0.0327% | 0 | 6.51% |

Pooled features failed earlier as a direct pose-repair input, but they are
excellent for the narrower visibility decision. Unlike the cached-only gate,
safe image thresholds still update 90-93% of clean frames. The linear model is
strictly safer than MLP16 here, so only the linear variants are promoted to a
causal state application. Jobs `181301`, `181302`, and `181303` apply, score,
and rope-control those variants without phase input.

## Learned causal state result

The gate updates the trusted pose whenever it predicts clean and otherwise
holds the last predicted-clean pose; K1 then applies the current rope update.
No phase value is read by this application. Jobs `181301`, `181302`, and
`181303` completed with exit code `0:0`; job `181307` directly compared the
learned result with the phase-oracle last-clean method.

| Method | Overall | Context | Masked | Recovery | Masked tips | Masked accel | Masked jitter |
|---|---:|---:|---:|---:|---:|---:|---:|
| K1 | 7.7848 | 6.8695 | 8.7622 | 6.7900 | 12.5808 | 6367 | 6280 |
| phase-oracle last-clean + K1 | 7.0728 | 6.8695 | 7.2003 | 6.7900 | 10.1113 | 5842 | 5736 |
| image linear, default | **7.0125** | 6.7691 | 7.1361 | **6.7648** | **9.9320** | 5827 | 5722 |
| image + cached linear, default | 7.0159 | **6.7664** | 7.1362 | 6.7807 | 9.9343 | **5827** | **5721** |
| image linear, zero-FN threshold | 7.0546 | 6.7683 | **7.1253** | 6.7974 | 9.9741 | 5829 | 5723 |

The simplest image-only default gate improves masked PA over K1 by 1.6261 mm,
CI `[-2.0166, -1.2736]` mm, and masked tips by 2.6488 mm, CI
`[-3.4126, -1.9463]` mm. Masked acceleration and jitter improve by 8.48% and
8.88%, respectively. Recovery is statistically tied/slightly better than K1.

| Masked horizon | K1 | Learned image-only default | Gain |
|---:|---:|---:|---:|
| 1 | 8.8063 | 6.7770 | 2.0293 mm |
| 5 | 8.8729 | 6.8886 | 1.9844 mm |
| 15 | 9.0723 | 7.0293 | 2.0430 mm |
| 30 | 8.7010 | 7.0507 | 1.6503 mm |
| 60 | 8.7446 | 7.5392 | 1.2054 mm |

The learned image-plus-cached result is statistically tied with the phase
oracle: masked delta -0.0641 mm, CI `[-0.1836, +0.0055]` mm. It does not need
to exceed the oracle; the important result is that a phase-free causal gate
recovers essentially all of its gain. Correct rope remains essential: the
image-plus-cached default method is 7.1362 mm with true rope and 9.3246 mm
with shuffled rope, paired delta -2.1884 mm, CI
`[-2.4211, -1.9080]` mm; masked tips differ by 5.3132 mm.

This is not an over-smoothing artifact. Delayed freeze and single false-update
controls substantially worsen PA, the learned gate refreshes 90-93% of clean
frames, and the rope-shuffled control collapses.

## Decision

The explicit-state hypothesis and the minimal learned causal tracker both
pass on this synthetic mask70 protocol. Preserve the freshest reliable pose,
never update it from the first corrupted frame, and use rope only as a
conditioned flexion correction.
Motion, prefix aggregation, fixed shape, low-dimensional visibility, geometric
finger masks, GT flex objectives, and rope-residual state arbitration do not
justify further expansion. Use the image-only linear default gate as the
minimal model; the nine cached features and MLP add no useful gain.

The remaining limitation is domain transfer, not mask70 accuracy: the gate was
trained and evaluated on the same synthetic black-mask family with GT-bbox
crops. Before claiming physical deployment, freeze this model and test it
without retraining on localized finger-end masks, blur/crop occlusions, and
detector-bbox crops. Do not return to larger GRUs or Transformers for that
test.

## Verification

- All final Slurm jobs listed above completed with exit code `0:0`.
- `python -m unittest discover -s tests`: 347 tests passed.
- `git diff --check`: passed.
