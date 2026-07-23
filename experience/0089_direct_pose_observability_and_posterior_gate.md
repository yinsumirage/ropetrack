# DirectPose observability and conditional-posterior gate

Date: 2026-07-24

## Question

This audit follows the failed equal-weight multi-domain update in 0087 and the
product matrix in 0088. It asks two narrower questions before any new LoRA,
decoder, or gate is trained:

1. Can five ideal rope values uniquely identify a 45D local MANO pose, or do
   they only disambiguate a visual pose hypothesis?
2. On natural HOT3D low-visibility frames, do the existing localized WiLoR
   tokens become specifically less useful, which would justify an
   occlusion-targeted LoRA?

This is deliberately different from the 0057/0074 visibility classifiers. It
trains no classifier, predicts no clean/occluded label, and makes no state
update. It measures fixed representations and an offline candidate ceiling.

## Frozen protocol

Run root:
`/data/wentao/ropetrack/runs/direct_pose_observability_20260724`.

The main protocol was frozen before results, SHA-256
`17c2ec910c37fb617a5c775b2f325d235e6dd12cd36e8d341109d6011da8af6b`.
It reuses the hash-locked training-only assets from 0088:

- ARCTIC uses the existing 31,006 valid training rows;
- HOT3D, HO3D v3, DexYCB, and InterHand use their fixed validated 27k assets;
- each domain contributes a deterministic 4,096-row reference bank and a
  512-row query bank from disjoint episode halves;
- reference/query IDs, episodes, subjects, hashes, and all external query IDs
  are recorded in `sample_manifest.json`;
- InterHand remains stress-only; no official eval/test GT enters training or
  protocol selection.

Localized 4x3 tokens use one fixed 32-channel Rademacher projection. The audit
compares rope, WiLoR base local pose, localized tokens, and equal-block
combinations. The external anchors are 1,024 fixed HOT3D fold-2 validation
rows (523 context, 501 low visibility) and 1,024 fixed ARCTIC P2-val stride-10
rows. HOT3D is reused participant-disjoint validation, not untouched test.

The offline candidate experiment retrieves K=64 training reference shapes
from token+base similarity, then selects among them by ideal rope distance.
Reference GT joints score the retrieved shape and target GT only defines the
top-K oracle. This is an observability ceiling, not a deployable candidate
generator.

Frozen primary gates were:

- call low-visibility tokens corrupted only if token+base rank-1 is at least
  `0.25 mm` worse than base-only, its episode-bootstrap CI is above zero, and
  context stays within `0.10 mm`;
- call rope reranking useful only if HOT3D low-visibility improves by at least
  `0.25 mm` with CI fully below zero;
- require the ARCTIC rerank not to worsen its *visual retrieval baseline* by
  more than `0.10 mm`.

The last item does not mean non-regression versus WiLoR. A post-hoc comparison
against WiLoR is reported explicitly below to prevent that misreading.

## Five ropes are informative but not identifiable

The differentiable wrist-to-fingertip rope map has rank 5 on all 64 sampled
poses in every domain:

| Domain | Median Jacobian rank | Local null-dimension lower bound |
|---|---:|---:|
| ARCTIC | 5 | 40 |
| HOT3D | 5 | 40 |
| HO3D v3 | 5 | 40 |
| DexYCB | 5 | 40 |
| InterHand | 5 | 40 |

This is the expected local limit: five scalar measurements constrain at most
five directions of a 45D pose. It does not say the remaining 40 dimensions
are all perceptually independent, but it proves that rope alone is not a
one-to-one local pose sensor.

The finite reference-bank collision diagnostic reaches the same conclusion.
For a raw normalized-rope RMS distance at most `0.02`, 35.5% of HOT3D queries
and 42.3% of ARCTIC queries find a near-rope neighbor; those nearest poses
still differ by `8.133 mm` and `11.809 mm` PA respectively. Five ropes can
strongly rank visual hypotheses, but they cannot determine the complete hand
pose without a visual/anatomical prior.

## Tokens do not show occlusion-specific collapse

On the fixed HOT3D external rows, token+base retrieval is slightly better than
base-only in both phases:

| Slice | Base-only rank-1 | Token+base rank-1 | Signed difference | 95% episode CI |
|---|---:|---:|---:|---:|
| Context | 10.558 | 10.164 | -0.394 | `[-1.062,+0.224]` |
| Low visibility | 12.112 | 11.761 | -0.351 | `[-1.207,+0.344]` |

The top-K oracle shows the same non-specific pattern: adding tokens changes
the context oracle from `5.734` to `5.896 mm` and the low-visibility oracle
from `6.498` to `6.628 mm`. Both are small degradations, not a selective
low-visibility collapse. The frozen token-corruption gate therefore fails.

This does not prove that every token is optimal under occlusion. It shows that
the current cached localized representation still contains comparable pose
hypotheses in context and low visibility. Combined with 0077, where rank-8
last-two-block Q/V LoRA gave no selected held-out gain, there is no current
evidence to restart generic or occlusion-targeted LoRA.

ARCTIC is different: token+base rank-1 is `+0.469 mm` worse than base-only,
CI `[+0.290,+0.655]`. Same-domain training retrieval also exposes a large
HO3D token mismatch (`7.880 mm` token+base versus `3.528 mm` base-only).
These are representation/domain diagnostics, not proof that a larger token
adapter will repair the shared multi-domain objective.

## Dataset manifolds differ

Rank-1 PA using base-pose similarity is shown below; rows are query domains
and columns are reference domains:

| Target \\ reference | ARCTIC | HOT3D | HO3D | DexYCB | InterHand |
|---|---:|---:|---:|---:|---:|
| ARCTIC | **8.205** | 12.490 | 13.117 | 10.957 | 14.028 |
| HOT3D | 13.060 | **9.200** | 14.704 | 10.736 | 16.610 |
| HO3D | 12.510 | 12.966 | **3.528** | 13.665 | 13.424 |
| DexYCB | 10.137 | 13.236 | 10.676 | **6.520** | 13.886 |
| InterHand | 14.287 | 17.122 | 17.489 | 14.387 | **5.930** |

Every diagonal is best by a large margin. The rope-only matrix is also
diagonal-dominant. This supports the 0087/0088 conclusion that the datasets
occupy different pose/rope operating distributions; it does not by itself
prove gradient conflict, which remains established by the separate
training-only gradient and one-step audit.

## Rope is valuable inside a visual hypothesis set

K=64 candidate reranking gives:

| Dataset/slice | WiLoR | Visual retrieval | Rope reranked | Rerank minus visual [95% CI] | Top-K oracle |
|---|---:|---:|---:|---:|---:|
| HOT3D all | 9.278 | 10.946 | 8.470 | -2.476 `[-3.073,-1.882]` | 6.254 |
| HOT3D context | 7.965 | 10.164 | 8.346 | -1.818 `[-2.417,-1.169]` | 5.896 |
| HOT3D low visibility | 10.648 | 11.761 | **8.599** | **-3.163 `[-4.068,-2.274]`** | 6.628 |
| ARCTIC | 6.701 | 10.509 | 9.935 | -0.574 `[-0.796,-0.367]` | 7.088 |

The post-hoc, explicitly non-gating comparison against WiLoR is:

- HOT3D low visibility: `-2.050 mm`, CI `[-3.099,-1.012]`;
- HOT3D context: `+0.381 mm`, CI `[-0.473,+1.291]`;
- ARCTIC: `+3.234 mm`, CI `[+2.927,+3.547]`.

Thus rope has a strong low-visibility selection signal, but this nearest-
neighbor construction is not an always-on or cross-domain model.

All five HOT3D fingers improve when reranking the visual candidates. On low
visibility the signed changes are thumb `-3.075`, index `-1.845`, middle
`-3.201`, ring `-3.384`, and pinky `-3.440 mm`. The signal is not confined to
thumb/ring; index is the weakest and ring/pinky are strongest.

The retrieved reference joints carry their own camera/global orientation.
Consequently root-relative error worsens by `+8.746 mm` in context and
`+7.287 mm` in low visibility. Only PA local-shape observability is meaningful
here. A deployable local-pose posterior must decode candidate local MANO poses
under the query's frozen global orientation, betas, scale, and external wrist
6DoF.

## How many hypotheses?

After the K=64 result, a separate K-sensitivity family was frozen before its
K-specific results. All cells use the exact same sample manifest hash
`f51b101c9ecbf4951ecc5bcced73ed8d919ceb1842c0f125a077f60a0283c97b`.

| K | HOT3D low-vis reranked | Delta vs WiLoR [95% CI] | Context delta vs WiLoR [95% CI] | ARCTIC delta vs WiLoR |
|---:|---:|---:|---:|---:|
| 4 | 9.980 | -0.668 `[-1.368,+0.072]` | +0.963 `[+0.243,+1.701]` | +3.177 |
| **8** | **9.614** | **-1.034 `[-1.842,-0.178]`** | +0.572 `[-0.127,+1.275]` | +3.047 |
| 16 | 9.440 | -1.208 `[-2.057,-0.311]` | +0.573 `[-0.152,+1.305]` | +2.916 |
| 32 | 9.063 | -1.585 `[-2.630,-0.581]` | +0.351 `[-0.418,+1.177]` | +3.080 |
| 64 | 8.599 | -2.050 `[-3.099,-1.012]` | +0.381 `[-0.473,+1.291]` | +3.234 |

K=4 is not statistically reliable and significantly harms context. K=8 is
the smallest tested candidate count with a significant low-visibility gain
and no significant context change. This bounds a future hypothesis count; it
does not determine hidden width or prove that a learned K=8 head can reproduce
the reference-bank ceiling.

## Decision

1. **Keep the current h128 DirectPose as the software product candidate.**
   Its actual 0088 HOT3D/ARCTIC predictions remain the relevant product
   evidence; exact per-channel/all-missing fallback remains mandatory.
2. **Do not restart LoRA.** The audit does not find occlusion-specific token
   corruption, while the prior matched rank-8 Q/V LoRA screen found no selected
   held-out gain. A new LoRA would need a different, predeclared token target
   such as hypothesis recall under real occlusion, not visibility labels or
   ordinary continuation.
3. **Do not train all four domains equally or unlock the WiLoR decoder.** The
   0087 failure remains binding. The diagonal-dominant retrieval matrices
   strengthen the domain-shift explanation.
4. **Do not implement the K=8 head yet.** The next operation is cheaper:
   align the already-generated 0088 DirectPose predictions to these exact
   1,024 query IDs and compare them with the K=8/K=64 oracle. The unmatched
   full-slice 0088 DirectPose result is `6.148 mm` on low visibility, already
   below this audit's K=64 retrieval (`8.599 mm`) and even below its `6.628 mm`
   oracle, so additional headroom is not currently established.
5. Only if the exact same-row oracle beats the current DirectPose head should
   one screen a HOT3D-centered conditional K=8 local-pose posterior, with a
   zero/WiLoR candidate always present, rope likelihood used for hypothesis
   weighting, exact invalid-channel fallback, ARCTIC one-sided retention, and
   HO3D/DexYCB/InterHand used as frozen training-only audit domains rather
   than equal update sources.

The practical software path today remains:

`hardware validity/CRC/staleness checks -> per-finger validity mask -> current
DirectPose only on valid channels -> exact WiLoR local pose otherwise`.

The model cannot infer plausible in-range bias, slack, hysteresis, or stale
measurements reliably from five values alone. Physical logging/calibration is
still required.

## Jobs, artifacts, and verification

- failed pre-compute smoke `194654`: empty WiLoR gitlink, `00:01:21`;
- pending retry `194674`: cancelled before allocation after the request was
  identified as unnecessarily large;
- corrected smoke `194680`: PASS, `00:05:22`, peak RSS `1,559,524 KiB`;
- main K=64 audit `194684`: PASS, `00:00:54`;
- K-sensitivity array `194693_0`--`194693_3`: all PASS; at most two GPUs
  active concurrently;
- aggregate allocated GPU time including the failed smoke is about
  `0.222 GPU-hours`; successful jobs use about `0.200 GPU-hours`;
- wall time from first GPU start through the final array completion is about
  `00:25:02`;
- exact code snapshot: `c2a8fa98cea4346f70ef4a9b76c453b638d26e1a`;
- main raw SHA-256:
  `1ed1988e333e11a3087c2ee0701d58ffdd39385a179b984c7948520cc36f5fe5`;
- main sample manifest SHA-256:
  `f51b101c9ecbf4951ecc5bcced73ed8d919ceb1842c0f125a077f60a0283c97b`;
- post-hoc WiLoR comparison SHA-256:
  `f1a3174bce99d7a2369ef3aa962f30fa544de7a2f5eb966eb676408f93b01131`;
- K-sensitivity post-hoc CI SHA-256:
  `e46ec48b1b352ca38243da5ff115560bcc1222bba162087d6d77ec34c19e8218`;
- independent verifier: PASS, SHA-256
  `f106aed58ae2fcecec229e29d5bc48efe3b57a38c51a8daca309d6ef9c5804cc`.

The clean remote worktree and pinned WiLoR submodule remained unmodified.
No data, cache, checkpoint, prediction, metric, or large figure is committed.

## Research boundary

All rope values are GT-derived ideal normalized wrist-to-fingertip geometry.
They are not measurements from the planned routed-string glove. This audit
does not validate physical routing, calibration, wear, drift, hysteresis,
latency, dropout detection, or commercial deployment.
