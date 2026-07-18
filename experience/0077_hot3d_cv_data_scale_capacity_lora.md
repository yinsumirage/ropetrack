# HOT3D participant CV: data scale, head capacity, and LoRA

## Question

The earlier ARCTIC+HOT3D result used only 2,166 HOT3D training hands and one
P0014/P0015 holdout. This experiment asks whether the direct RGB+rope result
survives prospective participant cross-validation, whether its next gain
comes from more trusted data or a larger correction head, and whether a
standard WiLoR Q/V LoRA continuation can improve an already converged frozen
feature solution.

## Frozen no-leak protocol

The protocol was fixed before scoring:

- fold 0 holds out all of P0001/P0002/P0003;
- fold 1 holds out all of P0009/P0010/P0011;
- fold 2 holds out all of P0012/P0014/P0015;
- each participant is scored only with the model that excluded every frame
  from that participant;
- evaluation contains all 8,147 hands from 133 natural-visibility episodes
  over the nine participants (3,990 context and 4,157 low-visibility rows);
- the expanded HOT3D training selection is deterministic and uses only
  sequence identity plus official visibility: each participant contributes
  1,500 rows from each of `[0,.25)`, `[.25,.75)`, and `[.75,1]`, with no
  replacement and no pose/error-based selection;
- each fold therefore uses 27,000 unique HOT3D hands from six participants,
  plus the same 31,006 ARCTIC P2 stride-10 hands;
- the small-data control uses the natural episode rows from the same six
  training participants: 6,490/4,875/4,929 HOT3D rows in folds 0/1/2;
- all six fold bundles passed an explicit held-out-participant audit before
  training. The selection SHA-256 is
  `46b0d2c8a7d9f92a301ed2a96416b3dfbf34c74347068f5a579f21a44aab645a`.

Run artifacts are under
`/data/wentao/ropetrack/runs/direct_pose_scale_20260718`. Jobs `185839` to
`185843` built the two official-GT exports, `185912` and `185913` produced
the frozen WiLoR/base caches, `185966` audited all fold bundles, and
`185972` to `185975` trained, applied, participant-stitched, and scored the
three main cells. All retained jobs completed with exit code 0.

## Main cross-validation result

All values are joint error in mm. Negative deltas are improvements.

| Cell | Trainable head params | PA all | PA low visibility | PA context | Root-relative all | Camera-frame all |
|---|---:|---:|---:|---:|---:|---:|
| WiLoR base | 0 | 8.984 | 10.271 | 7.643 | 66.574 | **138.581** |
| small HOT3D, h128 | 250,633 | 6.369 | 6.896 | 5.820 | 66.126 | 139.540 |
| expanded HOT3D, h128 | 250,633 | **5.732** | **6.148** | 5.298 | **65.342** | 139.539 |
| expanded HOT3D, h512 | 1,985,545 | 5.644 | 6.109 | **5.160** | 65.720 | 139.337 |

The expanded h128 cell improves PA by 3.252 mm versus base. More
importantly, increasing unique HOT3D coverage improves the matched h128
model by another 0.637 mm; the paired-episode 95% CI for
`small - expanded` is `[0.533, 0.743]` mm overall and
`[0.598, 0.906]` mm on low-visibility rows.

Making the head about 7.9 times larger adds only 0.087 mm overall. The
paired CI for `expanded_h128 - expanded_h512` is `[0.008, 0.164]` mm
overall but `[-0.077, 0.152]` mm at low visibility. It also worsens
root-relative error by 0.379 mm. This is a marginal capacity effect, not the
main source of improvement.

Correct rope is necessary in every cell. Inference-time rope shuffling raises
PA from 6.369 to 12.062 (small), 5.732 to 12.357 (expanded h128), and 5.644
to 12.261 mm (expanded h512). Thus the result is not an image-only
participant recognizer.

## Participant consistency and metric boundary

The participant-held-out PA gain is not driven by a few people:

- the small cell beats base for 9/9 participants;
- the expanded h128 cell beats the matched small cell for 9/9 participants;
- expanded h128 gains versus base range from 2.301 mm (P0015) to 5.497 mm
  (P0002).

The claim remains hand shape/articulation, not solved global pose. Expanded
h128 changes camera-frame joint error from 138.581 to 139.539 mm (slightly
worse) while PA changes from 8.984 to 5.732 and root-relative error changes
from 66.574 to 65.342 mm. These metrics must not be merged into one claim.

## External transfer retention

Jobs `186015` and `186016` applied and scored all three expanded h128 fold
models on independent anchors. The table reports the three-model mean and
range; the base and earlier joint values are from 0076.

| Anchor | WiLoR base | Earlier 2,166-HOT joint | Expanded-CV mean | Three-fold range |
|---|---:|---:|---:|---:|
| ARCTIC P2-val | 6.695 | 5.994 | **5.979** | 5.919-6.029 |
| FreiHAND mask70 | 10.485 | **8.254** | 8.648 | 8.403-9.027 |
| HO3D v2 mask70 | 9.436 | **8.156** | 8.344 | 8.266-8.383 |

The larger HOT3D set preserves substantial positive transfer on all three
anchors and slightly improves ARCTIC. It does, however, give back 0.394 mm
on FreiHAND and 0.188 mm on HO3D relative to the earlier ARCTIC-dominant
mixture. Dataset coverage is now useful enough that source mixture, rather
than head size, is the next controlled variable.

## Matched last-two-block Q/V LoRA screen

A bounded fold-2 screen used P0012/P0014/P0015 only for external evaluation.
Both arms started from the same converged expanded h128 checkpoint and used
the same 52,143 training rows, 5,863 episode-disjoint internal validation
rows, batches, learning rate, and continuation budget:

- head-only continuation: 250,633 trainable parameters;
- head plus rank-8 Q/V LoRA in the final two WiLoR transformer blocks:
  332,553 trainable parameters (81,920 added LoRA parameters).

The cached input to the final two blocks was reconstructed without editing
the WiLoR submodule. At zero-initialized LoRA, the rebuilt 4x3 tokens match
the original frozen cache with mean absolute difference 0.000142 and maximum
0.003807. Both arms early-stopped after four continuation epochs and retained
epoch -1: initial internal PA was 4.473 mm, while the final observed values
were 4.546 mm (head-only) and 4.537 mm (LoRA).

On the three held-out participants, frozen h128 scores 6.4546 mm; selected
head-only continuation scores 6.4545 and selected LoRA scores 6.4545 mm. The
paired CI is centered at numerical parity. Rank-8 correct-rope inference is
still 6.13 mm better than its shuffle control. The supported conclusion is
that this standard last-two-block Q/V LoRA continuation does not improve the
already converged frozen solution under the matched budget. It is not a claim
that every LoRA placement, rank, or from-scratch schedule is impossible.

The first LoRA submission (`186031`-`186033`) was canceled after repeated
GPU-utilization samples showed 0%: random NFS reads of the 40+ GB activation
cache were starving the GPU. A CPU preprocessing job (`186045`) created a
41 MB supervision bundle plus a participant-audited 30 GB sequential cache;
the corrected jobs `186046`-`186048` performed real GPU work and completed
with exit code 0. Do not repeat random row access to large NFS memmaps inside
a GPU training loop.

## Unique-ARCTIC stride-5 mixture cell

A final predeclared data-mixture cell doubled ARCTIC P2-train coverage from
31,006 stride-10 rows to 62,000 unique stride-5 rows over the same 267
sequences. It kept each HOT3D fold, h128 model, loss, and evaluation protocol
unchanged. It did not repeat rows or select examples using observed error.
Jobs `186072`-`186078` completed data construction, two-GPU WiLoR caching,
three-fold training, participant stitching, and HOT3D scoring with exit code
0. Jobs `186085` and `186114`-`186115` completed the external anchors.

| Test | Expanded h128, AR stride 10 | AR stride 5 | Stride-5 delta |
|---|---:|---:|---:|
| HOT3D participant CV PA | 5.732 | 5.727 | -0.005 |
| HOT3D low-visibility PA | 6.148 | 6.149 | +0.000 |
| HOT3D root-relative | **65.342** | 65.792 | +0.451 |
| ARCTIC P2-val PA, three-fold mean | 5.979 | **5.894** | -0.085 |
| FreiHAND mask70 PA, three-fold mean | 8.648 | **8.394** | -0.254 |
| HO3D v2 mask70 PA, three-fold mean | **8.344** | 8.557 | +0.213 |

The paired HOT3D PA CI for `stride10 - stride5` is
`[-0.070, +0.075]` mm overall and `[-0.091, +0.091]` mm at low visibility:
doubling unique ARCTIC data neither improves nor dilutes aligned HOT3D shape.
The participant result is mixed (6/9 improve, 3/9 regress), and root-relative
error becomes significantly worse. External transfer is also a trade-off:
ARCTIC and FreiHAND recover, but HO3D regresses, including an 8.808 mm fold-2
outlier. Therefore stride 5 is not a universal replacement or a Pareto
improvement.

The first external scorer `186086` was canceled before a complete score file
because its nine cells were serialized. The exact same frozen prediction and
scoring commands were resubmitted as nine independent CPU cells (`186114`),
then aggregated without model or protocol changes (`186115`).

## Decision

**Keep expanded h128 as the primary candidate; stop blind head enlargement,
stop this LoRA continuation path, and do not promote stride-5 as a universal
replacement.** Participant-disjoint evidence assigns
about 0.64 mm to additional HOT3D coverage, versus about 0.09 mm to an
approximately 7.9x larger head and no selected gain to matched rank-8 Q/V
LoRA continuation.

The stride-5 cell proves that source coverage can move the cross-dataset
frontier without changing HOT3D PA, but no single mixture dominates every
metric. Do not tune another stride or source weight directly on these test
anchors. A future mixture method needs its own participant/source validation
split, then one frozen evaluation on HOT3D, ARCTIC, FreiHAND, and HO3D. The
paper claim supported now is localized RGB+rope direct supervision plus
trusted-data scaling under participant CV, with explicit capacity and domain
mixture limits; it is not solved global hand pose or monotonic all-domain
scaling.
