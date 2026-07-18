# ARCTIC direct-GT RGB+rope pose head

## Question

The released Flex15 student mostly changes rope geometry and gives only a
small ARCTIC hand-PA gain. This screen asks whether the missing ingredient is
direct hand supervision plus localized image evidence, rather than simply a
larger rope-only MLP.

## Protocol

- training data: ARCTIC P2 train stride 10, 31,006 hands from 267 sequences;
- internal split: episode-disjoint 27,882 train / 3,124 validation hands;
- external test: the frozen 3,888-hand ARCTIC P2-val stride-10 protocol from
  0065/0074;
- base prediction and MANO state: frozen WiLoR; the release inference-cache
  builder is reused only to package the aligned base pose and rope inputs;
- B: a shared per-finger pose-residual head conditioned on current pose and
  five correct rope values;
- C: the same head plus five finger queries over a cached 4x3 grid of frozen
  WiLoR image tokens;
- supervision: differentiable PA-aligned 3D joint loss, wrist-relative 3D
  auxiliary loss, rope consistency, and a small pose-delta penalty.

The checkpoint and run artifacts are under
`/data/wentao/ropetrack/runs/direct_pose_head_20260718`. Raw score JSON values
are in cm; the table below follows the repo convention and reports mm.

## External ARCTIC P2-val result

Negative deltas are improvements relative to the same WiLoR base.

| Cell | PA joint | Delta | PA mesh | Delta | Camera-frame joint | Delta |
|---|---:|---:|---:|---:|---:|---:|
| WiLoR base | 6.695 | - | 6.564 | - | 52.272 | - |
| B: pose + correct rope | 6.289 | -0.406 | 6.157 | -0.406 | 51.799 | -0.472 |
| C: RGB tokens + correct rope | **5.944** | **-0.751** | **5.833** | **-0.730** | **51.582** | **-0.690** |
| C: RGB tokens, rope zero during training | 6.349 | -0.346 | 6.267 | -0.296 | 51.762 | -0.510 |
| C: RGB tokens, rope shuffled during training | 6.326 | -0.369 | 6.229 | -0.335 | 51.695 | -0.577 |
| C checkpoint, rope shuffled only at inference | 10.481 | +3.786 | 10.093 | +3.530 | 54.084 | +1.812 |
| C checkpoint, rope zero only at inference | 12.367 | +5.672 | 12.006 | +5.442 | 56.468 | +4.196 |

The correct RGB+rope cell reduces PA joint error by 11.2%. It beats B by
0.345 mm and the separately trained RGB-only cell by 0.404 mm. Shuffling rope
during training falls back to approximately the RGB-only result. Preserving
the trained model and disrupting rope only at inference is strongly harmful;
the marginal-preserving shuffle is the cleaner causal control, while the
zero-rope result is an out-of-distribution stress test.

The camera-frame gain is much smaller than the PA gain. Scale+translation
aligned joint error changes from 14.441 mm for base to 14.335 mm for C, while
the RGB-only cell is 14.235 mm. Therefore the supported claim is improved
aligned hand shape, not solved global wrist/camera articulation.

## Training health

All four training cells completed normally with early stopping:

| Cell | Best internal PA | Best epoch |
|---|---:|---:|
| B | 4.143 mm | 26 |
| C RGB+rope | **3.786 mm** | 27 |
| C RGB-only | 4.383 mm | 33 |
| C shuffled-train | 7.321 mm | 12 |

The external ordering agrees with the internal screen without using the
external split for checkpoint selection.

## Failures and fixes

1. Jobs `185667` and `185668_[0-1]` failed immediately because the isolated
   worktree contained an empty `third_party/wilor` submodule directory. Link
   the pinned worktree to the populated primary WiLoR checkout before any GPU
   submission. Resubmitted base/features jobs `185708` and `185709_[0-1]`
   completed.
2. Release-cache job `185710` looked for `base/pred.json`. The exporter stores
   joint-only benchmark payloads at `base/eval_input/pred.json`; using that
   existing file avoided rerunning WiLoR. The corrected chain `185729`-`185733`
   and zero-rope follow-up `185761`-`185762` all completed with exit code 0.

## Decision

**Continue the localized RGB+rope direct-GT line.** This is the first ARCTIC
cell in this phase that materially reduces actual hand PA rather than only a
rope proxy, and the matched controls show that RGB and correctly paired rope
are complementary.

Do not call it a release model yet. It is one ARCTIC train/val protocol with a
frozen backbone. The next cheapest gate is cross-dataset transfer of this
frozen C checkpoint on existing natural HOT3D/HO3D caches. Only if that passes
should the more expensive last-WiLoR-block fine-tuning cell be built.
