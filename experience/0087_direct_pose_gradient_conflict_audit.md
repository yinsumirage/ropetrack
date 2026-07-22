# DirectPose five-domain gradient-conflict and one-step transfer audit

Date: 2026-07-22

## Decision

- **A: STOP equal/near-equal four-core mixing.** The predeclared cosine gate
  fails, and a real fresh `AdamW(lr=3e-4)` step on one core domain causes
  significant off-domain PA regression in almost every core-to-core direction.
  A same-L2 normalized raw-gradient step is worse, so gradient scale alone does
  not explain the transfer failure.
- **B: exact fallback PASS.** With the frozen h128 checkpoint, all-valid output
  is bitwise identical to the historical head, every single-invalid finger has
  exactly zero 9D local-pose residual while the other fingers stay unchanged,
  and all-invalid output is exactly the original WiLoR local pose.
- **C: not implemented and not run.** The four matched local-decoder adaptation
  cells were gated on A and B together. A failed, so no decoder, retention loss,
  PCGrad, sampler, dataset-conditioned head, LoRA, or larger sweep was built.
- **Formal release unchanged.** RopeAlphaStudent remains the release. This audit
  concerns the experimental 45D local MANO `DirectPoseHead` only.

The strongest narrow conclusion is not “all datasets fight.” HOT3D and DexYCB
have a significant negative overall gradient cosine; InterHand conflicts
significantly with HOT3D but aligns with DexYCB and HO3D. The tested equal mix
is unsafe, while the best replacement recipe remains **NOT PROVEN**.

## Frozen protocol before results

Machine-readable protocol:
`/data/wentao/ropetrack/runs/direct_pose_gradient_conflict_20260722.protocol.json`,
SHA-256
`c63c16b337a50906caea56e3b7ce1108fa724d1f5ed052c875c56f544070ecae`.

- starting point: current h128 normal-joint triple fold-2 checkpoint,
  SHA-256
  `e3fdab428dcffbf9a22e75fa1659ccfb2e229a3fca9dd8f8343eab6efad30170`;
- frozen parts: WiLoR, MANO, betas, global orientation, camera translation,
  and scale;
- core audit/train domains: ARCTIC, HOT3D fold-2 training participants,
  HO3D v3 train, and DexYCB S1 train;
- InterHand: corrected train27k-v2 only, stress/audit only;
- EgoDex/EgoVerse: excluded;
- loss: unchanged PA L1 + `0.1 * root L1 + 0.1 * normalized-rope MSE
  + 0.001 * pose-delta MSE`;
- deterministic sampling: seed `20260722`, 12 update batches and 12 probe
  batches per dataset, batch size 32, 384 update and 384 probe rows per domain;
- update/probe separation: no shared sample and no shared episode. Subjects may
  appear on both sides and are recorded rather than falsely called held out;
- uncertainty: 2,000 paired bootstrap resamples over deterministic batch index;
- one-step primary: fresh AdamW, lr `3e-4`, weight decay `1e-4`, gradient clip
  `1.0`;
- scale control: raw gradient normalized to parameter L2 `0.15`, matching the
  observed AdamW step norms (`0.1473-0.1494`) rather than using an arbitrarily
  tiny direction step;
- equal-mix gate: all six core-pair overall-cosine CI lows must be at least
  `-0.10`, and neither step mode may produce an off-diagonal core PA/root
  regression above `0.05 mm` whose CI excludes zero.

No eval/test GT entered selection, gradient, update, or hyperparameter choice.
HOT3D used the six fold-2 training participants; DexYCB used official S1-train;
InterHand used official train/internal rows, never official val/test.

## Fixed sample boundary

| Dataset | Source rows | Update / probe | Update / probe episodes | Episode overlap | Update ID SHA-256 | Probe ID SHA-256 |
|---|---:|---:|---:|---:|---|---|
| ARCTIC | 31,006 | 384 / 384 | 206 / 199 | 0 | `ac841ed442109e25a2f422af78def812aa76c4fc03dfec02314f5e705c8f4afc` | `8411371750f15233dcdae97f1ad9a4b92214be4b83295fc48d20169adc4a9b27` |
| HOT3D | 27,000 | 384 / 384 | 80 / 81 | 0 | `8e0df6a0857899d8280ee0796b5cbadf9795ac18925ab922ba342438c04e1464` | `12f11094829f974fd11b08eb9c64a9f0b70513567c5f97d4d6e684698fd1057f` |
| HO3D v3 | 27,000 | 384 / 384 | 28 / 27 | 0 | `f9a81693f3f42c44942e045112fbd39da8d3f2c85219a241b19be16c0430e4fe` | `56c0dfe4c35f8ea7bddbcaccf7b4d2531225a0e3c09fea7e1a90fbc3aaef2b73` |
| DexYCB | 27,000 | 384 / 384 | 236 / 243 | 0 | `a507948ab63e458c145211036c7d14a4d461ca00cb7b61cb8066cca98ed4656e` | `c688436b7ecdf9b8e0c4adbdcfe81cd85f60d529081a9c9a8bc898e4c9b954d3` |
| InterHand | 27,000 | 384 / 384 | 252 / 259 | 0 | `e9caec52af03ad94df298e8d0ebda62af6201ae2bd20315b2d530091e18ff088` | `31e91473cbc43679d467169103c6e7d7be55f1ada82ffb940896aff1036265ea` |

The full subject lists, source hashes, row-level batch/position manifests, and
manifest hashes are under `sample_manifests/` in the run root.

## Overall gradient cosine

Mean cosine over 12 paired batches:

| Dataset | ARCTIC | HOT3D | HO3D v3 | DexYCB | InterHand |
|---|---:|---:|---:|---:|---:|
| ARCTIC | 1.000 | -0.071 | -0.002 | -0.016 | +0.048 |
| HOT3D | -0.071 | 1.000 | +0.034 | **-0.148** | **-0.195** |
| HO3D v3 | -0.002 | +0.034 | 1.000 | +0.039 | +0.079 |
| DexYCB | -0.016 | **-0.148** | +0.039 | 1.000 | +0.352 |
| InterHand | +0.048 | **-0.195** | +0.079 | +0.352 | 1.000 |

Key 95% CIs:

- ARCTIC-HOT3D `-0.071 [-0.166,+0.028]`: uncertain negative, but its lower
  bound fails the predeclared `-0.10` gate;
- ARCTIC-DexYCB `-0.016 [-0.129,+0.093]`: uncertain, also fails that gate;
- HOT3D-DexYCB `-0.148 [-0.189,-0.104]`: significant conflict;
- HOT3D-InterHand `-0.195 [-0.259,-0.125]`: significant conflict;
- HO3D-InterHand `+0.079 [+0.005,+0.141]` and DexYCB-InterHand
  `+0.352 [+0.276,+0.408]`: significant alignment. InterHand is therefore not
  uniformly opposed to all four core domains.

## Within-domain objective directions and norms

| Dataset | PA-vs-root cosine [95% CI] | PA grad norm | weighted root norm | weighted rope norm | weighted delta norm |
|---|---:|---:|---:|---:|---:|
| ARCTIC | -0.115 [-0.252,+0.015] | 0.00725 | 0.00190 | 0.00329 | 0.000168 |
| HOT3D | +0.208 [+0.081,+0.327] | 0.01171 | 0.00642 | 0.01447 | 0.000353 |
| HO3D v3 | +0.426 [+0.263,+0.575] | 0.00678 | 0.000905 | 0.00468 | 0.000040 |
| DexYCB | **+0.517 [+0.424,+0.614]** | 0.01623 | 0.00148 | 0.00285 | 0.000131 |
| InterHand | +0.626 [+0.557,+0.692] | 0.01432 | 0.00257 | 0.00859 | 0.000204 |

The table reports component norms after applying the frozen loss weights.
The raw norms remain in `gradient_summary.json`.

**DexYCB PA and root do not point in opposite directions at this checkpoint.**
The audit therefore rejects the proposed explanation that its historical
RGB-only root regression can be assigned to PA-vs-root gradient conflict.
What is significant in DexYCB is root-vs-rope conflict:
`-0.181 [-0.312,-0.026]`. Root-vs-rope is negative on all five domains, and
PA-vs-rope is significantly negative on ARCTIC, HOT3D, HO3D, and InterHand
but positive on DexYCB. This is a narrower auxiliary-objective conflict, not
a general PA/root conflict.

## Finger-specific conflict

The mean core-pair ranking from most to least negative is:

`ring (-0.120)`, `middle (-0.047)`, `pinky (-0.030)`,
`thumb (+0.044)`, `index (+0.051)`.

The conflict is **not exclusively concentrated in thumb/ring**:

- ring HOT3D-DexYCB `-0.404 [-0.454,-0.354]` and
  HO3D-DexYCB `-0.647 [-0.735,-0.520]` are the strongest pair conflicts;
- pinky HOT3D-DexYCB is `-0.413 [-0.490,-0.332]`;
- middle ARCTIC-HOT3D is `-0.260 [-0.358,-0.148]` and
  HOT3D-HO3D is `-0.281 [-0.403,-0.125]`;
- thumb ARCTIC-HOT3D is `-0.140 [-0.232,-0.025]`, but thumb is not globally
  the most conflicted finger.

All five full 5x5 finger matrices, per-batch values, and CIs are in
`gradient_summary.json`, `gradient_raw.npz`, and the generated `report.md`.

## One-step transfer

Probe baselines before any step were PA
`3.711 / 4.511 / 1.953 / 4.852 / 6.962 mm` and root-relative
`9.573 / 66.603 / 3.508 / 8.115 / 12.295 mm` in
ARCTIC/HOT3D/HO3D/DexYCB/InterHand order.

### AdamW PA: absolute after-step value (signed delta), mm

| Source \\ target | ARCTIC | HOT3D | HO3D v3 | DexYCB | InterHand |
|---|---:|---:|---:|---:|---:|
| ARCTIC | 4.443 (+0.732) | 5.020 (+0.509) | 2.269 (+0.316) | 5.109 (+0.258) | 7.300 (+0.339) |
| HOT3D | 4.350 (+0.639) | 7.023 (+2.512) | 2.360 (+0.407) | 5.509 (+0.657) | 7.843 (+0.881) |
| HO3D v3 | 4.060 (+0.349) | 5.151 (+0.641) | 2.437 (+0.484) | 4.919 (+0.067) | 7.174 (+0.213) |
| DexYCB | 4.242 (+0.531) | 5.342 (+0.831) | 2.564 (+0.611) | **4.241 (-0.611)** | **6.875 (-0.087)** |
| InterHand | 4.068 (+0.357) | 5.089 (+0.578) | 2.371 (+0.418) | **4.663 (-0.189)** | **6.661 (-0.301)** |

### AdamW root: absolute after-step value (signed delta), mm

| Source \\ target | ARCTIC | HOT3D | HO3D v3 | DexYCB | InterHand |
|---|---:|---:|---:|---:|---:|
| ARCTIC | 10.378 (+0.804) | 66.807 (+0.204) | 3.859 (+0.351) | 8.516 (+0.402) | 12.665 (+0.370) |
| HOT3D | 9.672 (+0.099) | 66.499 (-0.103) | 4.049 (+0.541) | 8.826 (+0.712) | 14.329 (+2.035) |
| HO3D v3 | 9.803 (+0.230) | 66.769 (+0.166) | 4.082 (+0.574) | 8.182 (+0.067) | 12.738 (+0.443) |
| DexYCB | 9.823 (+0.250) | 66.497 (-0.105) | 4.037 (+0.529) | **7.956 (-0.159)** | **12.046 (-0.249)** |
| InterHand | 9.877 (+0.304) | 67.090 (+0.488) | 3.858 (+0.350) | 8.200 (+0.085) | 12.337 (+0.042) |

The off-diagonal core PA regressions are mostly large and significant. Examples:

- ARCTIC -> HOT3D `+0.509 [+0.406,+0.615] mm`;
- HOT3D -> DexYCB `+0.657 [+0.573,+0.738] mm`;
- DexYCB -> HOT3D `+0.831 [+0.760,+0.910] mm`;
- HO3D -> DexYCB is the weak exception at `+0.067 [-0.034,+0.164] mm`.

Fresh AdamW also worsens same-domain PA for ARCTIC/HOT3D/HO3D, consistent
with stepping away from a selected checkpoint when optimizer state is not
restored. DexYCB and InterHand same-domain steps improve PA; this does not make
them safe mixture sources because their off-domain effects remain mixed.

The normalized raw-gradient step has the same parameter L2 (`0.15`) but causes
multi-millimetre PA/root regressions, including positive self-PA deltas on all
five domains. This rules out a simple “one domain only has a bigger gradient”
explanation. It does not prove that every smaller raw-gradient step would fail.

The generated remote report records absolute after-step values, signed deltas,
and paired 95% CIs for PA, root, all five fingers, and both step modes. Raw
before/after/delta arrays remain in `transfer_raw.npz`.

## Exact fallback safety gate

The test and real-checkpoint gate cover all requested paths:

| Condition | Result |
|---|---:|
| all valid vs historical checkpoint | bitwise equal |
| thumb/index/middle/ring/pinky single-invalid local pose vs WiLoR | max abs `0/0/0/0/0` |
| other valid fingers vs historical checkpoint | max abs `0/0/0/0/0` |
| all invalid vs WiLoR local pose | max abs `0` |

The implementation is an outer `ExactFallbackDirectPoseHead` subclass with the
same checkpoint state keys; `third_party/` is unchanged. It structurally masks
the corresponding 9D finger residual after the historical head. This proves
local-pose numerical fallback, not physical sensor detection or calibration.

## Artifacts, jobs, and verification

- run root:
  `/data/wentao/ropetrack/runs/direct_pose_gradient_conflict_20260722`;
- `gradient_raw.npz` SHA-256:
  `be0d39c090f22c9020cf5c69565e9902844960147209dd02c7e60cca4074429a`;
- `gradient_summary.json` SHA-256:
  `a043924f2a49d865aad3bb0e9f0f4ce40ad3f5f5a1f22ee551d5b6642effc504`;
- `transfer_raw.npz` SHA-256:
  `de8301520db6cd7096a7da8e94f95f57c111d95b06ebe0cc8e23247f44d607fd`;
- `transfer_summary.json` SHA-256:
  `8b0a77ed0a370a383e9161f3bed5df486007aba53254e4a6014d814171b0a458`;
- enhanced `report.md` SHA-256:
  `b4c92edb9ee14bbf84ee0bf5d6b743d558ecdfd4672f21e6bae60db1aad038ef`;
- independent verifier `192736`: PASS for immutable inputs, training-only
  role, sample/episode separation, hashes, matrix shapes, transfer
  reconstruction, and exact fallback;
- post-report verifier `192742`: PASS, with `artifact_verification.json`
  SHA-256
  `ea680bdfcdbd127e1295f4fabb6cb5c32cd7d9b630acc8a4f638690f2935e4bc`;
- CPU protocol freeze `192718`: completed `0:0`, `00:00:43`;
- GPU smoke `192719`: failed before gradient computation because the exact
  worktree had an empty WiLoR gitlink; the partial attempt is retained;
- queued retry `192723`: cancelled after the PowerShell quoting error was
  detected, before useful work;
- corrected 1-GPU smoke `192724`: completed `0:0`, `00:02:39`, verifier PASS;
- formal 1-GPU audit `192729`: completed `0:0`, `00:02:09`, peak RSS
  `1,321,512 KiB`;
- aggregate allocated GPU time including the failed/cancelled attempts:
  about `0.10 H200 GPU-hours`; successful smoke + formal audit used
  `0.08 H200 GPU-hours`;
- exact HPC computation snapshot: detached worktree at commit `bc41bfd`, with
  only the populated WiLoR checkout and `mano_data` linked; no third-party file
  was edited. The interval-complete Markdown report was regenerated from the
  immutable JSON/NPZ results with report-only commit `bb333fd` and reverified.

## Final boundary

**STOP** the four-cell local-decoder adaptation screen under the proposed
equal/near-equal four-core premise. The minimum next gate, if this direction is
reopened, is a training-only method that first predicts a non-regressive
combination from these fixed gradient/transfer artifacts. Do not answer this
result by immediately adding PCGrad, a complex sampler, dataset conditioning,
another final-score mixture, more data, three HOT3D folds, LoRA, or a larger
head.

The rope values here are GT-derived ideal normalized geometry. Nothing in this
audit validates physical rope calibration, slack, hysteresis, wear offset,
drift, latency, dropout detection, or sensor deployment.
