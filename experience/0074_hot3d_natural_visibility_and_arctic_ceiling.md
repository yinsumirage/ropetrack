# HOT3D natural visibility, trusted state, and ARCTIC action ceiling

## Question

This round tests three decisions raised after adding ARCTIC, HOT3D, and
EgoDex:

1. whether the larger datasets also need artificial finger masks;
2. whether a deployable gate can recognize natural occlusion rather than a
   synthetic mask flag;
3. whether the small Flex15 correction module, rather than its observation,
   is the limiting factor.

The run does not use EgoDex as MANO ground truth. The current EgoDex scaling
decision remains the confidence-filtered, all-task teacher result in 0073.

## Protocol correction before scoring

The first HOT3D selection exported 2,812 requested rows but only 2,768 rows
survived. All 44 missing rows had official
`mask_hand_pose_available=False`. Silently accepting the shorter export would
have biased the natural episode protocol, so `prepare_hot3d.py` now requires
that official mask and exact selection export remains strict.

The corrected audit contains:

- 136 Aria public-GT sequences and 9 participants;
- 821,286 rows after all four required official masks;
- 141,433 rows below visibility 0.25;
- 1,240 high-prefix to visibility-below-0.25 candidate runs;
- candidates in 135 sequences and all 9 participants.

The frozen screen samples 5 episodes per participant: 45 unique sequences,
2,822 frames, 1,350 context rows, and 1,472 natural low-visibility rows. It
uses 7 participants for gate fitting and reserves P0014/P0015 for gate
validation. This is a natural-hard screen, not a HOT3D leaderboard protocol.

## ARCTIC: action space is not the main bottleneck

The ARCTIC P2 validation stride-10 screen uses 3,888 hands. Errors are mm;
negative deltas are improvements.

| Method | PA joint | Delta vs base | PA mesh | Delta vs base |
|---|---:|---:|---:|---:|
| WiLoR base | 6.695 | - | 6.564 | - |
| rope teacher, Flex15 | 6.668 | -0.027 | 6.511 | -0.053 |
| shuffled rope teacher | 9.420 | +2.725 vs base | 9.143 | +2.579 vs base |
| oracle-chain Flex15 | 6.269 | -0.426 | 6.144 | -0.420 |
| oracle-chain Pose45 | 5.264 | -1.431 | 5.195 | -1.369 |

The shuffle control proves that rope/sample pairing is real, and the action
oracles show that Flex15/Pose45 can move enough to matter. The 0.027 mm rope
teacher ceiling is nevertheless far below the 0.15 mm continuation threshold.
More ARCTIC rows would mainly distill a weak teacher more precisely. Stop an
unconditional ARCTIC teacher expansion; the missing information is the
deployment observation, not parameter count.

## HOT3D: natural visibility is already a useful stressor

| Method | PA joint all | PA joint context | PA joint low visibility | Root-relative joint low visibility |
|---|---:|---:|---:|---:|
| WiLoR base | 9.782 | 8.635 | 10.834 | 88.667 |
| release student delta | -0.701 | -0.059 | -1.289 | +1.921 |
| 400-step rope teacher delta | -1.316 | -0.747 | -1.838 | +1.328 |
| shuffled rope teacher delta | +2.225 | +2.641 | +1.843 | +1.871 |

The teacher/base low-visibility delta has paired episode CI
`[-2.616, -1.067]` mm. Therefore artificial masking is not required for the
primary HOT3D question: the official data supplies abundant natural
visibility failures, and correct rope pairing improves aligned hand shape.
Synthetic masks remain useful only as controlled causal stress tests.

The root-relative column prevents an overclaim. Release and teacher improve
PA shape while worsening camera-frame/root-relative articulation. Flex15
cannot repair global wrist orientation, camera translation, or scale.

### Why the first HOT3D oracle looked inverted

The 400-step `oracle_chain/Flex15` lowers its actual wrist-relative objective
by 9.891 mm overall and 11.448 mm on low visibility, but worsens PA joint by
6.848 mm. Lower learning rates approach the base continuously; Pose45 at
lr=1 lowers root-relative error by 15.829 mm while worsening PA by 4.586 mm.
This is not a joint-order or optimizer-sign failure. With fixed erroneous
global orientation, fitting camera-frame joints can bend the local hand into
a shape that is worse after Procrustes alignment. HOT3D results must therefore
report PA and root-relative metrics separately.

## Natural trusted-state oracle

The state uses the last context WiLoR 45D MANO hand pose, current global
orientation/betas/camera translation, current rope, and the frozen release
student. Official phase is used only as a state-update oracle.

| Method | Low-visibility PA delta vs base | Paired episode 95% CI |
|---|---:|---:|
| last context only | -0.426 | [-1.411, +0.670] |
| framewise release | -1.289 | [-2.175, -0.413] |
| 400-step teacher | -1.838 | [-2.616, -1.067] |
| last context + release K1 | **-2.022** | **[-3.020, -1.057]** |
| state K1, rope shuffled within episode | -1.783 | [-2.739, -0.837] |

State K1 beats framewise release with paired CI corresponding to
`[0.061, 1.378]` mm additional improvement. Correct rope pairing beats the
state shuffle by 0.239 mm, CI `[0.092, 0.452]`. State alone is not sufficient;
the result needs both trusted visual state and current paired rope.

## Preliminary participant-disjoint learned image gate

Frozen WiLoR mean-pooled features are 1,280D. A linear classifier is fit on
P0001/P0002/P0003/P0009/P0010/P0011/P0012 and evaluated without retraining on
P0014/P0015.

| Gate | Held-out AUROC | Balanced accuracy | Context false freeze | Low-vis missed freeze |
|---|---:|---:|---:|---:|
| linear | **0.988** | **95.9%** | **2.0%** | 6.18% |
| MLP16 | 0.986 | 95.5% | 7.33% | **1.69%** |

The linear gate is preferred: it is smaller, has better AUROC/balanced
accuracy, and false state freezes are the more dangerous error.

On all low-visibility rows, learned-linear state K1 is -2.036 mm versus base,
statistically better than framewise release by paired CI `[0.114, 1.441]` mm,
and indistinguishable from the phase oracle (-2.022 mm; paired CI crosses
zero). On the 10 held-out episodes, learned linear is -1.705 mm and the phase
oracle is -1.704 mm. Their base-relative CIs cross zero because the validation
endpoint is underpowered, but correctly paired rope still beats the learned
state shuffle on held-out participants with CI `[0.077, 1.205]` mm. This
10-episode endpoint is superseded by the expanded validation below.

## Expanded held-out validation: visibility is not state usefulness

The follow-up keeps the same 35 training episodes and exhausts the strict,
unique-sequence candidates for P0014/P0015: 11 + 19 = 30 held-out episodes,
1,908 validation rows, and 1,008 low-visibility rows. No gate architecture or
training recipe changes.

| Gate | Held-out AUROC | Balanced accuracy | Context false freeze | Low-vis missed freeze |
|---|---:|---:|---:|---:|
| linear | **0.968** | **90.7%** | **6.11%** | 12.40% |
| MLP16 | 0.940 | 89.8% | 10.56% | **9.92%** |

The linear gate remains the better visibility classifier, but the downstream
endpoint reverses the small-screen result:

| Method | Held-out low-visibility PA delta vs base | 95% episode CI |
|---|---:|---:|
| framewise release | +0.471 | [-0.600, +1.545] |
| official-phase state K1 | +0.727 | [-1.045, +2.413] |
| learned-linear state K1 | +0.535 | [-1.090, +2.037] |
| learned-linear, rope shuffled | +0.750 | [-0.924, +2.268] |
| learned MLP16 state K1 | +0.523 | [-1.194, +2.127] |

Correct rope pairing still beats the learned-state shuffle by 0.215 mm, paired
CI `[0.040, 0.450]`, but it cannot overcome stale or inappropriate state.
Because even the official phase oracle is worse than base, better visibility
classification cannot fix this endpoint. The earlier 10-episode improvement
was selection-sensitive and is not the deployment conclusion.

### State-usefulness arbitration ceiling

A GT diagnostic chooses the lower-PA result per frame between framewise
release and official-phase state K1. This does not define a deployable method;
it asks whether selective state use is worth pursuing.

| Held-out low visibility | PA joint |
|---|---:|
| framewise release | 10.756 |
| phase state K1 | 11.012 |
| per-frame oracle minimum | **9.863** |

State is better on 45.3% of frames. The oracle minimum improves release by
0.893 mm with episode-bootstrap CI `[-1.490, -0.438]`. The split is
participant-dependent: phase state is 0.192 mm better on P0014 but 0.581 mm
worse on P0015. State-better fraction is 50.7%, 47.7%, 37.7%, and 48.8% over
age bins 1-5, 6-15, 16-30, and 31-60, so a fixed lifetime threshold does not
explain the natural result.

The next gate is therefore not an occlusion detector. It must arbitrate
whether the stored pose is useful relative to the current prediction using
observable disagreement/motion evidence. The 0.893 mm oracle ceiling is large
enough for one minimal arbitration screen, but it does not justify a larger
visibility network.

### Minimal observable state-usefulness screen

The frozen 7/2 participant split is reused. A linear classifier receives only
five deployable scalars: state age, frozen visibility score, PA disagreement
between current/state candidates, current-to-last-context pooled-image cosine
distance, and state-minus-current rope residual. No raw GT or participant id
is an input.

| Arbitration | Held-out AUC | Hybrid minus release | 95% episode CI |
|---|---:|---:|---:|
| age + visibility | 0.552 | +0.199 mm | [-0.743, +1.124] |
| five observable scalars | 0.627 | -0.062 mm | [-0.879, +0.742] |
| shuffled train labels | 0.499 | +0.256 mm | [-0.700, +1.188] |

The five-scalar gate recovers only about 7% of the 0.893 mm oracle ceiling,
does not pass the 0.15 mm continuation threshold, and has a CI spanning harm
and benefit. Stop simple state arbitration here; adding an MLP or more ad-hoc
scalars is not supported.

## EgoDex-expanded students on natural HOT3D and ARCTIC

The already-frozen checkpoints from 0073 are applied without retraining to the
65-episode HOT3D screen and the 3,888-hand ARCTIC stride-10 screen. This
directly tests whether more supervision can help the same small Flex15 module.

### HOT3D

| Student | All low-vis PA delta vs base | Held-out low-vis delta | Held-out delta vs release |
|---|---:|---:|---:|
| four-teacher release | -0.503 | +0.471 | - |
| EgoDex-only, 111 tasks | **-1.224** | **-0.679** | **-1.150** |
| four teachers + full EgoDex | -1.190 | -0.557 | -1.028 |
| four teachers + capped EgoDex | -1.022 | -0.394 | -0.865 |

Every EgoDex-expanded checkpoint beats release significantly on held-out
P0014/P0015: paired CIs are `[0.762, 1.595]`, `[0.703, 1.411]`, and
`[0.595, 1.190]` mm in release-minus-candidate order. Full EgoDex also beats
the capped mixture, CI `[0.040, 0.283]` mm. EgoDex-only and the full mixture
are statistically tied on held-out HOT3D. This is PA-shape transfer; all still
worsen root-relative error, although less than release.

### ARCTIC

| Student | PA joint | Delta vs base | Delta vs release | Episode-bootstrap CI vs base |
|---|---:|---:|---:|---:|
| WiLoR base | 6.695 | - | - | - |
| four-teacher release | 6.700 | +0.004 | - | [-0.102, +0.104] |
| EgoDex-only, 111 tasks | **6.504** | **-0.191** | -0.196 | **[-0.276, -0.112]** |
| four teachers + full EgoDex | 6.573 | -0.122 | -0.127 | **[-0.218, -0.033]** |
| four teachers + capped EgoDex | 6.657 | -0.038 | -0.043 | [-0.142, +0.059] |

EgoDex-only and full EgoDex both improve ARCTIC significantly; capped EgoDex
does not beat base significantly. The full-versus-capped ordering therefore
agrees across HOT3D, ARCTIC, and the 0073 matched matrix: more diverse EgoDex
teacher rows help this fixed small module when their signal transfers.

EgoDex-only is best on ARCTIC/HOT3D but remains weaker than the full mixture on
trusted FreiHAND/HO3D masked protocols. It is evidence for domain diversity,
not a universal release replacement. The full EgoDex mixture remains the
balanced experimental candidate.

## Decisions

1. **HOT3D masking: no for the primary branch.** Use official natural
   visibility drops. Keep artificial masks only for controlled ablations.
2. **Visibility gate: stop as the deployment decision.** Linear image features
   recognize natural visibility, but official phase itself does not determine
   when stored state helps. A deeper visibility classifier is not justified.
3. **Module size/data: data helps selectively, not indiscriminately.** ARCTIC's
   per-sample rope teacher is too weak to justify ARCTIC pseudo-label scaling,
   but diverse EgoDex teacher data improves the same Flex15 student on both
   ARCTIC and HOT3D; full EgoDex consistently beats capped EgoDex.
4. **Metric claim: split PA shape from root-relative/camera-frame accuracy.**
   Current RopeTrack corrects articulation shape but does not solve global
   wrist/camera errors.
5. **State line: stop.** The low-dimensional usefulness gate recovers only 7%
   of its oracle ceiling. Do not enlarge the visibility/arbitration gate.
6. **Checkpoint decision:** keep four-teacher release as the formal release and
   full-EgoDex mixture as the balanced experimental candidate. EgoDex-only is
   a useful natural-domain ablation, not a universal replacement.

## Reproducibility

- Corrected code commits: `d1dd1e6`, `0adbfe2`, `4d23c03`, `1976030`,
  `fc13f8f`, `9cf2299`, `b7ab937`.
- HOT3D run root:
  `/data/wentao/ropetrack/runs/hot3d_natural_screen_20260718`.
- Corrected processed screen:
  `/data/wentao/ropetrack/processed/hot3d/natural_screen_45ep_v2_20260718`.
- Main jobs: audit/select `185270`, export `185271`, merge `185272`, base
  `185273`, refine array `185274`, score `185275`.
- ARCTIC jobs: teacher/score `185237`/`185238`, oracle array/score
  `185266`/`185267`.
- HOT3D oracle-lr jobs: `185290`, corrected score `185303`, root-relative
  rescore `185314`.
- State oracle jobs: successful state/score `185315`/`185317`.
- Learned gate jobs: feature+gate/state `185322`, score `185323`.
- Expanded validation selection `185331`; export/merge/base/release/phase/gate/
  score chain `185338`-`185344`; arbitration ceiling `185355`.
- Expanded processed screen:
  `/data/wentao/ropetrack/processed/hot3d/natural_screen_65ep_v1_20260718`.
- State-usefulness probe `185369`; output
  `hot3d_combined_gate_v1/state_usefulness_probe.json`.
- HOT3D EgoDex-student apply/score `185370`/`185371`.
- ARCTIC EgoDex-student apply/score/paired-bootstrap `185397`/`185398`/`185402`.
- Full local suite after the gate implementation: 389 tests passed.

Failed v1 outputs are not result cells. In particular, selector jobs
`185240`/`185241`, merge `185246`, state attempts `185301`/`185312`, and the
first score environment `185291` led to the strict protocol/checkpoint/env
fixes above and must not be compared numerically.
