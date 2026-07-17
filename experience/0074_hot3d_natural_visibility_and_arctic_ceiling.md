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

## Participant-disjoint learned image gate

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
state shuffle on held-out participants with CI `[0.077, 1.205]` mm.

## Decisions

1. **HOT3D masking: no for the primary branch.** Use official natural
   visibility drops. Keep artificial masks only for controlled ablations.
2. **Gate: continue the linear image gate.** Natural participant-disjoint
   classification is already strong; a deeper gate is not justified.
3. **Module size/data: do not scale indiscriminately.** ARCTIC proves an
   observation gap, while HOT3D proves a natural domain where rope/state has a
   real ceiling. Dataset volume is useful only after a per-domain teacher or
   state upper bound passes.
4. **Metric claim: split PA shape from root-relative/camera-frame accuracy.**
   Current RopeTrack corrects articulation shape but does not solve global
   wrist/camera errors.
5. **Next smallest experiment:** expand only the held-out P0014/P0015 natural
   episodes to tighten the learned-gate endpoint CI. Do not enlarge the gate or
   start another broad student mixture first.

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
- Full local suite after the gate implementation: 389 tests passed.

Failed v1 outputs are not result cells. In particular, selector jobs
`185240`/`185241`, merge `185246`, state attempts `185301`/`185312`, and the
first score environment `185291` led to the strict protocol/checkpoint/env
fixes above and must not be compared numerically.
