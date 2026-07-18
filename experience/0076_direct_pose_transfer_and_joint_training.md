# Direct-pose transfer and ARCTIC+HOT3D joint training

## Question

The RGB+rope direct-GT result in 0075 used ARCTIC P2 train only. This
follow-up asks whether that checkpoint transfers beyond ARCTIC, whether the
small frozen-backbone head is already limited by WiLoR capacity, and whether
the next experiment should be data expansion or backbone fine-tuning.

## Protocol

- source checkpoint: ARCTIC-only C from
  `/data/wentao/ropetrack/runs/direct_pose_head_20260718/students/C_rgb_rope/model.pt`;
- transfer tests: HOT3D natural visibility, FreiHAND mask70, and HO3D v2
  mask70, with no checkpoint updates;
- joint checkpoint: the same localized 4x3 WiLoR-token RGB+rope head trained
  on 31,006 ARCTIC hands plus 2,166 HOT3D hands;
- HOT3D split: participants P0014 and P0015 are excluded from the added
  training bundle, leaving all 1,908 of their hands for held-out evaluation;
- no HOT3D repetition or source-balanced oversampling is used;
- WiLoR remains frozen, and all losses and head hyperparameters remain
  unchanged from 0075.

The joint-data support is commit `891dcfc`. The exact remote worktree was
`~/project/ropetrack_direct_pose_joint_891dcfc`; it used explicit symlinks to
the populated WiLoR checkout, shared checkpoints, and MANO assets. Run
artifacts are under
`/data/wentao/ropetrack/runs/direct_pose_transfer_20260718`.

## ARCTIC-only transfer

All values below are PA joint error in mm. Negative deltas are improvements.

| Target | WiLoR base | ARCTIC-only C | Delta | Control |
|---|---:|---:|---:|---:|
| ARCTIC P2-val | 6.695 | **5.944** | **-0.751** | inference shuffle 10.481 |
| FreiHAND mask70 | 10.485 | **8.109** | **-2.375** | inference shuffle 14.009 |
| HO3D v2 mask70 | 9.436 | **8.181** | **-1.255** | inference shuffle 10.387 |
| HOT3D natural, all | 9.280 | 9.173 | -0.108 | inference shuffle 13.264 |
| HOT3D natural, low visibility | 10.332 | 9.703 | -0.629 | inference shuffle 13.710 |
| HOT3D P0014/P0015, low visibility | 10.285 | 10.374 | **+0.090** | inference shuffle 14.399 |

The HOT3D all-frame delta has paired-episode 95% CI
`[-0.694, +0.472]`; the held-out low-visibility delta has CI
`[-0.980, +1.120]`. Therefore the ARCTIC-only checkpoint does not establish
natural HOT3D transfer. In contrast, the large FreiHAND/HO3D gains and their
shuffle collapses show that it does transfer to controlled occlusion anchors.
The issue is a natural participant/domain gap, not a completely
dataset-specific action space.

## ARCTIC+HOT3D joint result

The joint head stopped after epoch 26, with the retained checkpoint selected
at epoch 18 (internal PA 4.104 mm under the configured minimum-improvement
threshold). It used 29,770 training and 3,402 internal-validation hands after
the episode-disjoint split.

| Test | WiLoR base | ARCTIC-only C | Joint C | Joint delta vs base | Joint vs ARCTIC-only |
|---|---:|---:|---:|---:|---:|
| ARCTIC P2-val | 6.695 | **5.944** | 5.994 | -0.702 | +0.049 |
| HOT3D natural, all | 9.280 | 9.173 | **6.231** | **-3.049** | -2.942 |
| HOT3D P0014/P0015, all | 9.255 | 9.904 | **7.882** | **-1.372** | -2.022 |
| HOT3D P0014/P0015, low visibility | 10.285 | 10.374 | **8.215** | **-2.070** | -2.159 |
| FreiHAND mask70 | 10.485 | **8.109** | 8.254 | -2.231 | +0.144 |
| HO3D v2 mask70 | 9.436 | 8.181 | **8.156** | -1.280 | -0.025 |

The joint model retains 93.4% of the ARCTIC-only PA gain. On HOT3D, its
all-frame paired-episode CI versus base is `[-3.729, -2.362]`, and its
held-out low-visibility CI is `[-3.097, -1.019]`. Shuffling rope at inference
raises all-frame HOT3D PA to 12.313 mm and held-out low-visibility PA to
13.852 mm, so the gain is not an image-only participant memorization result.

The aligned-shape claim must remain separate from global pose. HOT3D
camera-frame joint error changes from 119.709 to 120.211 mm, while root-relative
joint error changes from 69.131 to 68.564 mm. The supported finding is a large
PA hand-shape improvement with a small root-relative gain, not solved global
wrist/camera articulation.

## Failure and monitoring record

Earlier failures were caused by an isolated worktree containing an empty
WiLoR submodule directory, not by Slurm or the direct-pose code. This run used
the exact-commit worktree and explicit dependency symlinks. Jobs `185810`
(bundle), `185811` (train), `185812` (apply), `185813` (score), `185820`
(anchor apply), and `185821` (anchor score) all completed with exit code 0.
The final CPU score was monitored through completion; it used about 60 GB and
completed both the 3,960-sample FreiHAND and 11,524-sample HO3D scores.

## Decision

**Continue trusted multi-dataset training before tuning WiLoR.** A small
amount of participant-separated HOT3D data closes the natural-domain gap and
preserves ARCTIC, FreiHAND, and HO3D. This is direct evidence that the current
limitation was data coverage rather than insufficient frozen-backbone
capacity.

LoRA is valid and should not be ruled out. The clean later capacity ablation
is: frozen head versus rank-8 Q/V LoRA on the last two WiLoR transformer
blocks (about 82k trainable parameters) versus one fully trainable last block
(about 19.7M parameters), under identical data, loss, step budget, and
evaluation. The full last block is an upper-capacity control, not the assumed
best method. Do not build that online-image training path before expanding a
participant-balanced trusted HOT3D subset: the frozen result has not yet
exposed a backbone-capacity ceiling.

EgoDex should not be appended as equivalent MANO ground truth. Its native
labels remain ARKit joints, so any use in this direct-pose line must be a
separate confidence-weighted or joint-only supervision cell rather than the
same trusted MANO bundle.
