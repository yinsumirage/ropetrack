# EgoDex public derivatives and training value

Date: 2026-07-17

## Question

Does the negative zero-shot RopeTrack MLP result mean EgoDex is unsuitable for
training, and is there a public cleaned or MANO-converted EgoDex release?

## Findings

- Raw EgoDex provides ARKit skeletal transforms and confidence, not native MANO
  ground truth. It is therefore not a drop-in source of trustworthy mesh/MANO
  supervision.
- Being-H0 standardizes datasets to MANO. Its paper says sources with only 3D
  joints are fit to MANO by gradient optimization with joint-angle constraints
  and temporal smoothness. This is derived supervision, not native GT.
- BeingBeyond now publicly hosts
  `BeingBeyond/UniHand_Preview/human_data/EgoDex_lerobot_v3.tar` (52,179,435,520
  bytes). The documented frame schema includes MANO `theta`, `beta`, wrist
  rotation, translation, aligned video, and validity horizons.
- The public package is described only as a preview/subset of the Being-H0.5
  pretraining mixture. The released documentation does not provide an
  EgoDex-specific confidence threshold, MANO fitting residual distribution,
  manual acceptance rate, overlap/coverage manifest against Apple file ids, or
  calibrated camera intrinsics. It explicitly supplies pseudo intrinsics for
  rendering. Treat it as public MANO pseudo-labels, not corrected benchmark GT.
- Being-H0.5 reports motion-quality filtering for HaWoR-derived in-the-wild
  estimates using detection confidence, DBA error, jitter, and discontinuity.
  The public docs do not establish that the same thresholds validate the
  joint-fitted EgoDex subset.
- H-RDT publishes an EgoDex processing/training path, but its 48D action is a
  direct concatenation of two wrist poses and ten fingertip positions. Its
  preprocessing does not consume confidence, fit MANO, or remove bad frames.
  This is evidence that EgoDex can be useful for trajectory/behavioral-prior
  pretraining without being high-precision MANO supervision.

## Interpretation for RopeTrack

The current negative result is zero-shot transfer: a student trained on other
datasets was applied to a new ARKit label/domain distribution. It is not an
EgoDex fine-tuning experiment and does not show that EgoDex training must fail.

Naive equal-weight fine-tuning against every raw EgoDex frame is nevertheless a
bad experiment. Low-confidence failures and the difference between ARKit joints
and the MANO manifold can teach the refiner to move good WiLoR predictions away
from valid hands. EgoDex remains useful for image/domain features, temporal
motion, occlusion/interaction diversity, wrist/tip trajectory learning, and
confidence-aware domain adaptation.

## Decision

Keep the already-running raw part downloads. Do not launch full-data training
yet and do not use the official test labels as editable GT. First run a Part 1
pilot:

1. freeze a confidence/geometry quality manifest without changing raw files;
2. fit or import MANO only for a high-confidence subset;
3. record joint fitting residual, temporal discontinuity, anatomical validity,
   and 2D overlays;
4. train an EgoDex-specific or domain-conditioned head with identity/retention
   protection, rather than mixing all rows into the release MLP;
5. require improvement on a frozen clean EgoDex test subset without regression
   on FreiHAND and HO3D before scaling to all five parts.

The 52 GB UniHand preview is worth a small inspection/download job after its
license and coverage metadata are checked. It should not replace the raw archive
until file coverage and fit quality are measured.

## Sources

- Apple EgoDex README: <https://github.com/apple/ml-egodex>
- Being-H0 paper: <https://arxiv.org/abs/2507.15597>
- Being-H0 repository: <https://github.com/BeingBeyond/Being-H0>
- Being-H0.5 / Being-H repository: <https://github.com/BeingBeyond/Being-H>
- UniHand Preview: <https://huggingface.co/datasets/BeingBeyond/UniHand_Preview>
- UniHand human-data schema: <https://huggingface.co/datasets/BeingBeyond/UniHand_Preview/blob/main/human_data/Human_Data.md>
- H-RDT repository: <https://github.com/HongzheBi/H_RDT>
