# 0022 Rope Phase 1/2 HPC Run

Date: 2026-07-05

## Purpose

Run the new rope label generator and prediction diagnostic scorer on the HPC CPU
partition against the existing clean and hard baseline outputs.

## Commands And Outputs

- Synced the outer repo rope files to `~/project/ropetrack`; `third_party/` was
  not touched.
- Remote smoke check passed:
  `python -m py_compile ropetrack/rope.py scripts/make_rope_labels.py scripts/rope_diagnostics/score_rope_predictions.py`
  and `python -m unittest tests.test_rope -v`.
- Main Slurm job: `165814`, CPU partition, failed after completing labels,
  visualizations, and clean scores.
- Failure cause: the local PowerShell here-string expanded remote bash variables
  such as `${kind}` and `${backend}` before the script reached HPC, producing a
  bad path like `freihand___original`.
- Follow-up Slurm job: `165825`, CPU partition, completed hard scoring.
- Output root:
  `/data/wentao/ropetrack/runs/rope_phase12_20260705_031056`.
- Local copied artifacts:
  `.local_checks/rope_phase12_20260705_031056/`.

Label counts:

- FreiHAND: `3960`
- HO3D v2: `11524`

Visualization count:

- FreiHAND: `24` PNGs
- HO3D v2: `24` PNGs

## Rope Diagnostic Summary

| Run | Samples | rope_norm_mae | rope_dist_mae_m |
|---|---:|---:|---:|
| clean_freihand_hamer | 3960 | 0.092896 | 0.008816 |
| clean_freihand_wilor | 3960 | 0.086987 | 0.008412 |
| clean_ho3d_v2_hamer | 11524 | 0.107721 | 0.009145 |
| clean_ho3d_v2_wilor | 11524 | 0.112022 | 0.009601 |
| hard_freihand_finger_end80_hamer | 3960 | 0.161023 | 0.015934 |
| hard_freihand_finger_end80_wilor | 3960 | 0.143768 | 0.013604 |
| hard_freihand_mask70_hamer | 3960 | 0.145940 | 0.014407 |
| hard_freihand_mask70_wilor | 3960 | 0.138291 | 0.013192 |
| hard_freihand_tip_square80_hamer | 3960 | 0.116532 | 0.011716 |
| hard_freihand_tip_square80_wilor | 3960 | 0.101734 | 0.010234 |
| hard_ho3d_v2_finger_end80_hamer | 11524 | 0.117060 | 0.009887 |
| hard_ho3d_v2_finger_end80_wilor | 11524 | 0.119421 | 0.010126 |
| hard_ho3d_v2_mask70_hamer | 11524 | 0.122715 | 0.010344 |
| hard_ho3d_v2_mask70_wilor | 11524 | 0.135857 | 0.011505 |
| hard_ho3d_v2_tip_square80_hamer | 11524 | 0.110153 | 0.009310 |
| hard_ho3d_v2_tip_square80_wilor | 11524 | 0.110041 | 0.009336 |

## Readout

- The label generator works on full FreiHAND and HO3D v2 eval splits.
- The visualization hook is useful for spot checks; sampled FreiHAND and HO3D
  images were non-empty and showed the GT chain overlay plus per-finger bars.
- The diagnostic is sensitive to hard splits. FreiHAND shows a large increase
  from clean to hard, especially `finger_end80` and `mask70`.
- HO3D v2 increases are smaller but still visible, with `mask70` strongest for
  WiLoR.
- This remains a post-hoc geometric diagnostic, not training evidence. It should
  be used to verify data/adapter correctness and choose hard cases before any
  rope-conditioned training.

## Do Not Repeat

- Do not write Slurm scripts with PowerShell double-quoted here-strings when the
  script contains remote bash variables. Use a single-quoted here-string with a
  placeholder replacement, then strip CRLF on HPC with `tr -d '\r'`.
