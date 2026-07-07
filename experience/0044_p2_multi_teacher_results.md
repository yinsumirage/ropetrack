# 0044 P2 Multi-Teacher Student Results

Date: 2026-07-07

## Scope

This records the final results of the four-teacher student submitted in `0043`.
The scoring job completed successfully:

```text
170177 p2m_score COMPLETED 0:0 00:27:21
```

Run root:

```text
/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104
```

Final table:

```text
/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104/tables/runs_summary.tsv
```

Local copy:

```text
.local_checks/p2_student_multi_tables/runs_summary.tsv
```

## Results

| Cell | PA cm | All-joint delta cm | Occluded-tip delta cm | Clean-tip delta cm | Closure |
|---|---:|---:|---:|---:|---:|
| `multi_mask70` | 0.843216 | -0.163559 | -0.525394 | -0.232165 | 0.439817 |
| `multi_finger_end80` | 0.835391 | -0.162267 | -0.333826 | - | 0.461030 |
| `multi_ho3d_wilor` | 0.846472 | -0.097152 | -0.233307 | -0.118128 | 0.418239 |
| `multi_hamer_mask70` | 0.912647 | -0.169706 | -0.560789 | -0.232493 | 0.457255 |
| `multi_clean` | 0.499656 | - | - | - | 0.289598 |
| `multi_mask70_noise0p05` | 0.852865 | -0.153909 | -0.498921 | -0.210314 | 0.436627 |
| `multi_shuffled_mask70` | 1.001144 | -0.005630 | -0.008224 | -0.009189 | 0.008717 |
| `noaug_mask70_noise0p05` | 0.865367 | -0.141407 | -0.503278 | -0.170812 | 0.488540 |

## Interpretation

- Multi-teacher training does not hurt the main FreiHAND mask70 result:
  `multi_mask70` PA is 0.843216 cm, essentially identical to the single-teacher
  `mask70_main` PA of 0.843164 cm in `0042`.
- It slightly improves the hard generalization axes:
  - HO3D WiLoR improves from 0.852483 cm to 0.846472 cm.
  - HaMeR mask70 improves from 0.916856 cm to 0.912647 cm.
  - finger_end80 improves from 0.837320 cm to 0.835391 cm.
- The shuffle control still collapses to near-zero gain, so the multi-teacher
  student is still using rope rather than memorizing the base pose prior.
- The no-augmentation model is better on clean/noiseless mask70, but under
  noise std 0.05 it drops to 0.865367 cm, worse than the augmented student's
  0.852865 cm. Keep the augmented student as the deployment/report default.
- The multi-teacher clean cell is slightly worse than the single-teacher clean
  cell (0.499656 cm vs 0.497948 cm), still close to the clean baseline scale
  but weaker than the single-teacher clean-neutral story.

## Report Decision

Use the multi-teacher student as the broader generalization result, but keep the
single-teacher main student table as the cleanest proof of the P2 distillation
claim. For deployment wording, the augmented student remains the default
because it is better under rope noise than the no-augmentation model.

Qualitative panels for the multi-teacher mask70 cell were generated at:

```text
/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104/report_panels
.local_checks/p2_student_multi_report_panels
```
