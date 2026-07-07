# 0042 P2 Student Results And Report Tables

Date: 2026-07-07

## Scope

This records the first P2 alpha-student results and the report-table/figure
artifacts generated after jobs `169878`-`169887` completed. The HO3D v3 stride4
train teacher job `169973` was still running at the time of this note and is
not included in the conclusions below.

## Aggregation Artifacts

Generated with:

```bash
python scripts/rope_refiner/summarize_runs.py ...
python scripts/rope_refiner/plot_report_figures.py ...
```

Output tables:

```text
/data/wentao/ropetrack/runs/rope_p2_student_20260707_135405/tables_final/runs_summary.tsv
/data/wentao/ropetrack/runs/rope_p2_student_20260707_135405/tables_final/runs_summary.md
/data/wentao/ropetrack/runs/rope_p2_student_20260707_135405/tables_final/runs_summary.json

/data/wentao/ropetrack/runs/rope_p2_queue_20260707_135405/tables_final/runs_summary.tsv
/data/wentao/ropetrack/runs/rope_p2_queue_20260707_135405/tables_final/runs_summary.md
/data/wentao/ropetrack/runs/rope_p2_queue_20260707_135405/tables_final/runs_summary.json
```

Report figure bundle:

```text
/data/wentao/ropetrack/runs/rope_report_tables_20260707_153000/dose/runs_summary.tsv
/data/wentao/ropetrack/runs/rope_report_tables_20260707_153000/noise/runs_summary.tsv
/data/wentao/ropetrack/runs/rope_report_tables_20260707_153000/dose_response.png
/data/wentao/ropetrack/runs/rope_report_tables_20260707_153000/noise_curve.png
```

The two report scripts came from commit:

```text
4d84d92 Add alpha student distillation tools
```

## P2 Student Main Results

FreiHAND mask70 baseline in the sliced scorer:

```text
all_joints_base_cm = 1.006775
```

| Cell | PA cm | All-joint delta cm | Occluded-tip delta cm | Clean-tip delta cm | Closure | Alpha mean abs |
|---|---:|---:|---:|---:|---:|---:|
| `mask70_main` | 0.843164 | -0.163610 | -0.526479 | -0.231194 | 0.437833 | 0.067341 |
| `mask70_noaug` | 0.835164 | -0.171611 | -0.556241 | -0.245343 | 0.476529 | 0.073074 |
| `mask70_seed1` | 0.843632 | -0.163143 | -0.525292 | -0.229662 | 0.434206 | 0.067013 |
| `mask70_seed2` | 0.843814 | -0.162961 | -0.529463 | -0.227947 | 0.437877 | 0.068152 |
| `mask70_shuffled` | 1.001978 | -0.004797 | -0.010340 | -0.007890 | 0.007089 | 0.001599 |
| `mask70_main_noise0p05` | 0.854106 | -0.152669 | -0.497773 | -0.206652 | 0.433108 | 0.068257 |

Interpretation:

- The main student recovers almost all of the strong teacher's mask70 gain:
  teacher PA was 0.838920 cm in `0036`, student main is 0.843164 cm.
- The recovery criterion was >=80% of teacher improvement; student main is
  roughly 97% by all-joint delta.
- The shuffled-rope control collapses to near-zero improvement, so the gain is
  actually coming from rope conditioning.
- Seeds are stable: seed0/1/2 are 0.8432/0.8436/0.8438 cm.
- No-augmentation is slightly better than the default augmented student on
  mask70 eval and slightly exceeds the teacher delta. A likely explanation is
  that the student learns a smooth average mapping from noisy per-sample
  teacher targets. The deployment choice still needs the `noaug@noise0.05`
  evaluation cell before replacing the augmented default.
- With rope input noise std 0.05, the student keeps most of the gain
  (-0.152669 cm all-joint, -0.497773 cm occluded-tip).
- At noise std 0.05, the augmented student also beats the noisy teacher result
  from `0038` (-0.152669 cm vs -0.1384 cm all-joint), supporting the intended
  distillation/augmentation denoising effect.

Generalization cells:

| Cell | PA cm | All-joint delta cm | Occluded-tip delta cm | Closure | Note |
|---|---:|---:|---:|---:|---|
| `finger_end80_main` | 0.837320 | -0.160338 | -0.328725 | 0.454829 | Cross-disturbance transfer works. |
| `ho3d_wilor_main` | 0.852483 | -0.091140 | -0.222336 | 0.407448 | Cross-dataset transfer is positive but smaller. |
| `hamer_mask70_main` | 0.916856 | -0.165497 | -0.545985 | 0.449297 | Cross-backend eval still improves. |
| `clean_main` | 0.497948 | - | - | 0.283306 | Clean split is effectively neutral: previous WiLoR clean baseline was about 0.4956 cm, so the student changes clean PA by only about +0.023 mm. |

## Queue Results

Additional train teachers:

| Cell | Samples | Closure | Alpha mean abs | Gated fingers |
|---|---:|---:|---:|---:|
| `q1_finger_end80_train/teacher` | 32560 | 0.524968 | 0.062851 | 0.350338 |
| `q2_hamer_mask70_train/teacher` | 32560 | 0.508044 | 0.051295 | 0.289982 |

Both teacher exports completed with 32,560 samples and 0 failures before
teacher optimization.

Noise curve supplement:

| Noise std | PA cm | All-joint delta cm | Occluded-tip delta cm | Clean-tip delta cm |
|---:|---:|---:|---:|---:|
| 0.025 | 0.847258 | -0.159517 | -0.533864 | -0.215503 |
| 0.075 | 0.898097 | -0.108678 | -0.436394 | -0.095023 |
| 0.150 | 0.987534 | -0.019240 | -0.245758 | 0.098768 |

This fills the gap between the earlier 0.05/0.1/0.2 points: moderate noise
still leaves useful signal, while 0.15 normalized noise is close to the
all-joint usefulness boundary and starts harming clean fingertips.

HO3D strong oracle:

| Backend | PA cm | All-joint delta cm | Occluded-tip delta cm | Clean-tip delta cm |
|---|---:|---:|---:|---:|
| WiLoR | 0.827601 | -0.116023 | -0.469764 | -0.308959 |
| HaMeR | 0.803475 | -0.128727 | -0.468296 | -0.336113 |

This confirms that HO3D has oracle headroom, not just FreiHAND. The correct
"remaining headroom" is the oracle-minus-rope gap, not the oracle total gain:
HO3D WiLoR has only about 0.11 mm left after the rope winner, while HaMeR has
about 0.63 mm left.

## Remaining Work

At the time this note was first written, the only still-running job in this set
was:

```text
169973 h3v3_s4_teach RUNNING
```

Once it completes, combine the HO3D v3 stride4 train teacher with the FreiHAND
train teacher for the multi-dataset student experiment.
