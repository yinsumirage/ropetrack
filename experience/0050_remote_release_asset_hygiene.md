# 0050 Remote Release Asset Hygiene

Date: 2026-07-08

## What Changed

Created a durable release copy on the HPC data root:

```text
/data/wentao/ropetrack/releases/p2_four_teacher_student/
  manifest.json
  student.pt
  train_log.json
```

The original provenance remains:

```text
/data/wentao/ropetrack/runs/rope_p2_student_multi_20260707_174104/students/student_multi/
```

A local mirror was also pulled to:

```text
.local_checks/releases/p2_four_teacher_student/
```

`RELEASE.md` now points to the release copy and keeps the original run path as
provenance.

## Data Root Signpost

Wrote:

```text
/data/wentao/ropetrack/README.txt
```

It records that the data root is managed by `~/project/ropetrack`, that
`RELEASE.md` and `experience/INDEX.md` are the stable provenance records, and
that run roots named by `RELEASE.md` are do-not-delete until advisor sign-off.

## Disk Accounting

`du -sh /data/wentao/ropetrack/*` on 2026-07-08:

| Path | Size |
|---|---:|
| `/data/wentao/ropetrack/logs` | 4.0K |
| `/data/wentao/ropetrack/rope_labels` | 22M |
| `/data/wentao/ropetrack/debug` | 69M |
| `/data/wentao/ropetrack/features` | 180M |
| `/data/wentao/ropetrack/FreiHAND` | 9.9G |
| `/data/wentao/ropetrack/HO3D_v2_eval` | 12G |
| `/data/wentao/ropetrack/pretrained_models` | 14G |
| `/data/wentao/ropetrack/hard` | 21G |
| `/data/wentao/ropetrack/HO3D_v3` | 68G |
| `/data/wentao/ropetrack/runs` | 93G |

No data was deleted. Any run GC should wait until after advisor sign-off and
should preserve summaries, scores, sliced metrics, alpha stats, logs, and any
run roots named by `RELEASE.md`.

## Rope Label Directory Audit

Observed label files:

```text
/data/wentao/ropetrack/rope_labels/freihand/evaluation_rope.jsonl
/data/wentao/ropetrack/rope_labels/ho3d_v2/evaluation_rope.jsonl
/data/wentao/ropetrack/rope_labels/ho3d_v3/training_rope_s4.jsonl
/data/wentao/ropetrack/hard/freihand/finger_end80_wilor_training/rope_labels.jsonl
/data/wentao/ropetrack/hard/freihand/mask70_hamer_training/rope_labels.jsonl
/data/wentao/ropetrack/hard/freihand/mask70_wilor_training/rope_labels.jsonl
```

No top-level `/data/wentao/ropetrack/rope/` directory was present during this
audit. New standalone labels should go under
`/data/wentao/ropetrack/rope_labels/<dataset>/`; hard training roots may keep
their local aligned `rope_labels.jsonl`.

## Report Assets

Copied the six small report figures into git-tracked docs assets:

```text
docs/assets/rope_report/dose_response.png
docs/assets/rope_report/noise_curve.png
docs/assets/rope_report/panel_00_sample00001731.png
docs/assets/rope_report/panel_01_sample00003609.png
docs/assets/rope_report/panel_02_sample00002490.png
docs/assets/rope_report/panel_03_sample00001658.png
```

The progress report now references these repo-local paths instead of
`.local_checks`.
