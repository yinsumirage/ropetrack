# 0010 FreiHAND Eval GT Upload

Date: 2026-07-03

## Purpose

Upload the missing FreiHAND evaluation ground truth package from local Windows
and merge it into the existing HPC FreiHAND data root.

## Source And Target

Local source:

```text
E:\Downloads\FreiHAND_pub_v2_eval.zip
E:\Downloads\FreiHAND_pub_v2_eval\*.json
```

Temporary remote upload path:

```text
/data/wentao/ropetrack/FreiHAND_eval/FreiHAND_pub_v2_eval.zip
```

Final data root:

```text
/data/wentao/ropetrack/FreiHAND
```

## CPU Job

Extraction and merge ran as a CPU Slurm job, not on the login node:

```text
job: 162058
partition: cpu
account: engram
elapsed: 00:15:09
state: COMPLETED
node: server10
```

The first attempt, job `162056`, failed before doing data work because the
temporary sbatch file had a shell quoting error. The uploaded zip remained in
place and was reused by job `162058`.

## Result

The CPU job merged the evaluation subdirectories into:

```text
/data/wentao/ropetrack/FreiHAND/evaluation
```

Observed counts after merge:

```text
rgb 3960
anno 3960
facemap 3960
```

The uploaded zip was removed after successful merge. The temporary
`/data/wentao/ropetrack/FreiHAND_eval` workspace was later removed.

The root evaluation JSON files were then copied directly from the local
extracted folder to `/data/wentao/ropetrack/FreiHAND/`, avoiding a second full
zip upload/extract:

```text
evaluation_errors.json 3960
evaluation_K.json 3960
evaluation_mano.json 3960
evaluation_scale.json 3960
evaluation_verts.json 3960
evaluation_xyz.json 3960
```

Sample annotation:

```text
/data/wentao/ropetrack/FreiHAND/evaluation/anno/00000000.json
```

It contains per-sample fields including `K`, `scale`, and `verts`. The next
FreiHAND benchmark path can use the root `evaluation_xyz.json` and
`evaluation_verts.json` for evaluator inputs, and `evaluation/anno` for any
per-sample metadata checks.
