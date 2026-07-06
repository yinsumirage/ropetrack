# 0031 P0 HO3D v2 Mask70 Rope Refinement Jobs

Date: 2026-07-07

## Purpose

Re-run HO3D v2 `mask70` rope refinement with the fixed optimizer joint-chain
logic from `experience/0029`.

This replaces the old HO3D optimize row in `experience/0028`, which used the
wrong chain indexing for the WiLoR MANO wrapper's `out.joints`.

## Run Root

```text
/data/wentao/ropetrack/runs/rope_p0_ho3d_v2_mask70_20260707_021218
```

Job manifest:

```text
/data/wentao/ropetrack/runs/rope_p0_ho3d_v2_mask70_20260707_021218/jobs.tsv
```

## Inputs

- Hard root: `/data/wentao/ropetrack/hard/ho3d_v2/mask70`
- Rope labels: `/data/wentao/ropetrack/runs/rope_phase12_20260705_031056/labels/ho3d_v2_rope.jsonl`
- Existing exports reused:
  `/data/wentao/ropetrack/runs/rope_optimization_ho3d_v2_mask70_20260705/{wilor,hamer}/export`

## Jobs

| Kind | Backend | Objective | Action | Job | Dependency |
|---|---|---|---|---:|---:|
| apply | WiLoR | rope | mult5 | 168940 | - |
| score | WiLoR | rope | mult5 | 168941 | 168940 |
| apply | WiLoR | rope | mult15 | 168942 | - |
| score | WiLoR | rope | mult15 | 168943 | 168942 |
| apply | WiLoR | rope | flex15 | 168944 | - |
| score | WiLoR | rope | flex15 | 168945 | 168944 |
| apply | WiLoR | oracle_tip | mult5 | 168946 | - |
| score | WiLoR | oracle_tip | mult5 | 168947 | 168946 |
| apply | WiLoR | oracle_tip | mult15 | 168948 | - |
| score | WiLoR | oracle_tip | mult15 | 168949 | 168948 |
| apply | WiLoR | oracle_tip | flex15 | 168950 | - |
| score | WiLoR | oracle_tip | flex15 | 168951 | 168950 |
| apply | HaMeR | rope | mult5 | 168952 | - |
| score | HaMeR | rope | mult5 | 168953 | 168952 |
| apply | HaMeR | rope | mult15 | 168954 | - |
| score | HaMeR | rope | mult15 | 168955 | 168954 |
| apply | HaMeR | rope | flex15 | 168956 | - |
| score | HaMeR | rope | flex15 | 168957 | 168956 |
| apply | HaMeR | oracle_tip | mult5 | 168958 | - |
| score | HaMeR | oracle_tip | mult5 | 168959 | 168958 |
| apply | HaMeR | oracle_tip | mult15 | 168960 | - |
| score | HaMeR | oracle_tip | mult15 | 168961 | 168960 |
| apply | HaMeR | oracle_tip | flex15 | 168962 | - |
| score | HaMeR | oracle_tip | flex15 | 168963 | 168962 |

## Initial Queue Check

Submission succeeded. `squeue` showed all GPU apply jobs pending on priority,
and all CPU score jobs pending on their apply-job dependencies.
