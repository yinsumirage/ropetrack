# 0030 P0 WiLoR FreiHAND Rope Refinement Jobs

Date: 2026-07-07

## Purpose

Start the P0 diagnostic matrix from `experience/0029` on WiLoR + FreiHAND hard
splits:

- splits: `mask70`, `finger_end80`;
- objectives: `rope`, `oracle_tip`;
- action spaces: `mult5`, `mult15`, `flex15`;
- per-cell outputs include standard scores, sliced scores, rope residuals, and
  alpha dead-zone analysis.

## Run Root

```text
/data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932
```

Job manifest:

```text
/data/wentao/ropetrack/runs/rope_p0_wilor_freihand_20260707_014932/jobs.tsv
```

## Jobs

`mask70` needed a fresh WiLoR export with `--save-mano-cache` because the older
hard-original export did not contain `mano_cache.npz`.

| Kind | Split | Objective | Action | Job | Dependency |
|---|---|---|---|---:|---:|
| export | mask70 | - | - | 168852 | - |
| apply | mask70 | rope | mult5 | 168853 | 168852 |
| score | mask70 | rope | mult5 | 168854 | 168853 |
| apply | mask70 | rope | mult15 | 168855 | 168852 |
| score | mask70 | rope | mult15 | 168856 | 168855 |
| apply | mask70 | rope | flex15 | 168857 | 168852 |
| score | mask70 | rope | flex15 | 168858 | 168857 |
| apply | mask70 | oracle_tip | mult5 | 168859 | 168852 |
| score | mask70 | oracle_tip | mult5 | 168860 | 168859 |
| apply | mask70 | oracle_tip | mult15 | 168861 | 168852 |
| score | mask70 | oracle_tip | mult15 | 168862 | 168861 |
| apply | mask70 | oracle_tip | flex15 | 168863 | 168852 |
| score | mask70 | oracle_tip | flex15 | 168864 | 168863 |
| apply | finger_end80 | rope | mult5 | 168865 | - |
| score | finger_end80 | rope | mult5 | 168866 | 168865 |
| apply | finger_end80 | rope | mult15 | 168867 | - |
| score | finger_end80 | rope | mult15 | 168868 | 168867 |
| apply | finger_end80 | rope | flex15 | 168869 | - |
| score | finger_end80 | rope | flex15 | 168870 | 168869 |
| apply | finger_end80 | oracle_tip | mult5 | 168871 | - |
| score | finger_end80 | oracle_tip | mult5 | 168872 | 168871 |
| apply | finger_end80 | oracle_tip | mult15 | 168873 | - |
| score | finger_end80 | oracle_tip | mult15 | 168874 | 168873 |
| apply | finger_end80 | oracle_tip | flex15 | 168875 | - |
| score | finger_end80 | oracle_tip | flex15 | 168876 | 168875 |

## Initial Queue Check

Submission succeeded. `squeue` showed:

- `finger_end80` GPU jobs pending on priority;
- `mask70` GPU jobs pending on dependency `168852`;
- CPU score jobs pending on their apply-job dependencies.

## Next

After jobs finish, collect:

- `scores/scores.json`;
- `sliced/sliced_scores.json`;
- `deadzone/alpha_deadzone.json`;
- `summary.json` rope residual closure.
