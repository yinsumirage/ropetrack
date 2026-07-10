# Dense history utility on mixed HO3D v3 episodes

Date: 2026-07-11

## Question

Can a causal temporal refiner use clean frames before an occlusion to improve
later masked frames, rather than merely fitting a collection of all-hard or
all-clean images?

## Protocol

- Code: `1524f98e73e637aee5d34a788daf2549da118f25`
- WiLoR: `fcb911312a38fa8badd30d9656a167485d61b8f9`
- Run root: `/data/wentao/ropetrack/runs/dense_history_v1_20260711`
- HO3D v3 episodes: 30 clean context frames, 60 mask70 frames, then 30 clean
  recovery frames. Phase labels are used only for scoring, never as inputs.
- Training: 83,325 dense frames, 321 segments, 575 complete episodes.
- Evaluation: 20,137 dense frames, 70 segments, 153 complete episodes.
- Comparison: framewise student; dense same-cadence K1; strict-past K16 and
  K96 residual GRUs; K96 with history disabled; shuffled-history K96.
- Selection used sequence-disjoint validation. Test scoring used 2,000 paired
  sequence bootstrap samples.

Jobs `176429` through `176436` all completed successfully. The chain covered
the phase gate, dense teacher, framewise/K1 training, K96 memory smoke, four
K16/K96 grid cells, shuffled-history control, test apply, and CPU scoring.

## Phase gate

The requested mixed setting is real and non-trivial:

| Phase | Base PA-MPJPE (mm) |
|---|---:|
| clean context | 7.2066 |
| mask70 | 9.9240 |
| clean recovery | 7.1424 |

Masking worsens the base prediction by 2.7175 mm versus context and 2.7817 mm
versus recovery. This is not an all-hard split.

## Validation selection

| Model | Best validation L1 |
|---|---:|
| dense K1 | 0.00339803 |
| K16, lr 3e-4 | 0.00337988 |
| K16, lr 1e-3 | 0.00338496 |
| K96, lr 3e-4 | 0.00337385 |
| K96, lr 1e-3 | 0.00337295 |
| shuffled K96, lr 1e-3 | 0.00338550 |

K96's validation gain over K1 was only `2.51e-5`. Shuffling history retained
`1.25e-5`, essentially half of that tiny gain, so the pre-registered validation
promotion gate failed.

## Test result

| Model | Masked PA-MPJPE | Masked occluded-tip PA-MPJPE | Jitter |
|---|---:|---:|---:|
| dense K1 | 8.76223 mm | 12.58081 mm | 4674.98335 mm/s² |
| K16 | 8.76459 mm | 12.58636 mm | 4675.11529 mm/s² |
| K96 | 8.76929 mm | 12.60194 mm | 4675.30073 mm/s² |
| shuffled K96 | 8.76231 mm | 12.58093 mm | 4675.00167 mm/s² |

Against matched K1, K96 is worse by 0.00706 mm on masked PA-MPJPE. The paired
bootstrap interval for `K96 - K1` is `[0.00401, 0.01037]` mm, excluding zero in
the wrong direction. It is also worse by 0.02112 mm on masked occluded tips,
0.45273 mm/s² on acceleration error, and 0.31738 mm/s² on prediction jitter.
The K96 no-history output is byte-identical to K1, confirming the comparison
isolates the history branch.

The framewise and K1 correction modules still improve masked PA-MPJPE over the
uncorrected base (9.92404 -> 9.13892 -> 8.76223 mm), but dense past history adds
no useful correction beyond the current-frame K1 model.

## Decision

Do not promote K16/K96, do not run extra seeds, and do not add a larger
GRU/Transformer on the same 5-rope plus pose input. The negative result is
statistically resolved and the shuffled control shows that the small validation
movement is not reliable history use.

If temporal work resumes, first add a new observable signal: localized visual
tokens plus an explicit visibility/quality estimate, or a controlled video
protocol with persistent subject/object state. Repeating sensor-only history
length or capacity sweeps is not justified by this run.
