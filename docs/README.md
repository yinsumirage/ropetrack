# Documentation

`docs/` is intentionally small. It contains only current reusable documents,
the retained P0-P2 report, its figures, and advisor-facing HTML snapshots.

| Document | Purpose |
|---|---|
| [current-code-and-artifact-map.md](current-code-and-artifact-map.md) | Current research, code, artifact, branch, and stop/continue boundary |
| [dataset-contract-matrix.md](dataset-contract-matrix.md) | Cross-dataset units, frames, handedness, joint order, validation gates, and metric-grade boundary |
| [interhand26m-oneview-protocol.md](interhand26m-oneview-protocol.md) | Reusable InterHand one-view protocol |
| [2026-07-08-progress-report.md](2026-07-08-progress-report.md) | Retained detailed P0-P2 report |
| [report/](report/) | Ignored advisor-facing HTML snapshots, including 2026-07-08 and 2026-07-22 |
| [../RELEASE.md](../RELEASE.md) | Frozen P2 release identity and reproduction path |
| [../experience/INDEX.md](../experience/INDEX.md) | Chronological experiment truth and topic index |

## Placement

- Put reusable current protocols or a single current status map here.
- Put dated plans, jobs, failures, measurements, and terminal decisions in
  `experience/`; do not create another dated planning series in `docs/`.
- Put generated HTML in `docs/report/`, named
  `YYYY-MM-DD-progress-report.html`.
- Keep report figures only while a retained report references them.
