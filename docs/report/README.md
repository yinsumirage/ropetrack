# Local Advisor Reports

This folder contains generated, advisor-facing HTML snapshots. Use the filename
`YYYY-MM-DD-progress-report.html`; keep one file per reporting date.

- `2026-07-08-progress-report.html`: earlier P0-P2 progress report.
- `2026-07-22-progress-report.html`: current DirectPose/data-coverage report.

Keep durable source material in the parent `docs/` folder and chronological
experiment truth in `../../experience/`. HTML files are ignored by git and may be regenerated. Do not
create a separate root `output/` folder for reports. Export a PDF only when it
is explicitly needed, and treat it as a temporary derivative rather than a
second source of truth.
