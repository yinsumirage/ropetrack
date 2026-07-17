# HOT3D v4 Shared-Copy Integrity Audit

Date: 2026-07-17

## Scope

Read-only CPU jobs checked `/data/xuelin/hot3d` against a sanitized manifest
built from the current local HOT3D Aria, Quest, and Assets v4 download-link
JSON files. The manifest retained sequence names, official byte counts, and
SHA-1 values but removed every signed download URL.

Run root:

```text
/data/wentao/ropetrack/runs/hot3d_integrity_20260717_v4
```

Jobs:

| Job | Result | Purpose |
|---:|---|---|
| `184813` | `FAILED 1:0`, 19:57 | Full 424-file VRS hash pass plus an initially over-strict extracted-file audit. |
| `184817` | `COMPLETED 0:0`, 05:04 | Corrected held-out/optional-file rules, reusing all 424 verified VRS hashes. |

Final outputs:

```text
/data/wentao/ropetrack/runs/hot3d_integrity_20260717_v4/results_v2/summary.json
/data/wentao/ropetrack/runs/hot3d_integrity_20260717_v4/results_v2/sequence_results.jsonl
```

## Verified Result

- Manifest and filesystem both contain exactly 424 sequences with no missing
  or extra names: 198 Aria and 226 Quest 3.
- All 424 `recording.vrs` files match the current official v4 byte count and
  SHA-1 individually. Expected and actual totals are both
  `715,020,274,037` bytes.
- Public-GT counts are 136 Aria and 158 Quest 3, total 294.
- All public-GT sequences have the required extracted files, non-empty MANO
  and UmeTrack trajectories, hand-bbox rows, valid JSON metadata, and fully
  parseable MANO/UmeTrack JSONL.
- Aria MPS inventories and summary JSON files pass. Three P0020 recordings do
  not have `personalized_eye_gaze.csv`; this is valid because Project Aria only
  produces personalized gaze when an in-session calibration was recorded.
- Assets contain all 33 expected GLB object models plus the expected metadata
  and license files. GLB headers and declared lengths pass.
- Final audit: zero errors, three expected optional-file warnings,
  `passed=true`, and an empty Slurm stderr log.

## Initial False Positives

The first job reported 913 errors, all explained by two audit assumptions:

- 910 errors were seven missing GT mask files across each of the 130 held-out
  sequences. These sequences intentionally withhold public hand/object GT, so
  the absent masks are not incomplete downloads.
- Three errors were the optional personalized eye-gaze files described above.

No VRS size or SHA-1 mismatch, public-GT parse failure, missing public-GT
sequence, or asset error was observed.

## Validation Boundary

The original GT, hand-data, and MPS ZIP archives were deleted after extraction,
so their archive SHA-1 values cannot be recomputed from the extracted tree.
For those components the audit combines the downloader status, exact expected
inventory, non-empty public-GT checks, and content parsing. The raw VRS files,
which account for 715.0 GB of download bytes, were compared exactly against the
official per-file SHA-1 values.

## Independent Owned Copy

Because the shared source is owned by another account and could later be moved
or removed, CPU job `184834` copied it with resumable `rsync` to:

```text
/data/wentao/datasets/hot3d
```

The job did not use `--delete`, completed `0:0` in `03:09:04`, and wrote its
logs and verification outputs under:

```text
/data/wentao/ropetrack/runs/hot3d_copy_20260717_v4
```

The destination occupies `770G` by `du`. A fresh audit recomputed, rather than
reused, every official VRS SHA-1 and passed with the same exact inventory:
424 sequences (198 Aria, 226 Quest 3), 294 public-GT sequences (136 Aria,
158 Quest 3), `715,020,274,037` expected and actual VRS bytes, zero errors, and
the same three valid optional eye-gaze warnings. The `SUCCESS` marker and
`verify/summary.json` are present.

Conclusion: `/data/wentao/datasets/hot3d` is now the independently owned,
complete, and usable HOT3D v4 full-format copy for planned experiments. The
original `/data/xuelin/hot3d` was not modified or deleted.
