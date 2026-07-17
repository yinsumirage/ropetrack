# Ego-Exo4D V2 Shared-Copy Integrity Audit

Date: 2026-07-17

## Scope

Read-only CPU jobs audited `/data/annie/egoexo4d`. The decisive comparison used
the official V2 manifests left by the Ego-Exo4D downloader under
`logs/manifest_cache/v2`; no dataset files were changed. A separate shallow
login-node preflight only listed the top level.

Run root:

```text
/data/wentao/ropetrack/runs/egoexo4d_integrity_20260717_v2
```

Jobs:

| Job | Result | Purpose |
|---:|---|---|
| `184847` | `COMPLETED 0:0` | Full-tree inventory on a CPU node. |
| `184860` | `COMPLETED 0:0` | Exact path-and-size comparison with cached official manifests. |
| `184864` | `COMPLETED 0:0` | Official EgoPose hand-subset asset coverage. |
| `184876` | `COMPLETED 0:0` | Exact expected, present, and missing byte totals for that hand subset. |

Final outputs:

```text
/data/wentao/ropetrack/runs/egoexo4d_integrity_20260717_v2/inventory/summary.json
/data/wentao/ropetrack/runs/egoexo4d_integrity_20260717_v2/inventory/inventory.tsv.gz
/data/wentao/ropetrack/runs/egoexo4d_integrity_20260717_v2/manifest_check/summary.json
/data/wentao/ropetrack/runs/egoexo4d_integrity_20260717_v2/manifest_check/nonmatching.jsonl.gz
/data/wentao/ropetrack/runs/egoexo4d_integrity_20260717_v2/hand_subset/hand_subset.json
```

## Verified Result

The shared tree is not a complete V2 default download. Its own downloader log
shows an unfiltered default-parts command, followed by 70,577 fetch exceptions
and a proxy-connection traceback on 2026-06-24. Fifteen large randomly suffixed
temporary media files also remain. This is an interrupted download, not a
deliberately filtered finished subset.

Exact cached-manifest comparison:

| Part | Expected | Exact-size matches | Missing | Byte completion |
|---|---:|---:|---:|---:|
| `annotations` | 5,069 files / 11,017,893,912 B | 5,069 | 0 | 100.000% |
| `metadata` | 6 files / 45,707,483 B | 6 | 0 | 100.000% |
| `takes` | 50,383 files / 10,553,486,203,264 B | 8,935 | 41,448 | 28.266% |
| `take_vrs_noimagestream` | 4,994 files / 995,592,133,908 B | 24 | 4,970 | 0.141% |

Every present final-named file in these four manifests has the expected byte
size; there are no size mismatches. The manifests contain no checksums, so this
matches the official downloader's path-and-size integrity rule but is not a
cryptographic content proof.

The inventory found 22,747 files and 16,661 directories, using 3.063 TB of
logical bytes. It includes all 5,035 take directories in metadata, but only
1,268 take directories contain any downloaded files. `captures` occupies the
documented 43.618 GB and has 787 directories, which is strong aggregate
evidence but not an exact per-file proof because its cached manifest is absent.
Observed trajectory data is only 4.416 GB, versus 509.503 GB documented for the
V2 `take_trajectory` part.

## EgoPose Hand Gate

Public manual hand annotations are complete for train and validation: 286 + 65
= 351 take UIDs. Camera-pose JSON is present for all 351. The corresponding raw
assets are incomplete:

| Requirement | Covered take UIDs |
|---|---:|
| Hand annotation | 351 / 351 |
| Camera pose | 351 / 351 |
| Ego RGB (`*_214-1.mp4`) | 226 / 351 |
| Every ego-view take path selected by the official recipe | 217 / 351 |
| `take_vrs_noimagestream` | 0 / 351 |
| RGB + camera pose + VRS intersection | 0 / 351 |

For these 351 UIDs, the official ego-view take selection totals
`123,876,243,015` bytes; `56,079,817,391` bytes are present and
`67,796,425,624` bytes are missing. Their no-image-stream VRS files total
`79,299,495,417` bytes and none are present. Completing the public manual
train/validation hand recipe therefore requires about 147.10 GB of additional
raw assets, not completion of the 12.11 TB default download.

No pre-generated `aria_calib_json` directory is present. Therefore this tree
cannot currently run the official EgoPose hand preparation path for even one
take end to end: the official recipe needs either the no-image-stream VRS for
calibration extraction or pre-generated calibration JSON. A custom uncalibrated
or already-distorted-image protocol would be a different protocol and must not
be presented as official benchmark readiness.

## Decision

- Reuse `annotations` and `metadata` as complete V2 parts.
- Do not label or copy the whole shared tree as a complete Ego-Exo4D dataset.
- A frozen 226-take RGB+annotation subset may support a deliberately custom
  inspection or training experiment, but it is not official-protocol ready.
- Completion should be performed by the credentialed owner rerunning the same
  official downloader command, which resumes by path and size. Do not use
  `--force` unless corruption is independently suspected.

Official references:

- <https://docs.ego-exo4d-data.org/download/>
- <https://docs.ego-exo4d-data.org/annotations/ego_pose/>
- <https://github.com/EGO4D/ego-exo4d-egopose/tree/main/handpose/data_preparation>
- <https://github.com/facebookresearch/Ego4d/blob/main/ego4d/egoexo/download/README.md>
