# EgoVerse download-link probe

Date: 2026-07-20

## Scope

This was a bounded acquisition check, not dataset integration or training. Raw
third-party data stayed outside the repository under
`/data/wentao/datasets/egoverse`. GitHub/PyPI connectivity was tested
separately from the data path; AWS Secrets Manager, the read-only SQL database,
and Cloudflare R2 were accessed with all proxy variables unset.

## Successful chain

CPU job `189211` completed in 17m23s and proved:

1. the public bootstrap credentials in the official README can retrieve the
   read-only DB and R2 secrets;
2. the read-only SQL table is reachable directly from a CPU node;
3. a deterministic academic pilot query selected RL2 `fold_clothes` episode
   `2025-09-20-17-42-51-000000` (`human_bimanual`);
4. direct, no-proxy R2 listing returned 26 objects totaling 187,017,579 bytes;
5. all 26 objects downloaded to
   `/data/wentao/datasets/egoverse/raw/2025-09-20-17-42-51-000000`;
6. every local object matched its remote byte size and no `.part` file remained.

The frozen audit manifest is
`/data/wentao/datasets/egoverse/audit/2025-09-20-17-42-51-000000.json`.

## Data finding

The episode contains 640x480 JPEG RGB, intrinsics, head pose, eye gaze, RGB
timestamps, left/right wrist and EE poses, raw Aria keypoints, and fitted
left/right 21x3 keypoints. It is therefore materially more useful than an
RGB-only weak dataset pilot.

The apparent frame-count mismatch is structured padding, not an isolated corrupt
episode:

- SQL and group metadata report 2,808 valid frames;
- every non-annotation array has 2,900 rows and is mutually aligned, consistent
  with padding to the next 100-frame boundary;
- `annotations` exists but has shape `[0]`.

Consumers must slice arrays to `num_frames`/`total_frames` and must not train on
the padded tail or assume language annotations are populated. No image decode,
overlay, coordinate, unit, handedness, or keypoint-quality claim is made by this
download probe.

## Multi-source check

Three additional no-proxy CPU jobs downloaded complementary small episodes in
parallel:

| Job | Lab / task / embodiment | Valid / stored frames | Objects | Bytes | Time |
| --- | --- | --- | ---: | ---: | ---: |
| `189251` | ETH / `scoop_granular_in_domain` / bimanual | 560 / 600 | 26 | 42,774,953 | 4m34s |
| `189252` | Song / `sort_utensils` / left arm | 406 / 500 | 18 | 21,222,160 | 1m38s |
| `189253` | Scale / `flagship_put_object_in_container` / right arm | 150 / 200 | 18 | 9,954,272 | 3m25s |

All three jobs completed `0:0`; their local file counts and total bytes match
their frozen manifests, with no `.part` files. All dense arrays again pad to the
next 100-frame boundary. The academic Aria episodes contain
`obs_aria_keypoints`, eye gaze, and RGB timestamps. The Scale episode omits
those source-specific fields but contains fitted left/right 21x3 keypoints and
one annotation entry. The three academic episodes have empty annotation arrays.

The paper describes the 21-point 3D hand poses as *estimated* in the camera
frame, not manual or motion-capture ground truth. Treat them as processed
pseudo-labels until projection, units, handedness, and temporal quality are
audited.

## Failures and fixes

- Git clone through `hpc:7999` stalled for 120 seconds even though a proxied
  GitHub HEAD request returned 200. The incomplete clone was not used.
- Installing the full AWS Python stack through the proxy stalled on the
  14.7 MiB `botocore` wheel. Direct PyPI/file access worked; the final probe
  required only cached `psycopg[binary]` plus stdlib/curl.
- Raw curl SigV4 first missed `x-amz-content-sha256`, then produced a signature
  mismatch when that header was added manually.
- R2 rejected `X-Amz-Security-Token` for ListObjectsV2. This matches the
  official downloader path, which uses the R2 access key and secret but not the
  returned session token.
- The working path generates short-lived R2 presigned URLs locally and uses
  four direct curl transfers with `.part` resume. Four workers were sufficient;
  do not copy the official default of 128 workers without a measured need.

## Judgment

The HPC acquisition path is validated for bounded multi-source subsets. The
three parallel jobs moved 73,951,385 bytes in 4m34s wall time, versus 9m37s if
run sequentially, so episode-level concurrency helps. More CPU cores alone do
not: the workload is network-bound and each episode is dominated by a large
image object. Prefer one CPU allocation with 4-8 concurrent episodes rather
than many cores or the official 128-worker default. Perform content/coordinate
overlays before exporting RopeTrack-owned derivatives. Do not run stale named
filters or an unbounded all-episode sync.
