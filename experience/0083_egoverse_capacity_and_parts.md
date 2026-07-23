# EgoVerse capacity and acquisition parts

Date: 2026-07-20

## Scope

This was a read-only SQL/R2-manifest capacity audit. It did not start a bulk
download. Raw third-party data remains under `/data/wentao/datasets/egoverse`.
All database, manifest, and data access ran with proxy variables unset.

## Human-data capacity estimate

CPU job `189316` queried all active `human_*` episodes and listed the R2 object
sizes for 20 deterministic episode samples per lab. Per-lab aggregate
bytes/frame was projected onto each lab's SQL frame total.

| Lab | Episodes | Hours at 30 FPS | Projected size |
| --- | ---: | ---: | ---: |
| ETH | 540 | 38.2 | 191 GB |
| RL2 | 1,558 | 55.7 | 327 GB |
| Song | 280 | 8.4 | 43.6 GB |
| Wang | 146 | 6.0 | 35.8 GB |
| Scale | 17,090 | 216.6 | 1.36 TB |
| Mecka | 41,617 | 828.4 | 4.91 TB |
| **Total** | **61,231** | **1,153.3** | **6.87 TB** |

The projected full human-data size is 6.25 TiB. A within-lab episode bootstrap
gives a 95% sampling interval of 6.47-7.32 TB; this does not cover future
dataset updates or systematic differences between sampled and unsampled
episodes. The live explorer's larger 83,118-episode / 1,811-hour headline also
includes non-human embodiments, so it must not be multiplied by the human
bytes/frame estimate.

## Useful download parts

SQL-only job `189345` found:

- all four academic Aria labs: 2,524 episodes, 108.3 hours, 598 GB;
- Scale flagship: 4,404 episodes, six tasks, 39.0 hours, 245 GB;
- Scale freeform: 12,686 episodes, 65 tasks, 177.6 hours, 1.12 TB;
- Mecka one deterministic episode per task: 4,858 episodes, 112.0 hours,
  664 GB;
- Mecka up to three per task: 8,812 episodes, 204.6 hours, 1.21 TB;
- Mecka up to five per task: 11,238 episodes, 261.9 hours, 1.55 TB.

The recommended first durable acquisition is therefore:

1. all academic Aria labs;
2. all Scale flagship episodes;
3. one deterministic Mecka episode per non-empty task.

This balanced part is 11,786 episodes, about 259 hours and 1.51 TB (rough
sampling range 1.42-1.61 TB). It preserves standardized academic coverage,
Scale flagship repetition, and Mecka task diversity without paying for all
4.91 TB of Mecka repetition.

## Throughput boundary

The three-episode pilot moved 73,951,385 bytes in 4m34s wall time, only
0.27 MB/s aggregate. More CPU cores do not accelerate a large R2 object; useful
parallelism is across episodes and nodes. At the measured pilot rate the full
corpus would be impractically slow. A bounded 32-episode, multi-node scaling
pilot is required before bulk acquisition. If sustained aggregate throughput
reaches 3-10 MB/s, the balanced 1.51 TB part would take about 5.8-1.7 days and
the full 6.87 TB about 26.5-8.0 days. These are scenarios, not measured promises.

The shared `/data` filesystem reported 1.4 PB available, but no per-user quota
was reported; confirm policy/quota before a multi-TB pull.

## Projection check status

Repository inspection confirms that academic `obs_aria_keypoints` are raw Aria
layout and `obs_keypoints` are MANO-fitted canonical joints in the same world
frame. The official browser transforms world keypoints with
`inv(obs_head_pose)` before applying per-episode intrinsics. A standalone
overlay was prepared using that exact convention, but its isolated Zarr 3
environment did not complete because the HPC proxy repeatedly timed out on the
16.1 MiB NumPy wheel. No data defect was observed; overlay remains a
non-blocking environment task and must not be claimed as visually validated.

## Judgment

Do not download all 6.87 TB now. Freeze manifests for the balanced 1.51 TB part,
run a 32-episode throughput pilot, then acquire the academic 598 GB first. Add
Scale flagship and Mecka-one-per-task only after byte completeness and a small
projection review pass. Keep Scale freeform and repeated Mecka episodes as
later expansion parts.

## Acquisition jobs started on 2026-07-21

The bulk downloader reuses the byte-verified direct R2 path from the link probe,
keeps all proxy variables unset, downloads multiple episodes concurrently, and
can resume `.part` objects. The user submitted these CPU jobs from an existing
HPC login session after all four selectors passed their dry runs:

| Job | Selection | Size of selection |
| --- | --- | ---: |
| `189644` | all academic Aria labs | 2,524 episodes |
| `189645` | Scale flagship sample | 18 episodes, three per task |
| `189646` | Scale freeform sample | 20 distinct tasks |
| `189647` | Mecka sample | 20 distinct tasks |
| `189648` | dependent sample completeness/schema audit | runs after `189645-189647` |

The academic array has 16 deterministic shards with at most four active at
once. Each sample has four shards with at most two active. Do not submit a
second copy of these arrays: inspect `sacct`, logs, selection manifests, and
per-episode byte manifests first. Scale flagship full acquisition remains
gated on RGB/keypoint projection and temporal-quality checks; Scale freeform
and Mecka bulk acquisition remain manual decisions even if their samples pass.

A two-hour Codex automation tracked these jobs during acquisition. It was
deleted after the final 2026-07-21 closure pass: all selected bytes were
verified, no job remained active, and both Scale parts were blocked at the
quality gate. No background acquisition or automatic bulk submission remains.

## Automated monitoring pass on 2026-07-21 05:52 +08:00

The first automation pass found the sample arrays healthy at the byte gate but
the academic array only partially complete:

| Selection | Verified episodes | Verified remote/local bytes | Extra local in-flight bytes | `.part` |
| --- | ---: | ---: | ---: | ---: |
| academic Aria | 1,745 / 2,524 | 417,140,352,763 | 4,204,923,610 | 22 files / 1,975,489,543 bytes |
| Scale flagship sample | 18 / 18 | 1,630,917,993 | 0 | 0 |
| Scale freeform sample | 19 / 20 | 2,862,545,534 | 432,301,858 | 0 |
| Mecka sample | 20 / 20 | 2,973,416,968 | 0 | 0 |

The academic array's failed shards were not data/schema failures. Shards 6, 7,
and 12 saw transient DNS failures to R2 or AWS Secrets Manager; shard 5 saw a
single interrupted curl transfer; shard 14 saw the read-only PostgreSQL server
close the connection during startup. The downloader writes to `.part`, resumes
with `curl --continue-at -`, skips complete byte-verified manifests, and only
writes an episode manifest after all object sizes match. Therefore only the
failed shard IDs were resubmitted: job `189898` for shards 5, 6, 7, and 12, and
job `189903` for shard 14. No second 16-shard academic array was generated.

At the snapshot time, academic acquisition had sustained about 51 MB/s from
03:34, with roughly one hour remaining against the earlier 598 GB projection;
this ETA remains approximate because shard retries and episode sizes are
uneven. The completed Scale flagship and Mecka samples averaged about 0.25 MB/s
and 7.32 MB/s respectively. Scale freeform still had one unverified episode, so
its completion ETA was not defensible from the partial object alone.

The dependent structural audit `189648` remained pending on Scale freeform.
That audit checks manifests, shapes, padding, annotation counts, and field
coverage but is not sufficient for the quality gate. A separate CPU audit was
added under ignored `.local_checks/` to decode every sampled RGB frame, measure
NaN/zero/missing 21-point keypoints, projection landing, root-relative temporal
jumps, handedness metadata, padding/truncation, and annotation coverage, and to
render an overlay montage. Its first submission `189902` failed before reading
data because a Python 3.10 environment cannot install the required Zarr 3.1.5;
the corrected Python 3.11 submission is `189904`. Scale flagship full remains
blocked until that report and montage pass review, and no Scale freeform or
Mecka bulk job was submitted.

### Scale flagship quality result

Corrected quality job `189904` completed successfully over all 18 episodes,
23,184 RGB frames, six tasks, and 1,630,917,993 byte-verified bytes. Every RGB
frame decoded, all 30 expected hands had zero NaN/zero/missing keypoints, the
minimum expected-hand projection landing ratio was 94.96%, handedness metadata
matched embodiment, all 174 annotation records parsed and covered 100% of valid
frames, and no array was shorter than `total_frames`. The 104 padded array
instances had 2-96 extra frames and were correctly truncated to the declared
valid range.

The temporal gate failed in two `flagship_folding_clothes` episodes. Episode
`2026-05-02-18-31-58-322530` had a right-hand 13.8 cm root-relative one-frame
jump affecting 19/21 joints; episode `2026-05-02-18-59-30-658862` had an 8.6 cm
left-hand jump affecting 18/21 joints. Follow-up montage job `189907` confirmed
these occur between visually near-adjacent 30 FPS frames rather than across a
scene cut. This is label jitter, not a byte/decode/projection failure. Therefore
the Scale flagship sample does **not** pass the requested quality gate and no
`scale_flagship_all` job was submitted.

By 06:02, academic acquisition had advanced to 2,034/2,524 verified episodes
and 482,839,750,467 verified bytes, plus 5.18 GB of in-flight local data
(including 20 `.part` files totaling 917,291,008 bytes). Average verified
throughput since 03:34 was about 54.7 MB/s, leaving roughly 35 minutes against
the 598 GB projection. Retry shards 5, 6, and 7 had completed; retry 12 and the
separate shard-14 retry were still running alongside original shards 10, 11,
13, and 15.

### Mecka sample quality and remaining dependent audit

Mecka quality job `189908` initially reported five invalid annotations, but the
raw records all ended exactly at `total_frames`. Repository documentation
defines annotation spans as half-open `[start_idx, end_idx)`, with
`end_idx <= total_frames`; the audit had incorrectly required
`end_idx < total_frames`. The shared audit was fixed and corrected job `189919`
passed all automated gates over 20 episodes, 48,564 RGB frames, 40 expected
hands, 2,973,416,968 byte-verified bytes, and 142 annotation records. RGB decode,
NaN/zero/missing keypoints, handedness, array lengths, projection (minimum
96.78%), and temporal-jump thresholds all passed. Annotation coverage varies
from 57.43% to 100% by episode, which is reported coverage rather than a parse
or bounds defect. The overlay montage showed no systematic left/right swap or
padding/truncation failure. No Mecka bulk download was submitted.

Freeform remained at 19/20 verified episodes; quality job `189909` is queued
with `afterok:189646` and will run only after the last sample shard completes.
At 06:10, academic acquisition had reached 2,163/2,524 verified episodes and
519,627,335,813 verified bytes, with about 55.9 MB/s average verified
throughput and roughly 23 minutes remaining against the 598 GB projection.

## Automated monitoring pass on 2026-07-21 07:45 +08:00

The original academic shards 13 and 15 failed together during an R2 DNS outage;
they left 104 and 117 episodes incomplete. The Scale freeform shard 2 also
failed on its final episode after two 215-second connection timeouts exhausted
the five-minute signed URL and the last curl retry received HTTP 403. These are
transport failures, not selection or schema failures.

The byte snapshot before the corrected retry was:

| Selection | Verified episodes | Verified remote bytes | Local bytes | `.part` |
| --- | ---: | ---: | ---: | ---: |
| academic Aria | 2,305 / 2,524 | 558,940,004,877 | 567,228,186,183 | 1 / 118 bytes |
| Scale flagship sample | 18 / 18 | 1,630,917,993 | 1,630,917,993 | 0 |
| Scale freeform sample | 19 / 20 | 2,862,545,534 | 3,465,783,881 | 2 / 236 bytes |
| Mecka sample | 20 / 20 | 2,973,416,968 | 2,973,416,968 | 0 |

All three 118-byte `.part` files contained the complete R2
`ExpiredRequest` XML response rather than object bytes. This exposed a silent
integrity risk in the shared ignored downloader: `curl --fail-with-body` could
append an error response to `.part`, after which a later byte-range resume could
reach the expected size with corrupt content. The shared R2 helper now records
the pre-call partial length, truncates back to it after any failed curl, and
uses a 30-second connect timeout so retries stay inside the signed-URL window.
The focused rollback self-check passed in the existing HPC EgoVerse environment.
The three confirmed error-body files were deleted; no valid dataset bytes were
removed.

The first attempted retries (`189983`, `189984`) were cancelled as soon as this
risk was found. Corrected resumable retries are academic shards 13 and 15 in
`189996`, and Scale freeform shard 2 in `189997`; all download proxy variables
remain unset. Replacement structural audit `189999` and freeform deep-quality
audit `190000` depend on `189997`. The obsolete dependency-never-satisfied jobs
`189648`, `189909`, `189986`, and `189987` were cancelled.

At 07:45 all three corrected download tasks were running. Academic verified
throughput averaged about 37.2 MB/s from the original 03:34 start, depressed by
the DNS outage and idle retry gap. About 39.1 GB remained against the earlier
598 GB projection, so the completion estimate was roughly 20-40 minutes at the
observed healthy two-shard retry rate; this remains approximate because episode
sizes are uneven. Freeform had about 603 MB already local but no verified
completion, so its audits remained pending and no defensible quality conclusion
or ETA beyond the active single-episode retry was claimed.

The first post-fix progress snapshot at 07:46 advanced academic Aria to
2,320/2,524 verified episodes and 561,733,490,814 verified bytes. Two newly
created `.part` paths were present but both were zero bytes during the snapshot;
the three 118-byte error bodies did not recur. Scale freeform remained 19/20
with 603,238,111 local bytes in its final episode and no `.part` file.

The quality decisions are unchanged. Scale flagship still fails the requested
gate because two folding-clothes episodes have visually confirmed implausible
one-frame hand-label jumps; no `scale_flagship_all` job exists or was submitted.
Corrected Mecka quality remains a pass over 20 episodes, 48,564 RGB frames, 142
valid annotations, zero decode/NaN/zero/missing/handedness/short-array failures,
and at least 96.78% expected-hand projection landing. No Mecka or Scale freeform
bulk download was submitted.

## Automated monitoring pass on 2026-07-21 09:48 +08:00

The original academic array `189644` has nine completed and seven failed shards.
The earlier targeted retries completed shards 5, 6, 7, 12, and 14. In retry
`189996`, shard 13 completed but shard 15 failed again; Scale freeform retry
`189997_2` also failed. Both failures were the same transient direct-R2 DNS
resolution outage. The failure path left no `.part` files or error-response
bodies, confirming that the rollback hardening from the prior pass held.

The 09:44 byte snapshot was:

| Selection | Verified episodes | Verified remote bytes | Local bytes | `.part` |
| --- | ---: | ---: | ---: | ---: |
| academic Aria | 2,461 / 2,524 | 601,448,496,450 | 602,543,931,648 | 0 |
| Scale flagship sample | 18 / 18 | 1,630,917,993 | 1,630,917,993 | 0 |
| Scale freeform sample | 19 / 20 | 2,862,545,534 | 3,482,816,204 | 0 |
| Mecka sample | 20 / 20 | 2,973,416,968 | 2,973,416,968 | 0 |

Academic advanced by 39,715,005,636 verified bytes from the 07:46 snapshot.
Those bytes landed during the roughly 17.5-minute healthy retry window, about
37.7 MB/s aggregate; the long idle interval after the DNS failure is excluded.
Sixty-three academic episodes remained. At shard 13's observed episode rate,
shard 15 should need roughly 10-20 minutes if DNS stays healthy, but episode
sizes are uneven and the old 598 GB capacity projection has already been
exceeded. The final freeform episode had 620,270,670 local bytes, but an ETA from
partial complete objects is not defensible.

At 09:45 the R2 hostname resolved again and no equivalent live downloader was
present. Safe resume therefore submitted only academic shard 15 as `190179` and
freeform shard 2 as `190180`; it did not regenerate either array. Replacement
structural audit `190181` and freeform deep-quality audit `190182` depend on
`190180`. Obsolete dependency-never-satisfied jobs `189999` and `190000` were
cancelled. At 09:48 both download retries were running with empty stderr;
`190179` had already byte-verified three previously incomplete episodes. All
data-download proxy variables remain unset.

`sample_audit.json` and the freeform quality JSON/montage do not exist yet and
must not be treated as passes; they are the outputs of `190181` and `190182`.
The Scale flagship and Mecka JSON/montage artifacts are present. The quality
decision is unchanged: Scale flagship fails the requested gate because of the
two visually confirmed 13.8 cm and 8.6 cm one-frame label jumps, so no
`scale_flagship_all` job was submitted. Mecka remains a quality pass, but no
Mecka or Scale freeform bulk task was submitted.

## Automated monitoring pass on 2026-07-21 11:44 +08:00

Academic retry `190179_15` completed all remaining shard-15 episodes. The
frozen `academic_all` selection is now complete at 2,524/2,524 episode
manifests and 615,969,380,047 byte-verified local/remote bytes, with no
`.part` files. It added 14,520,883,597 verified bytes after the 09:44 snapshot
in 1:03:36 wall time, about 3.8 MB/s over the whole retry allocation; remaining
time is zero. The final measured total is about 18 GB above the old 598 GB
sample projection.

Freeform retry `190180_2` failed again on the one incomplete episode after
19:44. Its log shows only direct-R2 DNS-resolution failures, not a manifest,
schema, or byte-integrity failure. The snapshot remains 19/20 verified episode
manifests and 2,862,545,534 verified remote bytes, with 640,182,895 local bytes
in the incomplete episode and zero `.part` files. Because its expected remote
total is not yet recorded in an episode manifest, neither throughput nor ETA is
defensible from the partial local tree.

At 11:42 the R2 hostname resolved and no equivalent downloader was active.
Obsolete dependency-never-satisfied audits `190181` and `190182` were cancelled,
then only freeform shard 2 was resumed as `190289`; replacement structural and
deep-quality audits are `190290` and `190291`, both dependent on `190289`.
Download proxy variables were unset. At 11:44 `190289_2` was running and both
audits were pending on dependency. No academic duplicate, Scale flagship full,
Scale freeform full, or Mecka bulk task was submitted.

Quality decisions remain unchanged. Scale flagship fails because its two
folding-clothes episodes have visually confirmed implausible one-frame label
jumps. Mecka passes the sampled decode/keypoint/projection/temporal/handedness/
padding/annotation gate. Freeform remains unknown until `190290` and `190291`
produce `sample_audit.json`, the quality JSON, and the montage.

## Automated monitoring pass on 2026-07-21 15:52 +08:00

Freeform retry `190289_2` completed the one remaining episode at 13:59:42.
The frozen sample is now 20/20 episode manifests and 3,766,991,911
byte-verified local/remote bytes with zero `.part` files. It added 904,446,377
verified bytes over the 2:15:30 allocation, about 0.11 MB/s effective verified
throughput; most bytes of that episode had already landed before this retry,
and recurring R2 DNS stalls dominated wall time. Remaining time is zero.
Academic Aria remains complete at 2,524/2,524 and 615,969,380,047 bytes;
Flagship remains 18/18 and 1,630,917,993 bytes; Mecka remains 20/20 and
2,973,416,968 bytes. Every selection has matching local/manifest bytes and
zero `.part` files.

Dependent structural audit `190290` and deep-quality audit `190291` produced
fresh `sample_audit.json`, `quality_scale_freeform_sample.json`, and the
Freeform montage. The structural audit found 20 tasks, 275 valid annotation
spans, and 14 padded episodes. The deep audit covered 58,407 decoded RGB
frames and 39 expected hands with zero decode, NaN, zero-point, missing-hand,
short-array, handedness, or annotation-parse failures. Minimum expected-hand
projection landing was 97.34%; per-episode annotation coverage was
98.54-99.98%. There were 112 padded arrays across 14 episodes, each 3-82
frames longer than `total_frames`; all were safely truncated to the declared
valid range. The overview montage showed sensible projection and no systematic
left/right swap.

Freeform nevertheless **fails the temporal quality gate**. Episode
`2026-05-02-18-06-43-995379` (`freeform_unload_dishwasher`) has 0.5706% of
right-hand root-relative joint transitions over 5 cm, just above the 0.5%
threshold. Focused CPU render `190591` identified frame 198->199 as a 10.5 cm
maximum jump affecting 17/21 joints. Its first `/data` JPEG was not visible
from the login node despite a successful job, so the same read-only render was
repeated to shared home storage as `190598`; that job completed in three
seconds and the montage visually confirmed the label-shape jump between
near-adjacent 30 FPS images. This is not a byte, projection, or padding defect.

The shared local and HPC `egoverse_link_probe.py` now gives both signed-AWS and
R2 curl calls a 30-second connect timeout, `--retry-all-errors`, and a two-second
retry delay while preserving resumable `.part` files, failed-response rollback,
and per-object byte verification. The focused rollback/retry-option check and
Python compilation passed in the existing EgoVerse environment; no package was
installed. No active EgoVerse job or log evidence of `scale_flagship_all`,
Scale Freeform full, or Mecka bulk exists, and none was submitted. Scale
Flagship full remains blocked by its two confirmed jitter failures; Freeform is
also now a sampled temporal-gate failure.

## Four-lab academic Aria content audit on 2026-07-21

CPU job `190618` selected six task/embodiment-diverse episodes from each of
RL2, ETH, Song, and Wang, then audited every valid frame in those 24 episodes.
The sample covered 113,529 decoded RGB frames and 5,987,827,995 byte-verified
bytes. RGB decode, NaN/zero/missing-hand, short-array, handedness, and
annotation parsing checks had zero failures. All 24 academic episodes had zero
dense annotation records. There were 252 padded arrays, all longer than the
declared valid frame count and safely truncated.

The fitted-keypoint temporal gate failed: 24 of 39 expected hands exceeded the
0.5% threshold for root-relative joint transitions over 5 cm. The worst ETH
`sort_utensils` hand had 17.03% such transitions, a 17.54 cm p99, and only
8.15% of projected points inside the image. Focused render `190641` visually
confirmed 22-26 cm one-frame tracking/association jumps rather than a mere
out-of-frame statistical artifact. Separate worst-case renders `190644-190646`
showed additional tracking loss or shape jumps in RL2, Song, and Wang.

Job `190656` compared the source `obs_aria_keypoints` with the canonical
MANO-fitted `obs_keypoints` on the same 113,529 frames. Raw Aria failed 25/39
hands with a 0.5696% median over-5-cm transition rate; MANO-fitted failed 24/39
with a 0.6530% median. The worst rates were 16.89% raw and 17.03% fitted. Thus
the MANO fitting step is not the root cause and choosing raw Aria points does
not remove the temporal-noise problem. The raw-versus-fitted montage shows
good visible-hand alignment in many RL2/Song/Wang frames, but both
representations follow the same upstream failures in bad frames.

Under the same strict precision-first gate, the current 20-episode Mecka sample
is the strongest EgoVerse source: 48,564 decoded frames, 40 expected hands,
zero failed hands, minimum 96.78% projection landing, and a maximum 0.3289%
over-5-cm transition rate. Flagship and Freeform remain spatially healthy but
have two and one visually confirmed temporal failures respectively. This is a
sample-level comparison, not proof that all Mecka episodes are clean.

The next defensible acquisition, if more diversity is desired, is about 200
deterministic one-per-task Mecka episodes (roughly 27-30 GB from the measured
and projected per-episode rates), followed by the same quality audit. Do not
jump directly to the 4,858-episode / 664 GB one-per-task part. Academic Aria is
retained as noisy pseudo-label data and requires visibility/temporal filtering
before any training use. No additional data was downloaded in this audit.

## Staged 200-task Mecka acquisition on 2026-07-21

The user approved the bounded Mecka expansion. Profile `mecka_200` selects one
deterministic episode from each of 200 distinct non-empty Mecka tasks, ordered
by `md5(task)` and then `md5(episode_hash)`. Its dry run selected exactly 200
episodes, with 25 episodes in shard 0 of 8. Because this is a prefix extension
of the prior deterministic 20-task sample, already byte-verified episodes are
reused rather than downloaded again.

Slurm array `190812` was submitted as 8 CPU shards with at most 4 active at a
time. All data proxy variables remain explicitly unset. Dependent CPU job
`190813` will run the existing full RGB decode, keypoint integrity, projection,
handedness, padding, annotation, temporal-jitter, and montage audit only after
every download shard succeeds. The measured sample rate projects about 27-30
GB total, including the existing 2.97 GB sample. At submission time shard 0 was
running and the other shards were waiting for CPU resources; no Scale bulk or
Mecka 4,858-task bulk was submitted.

Real annotation inspection confirms that these sources provide temporally
bounded language actions, not only episode-level task names. Examples include
Flagship `Open the cutlery drawer organizer` at frames 0-117, Freeform
`Retrieve the wooden spatula from the upper dishrack kitchen counter` at frames
0-217, and Mecka `dip sponge in soap water, scrub tray with sponge` at frames
306-618. These spans are useful language supervision but are not independent
metric 3D ground truth.

The initial array finished six shards successfully but shard 0 failed before
selection on one SQL connection timeout, while shard 7 failed one episode after
three R2 connection timeouts. The impossible dependent audit `190813` was
cancelled. Database selection now makes up to three 30-second connection
attempts with five-second gaps. Only shards 0 and 7 were resumed as array
`190875`; completed manifests are byte-checked and skipped. Shard 7 then
downloaded the sole missing episode `696c601818824db10c3a76a1` (142,997,321
bytes) and completed cleanly, while shard 0 entered normal running state with
no error log. Replacement full-quality audit `190876` depends on this retry.

## Mecka-200 result and training-pool expansion on 2026-07-22

Retry `190875` and quality audit `190876` completed successfully. The frozen
Mecka-200 selection contains 200/200 byte-verified episodes, 28,769,035,664
bytes, 494,736 decoded RGB frames, and 1,624 valid language spans. There were
zero RGB decode, array-length, NaN, missing-hand, handedness, or annotation
parse failures. All episodes have annotations; median frame coverage is
99.96%, minimum 62.78%, and only five episodes are below 99%.

The strict trusted-GT gate failed only on temporal jitter: 4/400 expected hands
across four episodes exceeded 0.5% root-relative joint transitions over 5 cm.
The median hand rate is zero, p95 0.176%, p99 0.349%, and maximum 2.087%.
Minimum projection landing is 86.14%. Focused render `191436` confirmed that
the worst `sorting_plastic_flakes` right hand has localized 9.5-11.2 cm shape
jumps around occlusion or rapid pose changes, rather than corrupt RGB or a
systematic coordinate error. Mecka therefore qualifies as filterable
large-scale pretraining data, but not unfiltered precise GT.

The refreshed acquisition choices are 4,858 one-per-task Mecka episodes
(663,983,927,499 projected bytes), 4,404 Scale Flagship episodes
(244,870,626,906 bytes), and 12,686 Scale Freeform episodes
(1,116,633,305,199 bytes). With 1.3 PB shown available on `/data` and no quota
output, jobs `191445`, `191446`, and `191447` were submitted for this roughly
2.02 TB training pool. Aggregate active-array concurrency is capped at 16
CPU jobs (8+4+4); all proxy variables are unset. Dry runs matched the exact
database counts. All three arrays entered running state, Mecka wrote new
byte-verified episodes, and Scale created growing episode directories with
empty error logs. The remaining 4.25 TB of repeated Mecka episodes was not
submitted; decide on it after complete task coverage is manifest-verified.

## Initial full-pool throughput estimate on 2026-07-22

Jobs `191445-191447` were submitted at 02:46:44 +08:00 and their first array
tasks started at 02:47:33, so queue delay was only 49 seconds. A CPU-node
snapshot at about 03:19 included complete manifests plus in-flight local bytes.
Mecka had 1,106/4,858 verified episodes and 159.61 GB local (24.04% of the
projection); 130.84 GB was new beyond the frozen Mecka-200 baseline, about
66.9 MB/s aggregate. Its current ETA is roughly 2.1 hours remaining, or
05:20-05:40 if the first-wave rate holds.

Scale is source/network limited rather than queue limited. Flagship had
31/4,404 verified episodes and 2.69 GB local (1.10%); its 1.06 GB of new bytes
implies about 0.54 MB/s and a byte-based ETA of 5.2 days. Episode-rate
extrapolation is slower, so use a 5-8 day range at the current four-task
throttle. Freeform had 28/12,686 verified episodes and 4.88 GB local (0.44%);
its 1.12 GB of new bytes implies 0.57 MB/s and a byte-based ETA of 22.5 days.
Episode-rate extrapolation gives about 36 days, so use a 3-5 week range at the
current throttle. There were no failure lines; active `.part` files represented
normal resumable writes. Higher Scale array throttles could shorten these
ranges, but scaling should be based on a mature first wave because R2/object
metadata rather than CPU is the bottleneck.

## Automated download pass at 2026-07-22 05:05 +08:00

Mecka array `191445` had 47 completed shards, seven running, two failed, and
nine pending. The two failures were isolated R2 HTTP 503 responses in shards
21 and 22; all other episodes in those shards were already byte-verified.
Only those two shards were resumed as `191634`, with all data proxy variables
unset. The retry downloaded the two missing episodes (138,523,492 and
139,947,175 bytes), completed in under one minute, and produced empty error
logs. No full-array duplicate was submitted.

The CPU progress snapshot `191635` found Mecka at 3,981/4,858 verified
episodes, 564.81 GB verified and 568.85 GB local, with 13 normal in-flight
`.part` files. Since the 03:19 snapshot, local data grew by 409.24 GB, about
64.4 MB/s; the remaining task-coverage download should take roughly 25-45
minutes if the rate holds.

Scale had no failed log lines, but no mature first shard yet, so array
throttles remained at four each. Flagship was 91/4,404 verified, 9.76 GB
verified and 11.56 GB local. Freeform was 48/12,686 verified, 5.92 GB verified
and 9.00 GB local. The early aggregate capacity projections are now visibly
low for Scale: current verified averages imply about 473 GB for Flagship and
at least 1.57 TB for Freeform, while in-flight and the earlier frozen samples
put Freeform plausibly near 1.8-2.4 TB. At the current four-task throttles,
use roughly 4-5 days for Flagship and 4-6 weeks for Freeform until completed
first shards provide a better rate. Do not raise concurrency before that gate.

## Automated download pass at 2026-07-22 07:08 +08:00

Mecka one-per-task is complete by content, not merely by Slurm state:
4,858/4,858 selected episodes have manifests, verified remote and local bytes
both equal 689,389,192,671, and `.part` count is zero. This is 3.8% above the
663.98 GB early projection. The only failures were the two transient R2 HTTP
503 episodes already recovered by retry `191634`; no data gap remains.

Flagship produced its first mature shard: shard 2 completed 92 episodes in
2:32:21 with no failure. Its array throttle was therefore raised one step from
4 to 8 via `scontrol`; six tasks entered running state immediately and two
waited for CPU resources. Flagship was 388/4,404 verified, 21.09 GB verified
and 24.07 GB local at the snapshot. Its verified mean now agrees with the
original roughly 245 GB projection; wait for post-throttle throughput before
changing the limit again.

Freeform remained at throttle 4. It was 76/12,686 verified, 9.58 GB verified
and 13.58 GB local. Four episodes in original shards 1-3 recorded R2 connect
timeouts after three attempts. Those shards were still running, so no retry
was submitted yet: retrying a live shard would duplicate its remaining work.
The current verified/in-flight average projects roughly 1.6-1.8 TB rather than
the early 1.12 TB estimate, and the current rate still implies several weeks.

## Automated download pass at 2026-07-22 09:05 +08:00

The Flagship throttle-8 trial failed its safety gate. After the increase,
Flagship accumulated 22 R2 connect-timeout episode failures; shards 3 and 4
failed after several hours, shard 5 exhausted all three SQL connection attempts,
and running/cancelled shards 6 and 9 also logged R2 failures. The throttle was
returned to 4. Because lowering the throttle does not stop already-running
tasks, the four later tasks 9-12 were cancelled to restore four actual active
downloaders. Their completed objects remain reusable by size on retry.

The shared R2 helper now allows up to six total curl attempts with a three-
second delay and a 240-second retry ceiling, safely below the 300-second signed
URL lifetime. The existing failed-response rollback test was updated and
passed in the HPC EgoVerse environment. This change applies only to future
shards/retries; active jobs retain the old process code.

CPU snapshot `191727` recorded Flagship at 548/4,404 verified, 32.59 GB
verified and 35.53 GB local, with eight normal `.part` files. The last two-hour
growth was 11.46 GB, but it came from the rejected throttle-8 interval and
cannot justify another increase. The verified mean projects about 262 GB total;
allow roughly 2-3 days at stable throttle 4 plus retries.

Freeform remained at throttle 4 and had six R2 timeout failures across original
shards 0-3, which were still running. It was 92/12,686 verified, 11.72 GB
verified and 16.94 GB local with two `.part` files. Current observed episode
sizes project about 1.6-1.8 TB and the recent rate implies roughly 5-6 weeks.
No Scale retry was submitted while the source shards remained active; once
terminal, retry only their exact shard indices with the hardened downloader.

## Automated download pass at 2026-07-22 11:05 +08:00

Mecka remains complete by content at 4,858/4,858 episodes and
689,389,192,671 verified bytes, with no `.part` files.

Flagship has now run for about 8 hours 19 minutes since submission. Its original
array has three completed, four running, three failed, four intentionally
cancelled, and 34 pending shards. The active set is exactly four tasks under the
restored throttle-4 limit. Failure-line count remains 22, so no new Flagship
connection failure appeared in this interval. CPU snapshot `191837` found
725/4,404 verified episodes, 39,918,816,758 verified bytes, 43,488,362,531
local bytes, and ten in-flight `.part` files totaling 222,691,328 bytes. Local
growth since 09:05 was 7,957,648,746 bytes, about 1.11 MB/s aggregate. The
current verified mean projects about 242.5 GB total and gives a provisional
50-hour remaining-byte ETA if this stable throttle-4 rate holds. Allow roughly
2-3 days plus exact-shard retries.

Freeform still has four running and 124 pending shards after about 8 hours
19 minutes; none is terminal. Failure-line count increased only from six to
seven, still within the live original shards. Snapshot `191837` found
248/12,686 verified episodes, 28,034,138,405 verified bytes, 31,900,700,404
local bytes, and seven in-flight `.part` files totaling 102,023,168 bytes.
Local growth since 09:05 was 14,955,848,554 bytes, about 2.08 MB/s aggregate.
The current verified mean projects about 1.43 TB total; the short-window ETA is
about 7.8 days, so use a provisional 8-12 day range until the first mature
Freeform shard completes.

No retry was submitted. The original arrays already occupy all eight safe Scale
download slots, and retrying terminal Flagship shards now would recreate the
rejected aggregate concurrency. Keep both throttles at four. Retry only exact
terminal shard indices after active slots open, using the hardened downloader;
do not duplicate either whole array.

## Scale small-object bottleneck diagnosis on 2026-07-22

Manifest statistics from CPU job `191898` explain the large source-dependent
throughput gap. Complete Mecka episodes average 141.9 MB, 2,489 frames, and only
34.4 R2 objects; the mean object is 4.12 MB. The currently verified Flagship
episodes average 54.0 MB and 323.3 objects, only 167 KB per object. Freeform
averages 108.8 MB and 833.8 objects, only 131 KB per object. Thus Mecka is
actually larger per episode but packages its data into roughly 25-32 times
larger objects. Scale also has a strong long tail: its median is only 19 objects
per episode while the mean is hundreds.

The downloader signs and launches one curl request per object. Scale therefore
pays far more process, TCP/TLS, HTTP, signature, and R2 request latency per
useful byte, and its many requests amplify connection-timeout probability.
This matches the observed failure pattern: Mecka reached tens of MB/s with only
two isolated HTTP 503 episodes, while Scale remains near 1-2 MB/s and has
repeated connect timeouts. More CPU does not remove this bottleneck. The next
optimization candidate, if needed, is an A/B test of connection reuse or
batched transfers on one failed Scale shard, not another array-concurrency
increase.

## Automated download pass at 2026-07-22 16:06 +08:00

Mecka remains content-complete at 4,858/4,858 episodes and
689,389,192,671 bytes with `.part=0`.

After 13 hours 20 minutes, Flagship has three completed, four running, four
failed, four intentionally cancelled, and 33 pending shards. Snapshot `192101`
found 882/4,404 verified episodes, 46,409,377,409 verified bytes,
51,620,338,577 local bytes, and 12 `.part` files totaling 296,951,808 bytes.
The 5.15-hour local-byte growth since snapshot `191837` was 8.13 GB, about
0.44 MB/s; the current mean projects about 232 GB total and a roughly 4-6 day
remaining range before retries.

Freeform has four running, two failed, one node-failed, no completed, and 121
pending shards. Snapshot `192101` found 382/12,686 verified episodes,
43,720,481,678 verified bytes, 49,206,558,781 local bytes, and 11 `.part` files
totaling 759,832,576 bytes. The same-window local growth was 17.31 GB, about
0.93 MB/s; the verified mean projects about 1.45 TB total and a provisional
17-25 day remaining range before retries.

Failure lines rose from 22 to 67 for Flagship and from 7 to 33 for Freeform.
Terminal failures remain R2 curl error 28 connection timeouts, plus Flagship
shard 5's SQL timeout and Freeform shard 5's Slurm node failure. The new retry
settings do not eliminate the issue: recently launched Flagship shard 14 has
41 failed episodes and Freeform shard 4 has 24, while contemporaneous tasks on
other compute nodes have zero to five. This points to a strong node/network-path
component in addition to Scale's small-object amplification. No retry was
submitted while both arrays already had four active tasks. Retry only terminal
indices after slots open; do not duplicate either full array.

## Automated download pass at 2026-07-22 19:55 +08:00

The node/network diagnosis became conclusive. Flagship failure lines reached
473 and Freeform reached 142. Almost every newly failed Flagship shard from
13-36 ran on `server13`; several exited in 18-21 seconds because even AWS
Secrets Manager DNS resolution failed. Longer failures on the same node show
R2 curl error 6 (`Could not resolve host`). Freeform shards 8-9 failed on the
same node for the same reason. This is an HPC node DNS/path fault, not a bad
episode mapping or byte mismatch.

To stop the original arrays from immediately feeding more work to the faulty
node, pending tasks in both arrays now exclude `server13`. Active Flagship
tasks 37-40 and Freeform task 11 on that node were cancelled; their completed
objects remain resumable. Pending-task throttles were reduced to two for
Flagship and three for Freeform. Four Flagship tasks had already started on
healthy nodes before the lower throttle took effect; the limit will apply as
they finish. Total live Scale concurrency remains eight.

Only terminal shard indices were queued for retry, with proxy variables unset,
the hardened downloader, and `server13` excluded: Flagship retry `192337`
contains shards 3-7 and 9-40 at throttle two; Freeform retry `192338` contains
0-5, 8-9, and 11 at throttle one. Both retries are pending behind the healthy
original work and will byte-verify and skip already completed objects. No full
array was duplicated.

CPU snapshot `192328` found Flagship at 1,749/4,404 verified episodes,
86,652,534,170 verified bytes and 91,514,414,112 local bytes, with 16 `.part`
files totaling 552,214,528 bytes. Growth since 16:06 was 39.89 GB, about
2.96 MB/s, but includes the fault-heavy interval; use a provisional 2-4 day
remaining range after retries. Freeform was 606/12,686 verified,
61,343,447,612 verified bytes and 65,754,309,665 local bytes, with eight active
`.part` paths. Growth was 16.55 GB, about 1.23 MB/s; its current size mean
projects roughly 1.28 TB and a provisional 14-21 day remaining range.

A healthy-node Freeform task then exposed a separate retry edge case: curl can
append a partial metadata response to the successful retry when stdout is a
non-seekable pipe, producing two concatenated XML documents. Metadata responses
now use a seekable temporary file, allowing curl to rewind before retry while
leaving resumable object-file behavior unchanged. The rollback test now also
checks metadata capture and passes on HPC (`FAILED_PART_ROLLBACK_OK`). Pending
retry jobs will load this fix; already-running tasks retain their process code.

## Automated download pass at 2026-07-23 01:08 +08:00

The `server13` isolation is working. Flagship retry `192337` completed shard 3
and is running shards 4-5 on healthy nodes; Freeform retry `192338` completed
shard 0 and is running shard 1. Both retry arrays have zero failure lines and
are correctly skipping byte-verified objects. The original arrays continue on
healthy nodes only. No new retry or concurrency change was needed.

CPU snapshot `192856` found Flagship at 1,920/4,404 verified episodes,
98,141,381,488 verified bytes and 104,818,879,112 local bytes. There are 42
in-flight `.part` files totaling 665,534,464 bytes. Since 19:55, local data grew
13.30 GB, about 0.71 MB/s; the remaining range stays roughly 2-4 days including
the queued exact-shard retries.

Freeform reached 671/12,686 verified episodes, 67,781,052,516 verified bytes
and 74,681,088,858 local bytes, with 13 `.part` files totaling 3,866,624 bytes.
The same-window growth was 8.93 GB, about 0.48 MB/s. At this slower stable-node
rate the provisional remaining range returns to roughly 4-6 weeks. Original
failure counts increased only slightly during this interval, while the fixed
retry jobs remained clean.

## Automated download pass at 2026-07-23 07:09 +08:00

Isolation remains effective: no task returned to `server13`. Flagship retry
`192337` completed shards 3, 4, and 7, is running 6 and 10, and has two terminal
failed shards (5 and 9) containing only three residual R2 SSL reset/timeout
episodes. Freeform retry `192338` completed shards 0-3 and is running 4. Its
active/completed work has only two failure lines total. Do not start a second
retry layer while all eight intended Scale slots are occupied.

CPU snapshot `193183` found Flagship at 2,382/4,404 verified episodes,
124,131,575,356 verified bytes and 129,816,355,803 local bytes. There are 77
in-flight `.part` files totaling 552,210,703 bytes. Six-hour local growth was
25.00 GB, about 1.16 MB/s; current evidence gives roughly 1.5-3 days remaining
including the exact retries still queued.

Freeform reached 1,020/12,686 verified episodes, 101,393,282,672 verified bytes
and 106,348,440,536 local bytes, with 30 in-flight `.part` files. Six-hour local
growth was 31.67 GB, about 1.47 MB/s. The current verified mean projects about
1.26 TB total; use roughly 10-18 days remaining plus final exact-shard cleanup.
No concurrency or downloader change was made this pass.

## Direct versus proxy endpoint A/B on 2026-07-23

CPU job `193425` made ten tiny requests to the same R2 endpoint in each mode,
so proxy traffic was negligible. Direct access succeeded 10/10 with median
total latency about 1.40 seconds. The `hkuhpc.com:7999` proxy succeeded only
7/10; successful requests had median latency about 7.12 seconds and the other
three hit the ten-second TLS timeout. The proxy is therefore about five times
slower for the request-dominated Scale layout and less reliable. Do not route
bulk EgoVerse data through it.

The remaining acceleration target is downloader-side connection reuse, not
more CPU nodes or a proxy. Freeform currently averages hundreds of objects per
episode and the downloader starts a new curl process/connection for each. Gate
any replacement on one identical-shard A/B with exact manifest/object-byte
parity before changing or restarting the remaining array.

## Automated download pass at 2026-07-23 13:12 +08:00

The original Flagship array is now terminal except for completed shard 47;
retry `192337` continues through its queued exact indices. Since two intended
concurrency slots became free, a second cleanup array `193500` was submitted at
throttle one for terminal Flagship shards 5, 9-10, and 41-46. Freeform cleanup
array `193501` was likewise submitted at throttle one for terminal shards 4,
6-7, 10, and 12-14. Both exclude `server13`, disable all data proxies, and use
manifest/byte checks to skip completed objects. They are pending behind the six
active original/retry tasks, keeping the theoretical aggregate cap at eight.

CPU snapshot `193503` found Flagship at 2,669/4,404 verified episodes,
137,254,760,492 verified bytes and 142,893,698,170 local bytes. There are 76
`.part` files totaling 860,692,480 bytes. Six-hour growth was 13.08 GB, about
0.61 MB/s; keep a provisional 2-4 day range including cleanup retries.

Freeform reached 1,322/12,686 verified episodes, 122,299,703,645 verified bytes
and 130,781,188,115 local bytes, with 43 `.part` files totaling 482,975,744
bytes. Six-hour growth was 24.43 GB, about 1.13 MB/s. The current verified mean
projects about 1.17 TB total, giving roughly 12-20 days including retry tails.
No proxy or concurrency increase was used.

## Persistent R2 transport gate and Freeform switch on 2026-07-23

The object downloader now has an opt-in `EGOVERSE_TRANSPORT=persistent` path
using one thread-local HTTPS connection per object worker. It preserves the
existing six-attempt retry, exact `.part` rollback/resume, final byte check, and
atomic rename contract. The curl path remains the default. Mocked connection
reuse and interrupted-resume tests pass on HPC.

The first two-connection persistent smoke, job `193582` on `server14`, was
cancelled after 21:35 because only about 749/4,464 objects and 43.7/243.5 MB
had arrived after roughly 20 minutes. Connection reuse alone did not remove
the per-object response latency. A second smoke, job `193616` on `server56`,
used 16 persistent object workers on the identical Freeform episode
`2026-05-01-17-48-01-502501`. It completed in 457.92 seconds and passed exact
parity: 4,464 objects, 243,532,936 bytes, and zero `.part` files. The cancelled
curl baseline had still not completed after 52:41, so the proven speedup lower
bound is greater than 6.9x. The useful change is persistent connections plus
enough small-object concurrency, not persistence by itself.

All remaining old Freeform tasks in arrays `191447`, `192338`, and `193501`
were then cancelled without deleting data. CPU manifest audit `193640` found
1,362/12,686 episodes verified and 11,324 incomplete across shards 4-127;
verified shards 0-3 were not resubmitted. Continuation array `193648` contains
only shards 4-127, excludes `server13` and `server23`, disables all proxies,
and uses two episode workers with 16 persistent object workers each. It began
at throttle one. Production shard 4 completed in 11:27, resumed and verified a
1,046,636,449-byte episode, and ended with `EGOVERSE_PART_OK`; the array
throttle was then raised to three, for at most 96 persistent R2 connections
across the three live shards.

The first production follow-up confirmed the switch remained healthy. Shards
6 and 7 completed in 9:14 and 14:29, while shards 5, 8, and 9 continued with
zero stderr failures. CPU snapshot `193712` measured 1,378/12,686 verified
Freeform episodes, 135,321,883,303 verified bytes, 142,049,946,839 local
bytes, and 81 `.part` files totaling 1,051,688,960 bytes. With roughly
0.98 TB and 11,308 episodes remaining, the 16-connection object rate suggests
about two ideal days; use a conservative 3-5 day operational ETA until several
fully fresh shards, rather than resumed early shards, complete.

Flagship was moved to the same validated transport rather than maintaining a
second downloader. Old retry arrays `192337` and `193500` were cancelled with
their files left resumable. CPU manifest audit `193717` found 2,765/4,404
episodes verified and 1,639 incomplete across 36 exact shards. Continuation
array `193719` contains only shards 10-12 and 14-46, excludes `server13` and
`server23`, and uses the same two-episode by 16-object-worker persistent
configuration. At throttle one, shard 10 downloaded and verified a new
89,683,892-byte episode and completed the full shard in 2:30 with no errors.
The throttle was then raised to three; tasks 11, 12, and 14 started normally.
About 101.6 GB remained at the preceding snapshot, giving a provisional
12-24 hour Flagship ETA until fully fresh shards provide a stable rate.
