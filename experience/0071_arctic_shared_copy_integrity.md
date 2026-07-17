# ARCTIC shared copy and final integrity gate

Date: 2026-07-18

## Outcome

The official ARCTIC image set is available at
`/data/wentao/datasets/arctic/unpack/arctic_data/data/cropped_images`.
It was copied from the readable shared extraction at
`/data/lingang_data/data1/handdata/arctic/unpack/arctic_data/data` rather than
waiting for the remaining slow upstream downloads.

Final copy gate (job 184883):

- 339 official sequence directories and 3,051 view directories;
- 2,190,652 JPEG files and 127,994,982,876 image-content bytes;
- post-copy `rsync -ani --delete` diff was empty;
- `SHARED_COPY_COMPLETE` exists and the job completed with an empty error log;
- the non-official shared directory `s08/scissors_use_02` was excluded.

The 246 complete upstream archives remain under the download root. All were
already checked against the published archive hashes. CPU job 184879 then
matched 1,599,353 extracted JPEGs (92,454,921,746 bytes) against their official
zip-member CRCs with zero failures. The remaining 93 incomplete `.part` files
were removed after the shared-copy gate passed.

## Annotation and frame audit

CPU job 184866 compared the shared data with the official 339-entry URL list
and with our verified base extraction:

- all 339 official sequences are present;
- `meta` (124 files), `raw_seqs` (1,204 files), and `splits_json` (2 files)
  match the verified official trees exactly by size and SHA-256;
- all nine views are present for every official sequence;
- 337 sequences have identical frame-name sets across all nine views.

Two release/source gaps remain and must not be silently imputed:

- `s01/laptop_use_04/7/00737.jpg` is absent; the checksum-verified official zip
  has the identical member set, so this is an upstream release gap.
- `s08/mixer_use_02/3/00295.jpg` is absent; its archive was not among the 246
  completed downloads, so this one image remains a disclosed uncertainty.

The old proxy downloader 184404 and dependent extractor 183954 were cancelled
only after the shared copy passed. Complete archives were retained; no partial
downloads remain.

## Research judgment

The annotation quality and zero-shot model result are separate. ARCTIC has
coherent calibrated multiview geometry, native side-specific MANO supervision,
validity fields, and subject-disjoint splits, so it is suitable for later
training and materially stronger supervision than the audited EgoDex labels.
The two isolated missing-view images should be excluded by existence checks.

The frozen zero-shot result remains mixed: Flex15 closes the rope residual but
does not consistently improve WiLoR geometry across sides. That result does not
weaken the dataset's training value.
