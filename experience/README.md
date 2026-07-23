# Experience

This folder is the repo memory. Every non-trivial experiment, failure, data
finding, environment fix, submodule/Git incident, or engineering decision gets
one short note here.

`docs/` is for the small current status/protocol/report set. `experience/` is
for what actually happened: plans, commands, errors, fixes, paths, scores, and
next actions.

The numbered files intentionally stay in one flat, append-only sequence. Many
plans, code comments, and later notes link to their exact paths, so physically
splitting old notes into `datasets/`, `models/`, or `evaluation/` would create
link churn without improving the evidence itself. Use the stage timeline and
topic map at the top of `INDEX.md` as the two supported views.

Before new work:

1. Read `experience/INDEX.md`.
2. Reuse the closest previous command/config.
3. Do not repeat a failed path unless the note says what changed.

After new work:

1. Add a numbered note at `experience/NNNN_short_slug.md`.
2. Record command, data split, backend commit, result/error, fix, and next
   action.
3. Link it from the chronological section in `experience/INDEX.md`; update the
   topic map only when a genuinely new research area appears.
