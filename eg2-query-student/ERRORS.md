# ERRORS — eg2-query-student

What broke, what it cost, and what changed because of it.

1. **A chained background job hit the 2-hour cap counting its own wait.** The
   165K-target run was launched as `tail --pid=<control> -f /dev/null; python3
   a3_distill.py ...` so it would start when the control run finished. The
   harness cap runs from launch, so the 25 minutes of waiting counted, and the
   run died at step ~2,400 of a 2,572-step epoch. Checkpoints were per epoch
   only, so 1.5 hours of CPU training were lost. Fixed: `a3_distill.py
   --save-every` writes model, optimizer state and position to
   `out/a3/<tag>/last` every 400 steps and resumes from it (tested by deleting a
   finished epoch and resuming at step 15 of 16: same real_test overlap,
   0.4757). Launch long jobs directly, never behind a wait.
2. **GitHub's web uploader refused the GPU shards.** Four 23 MB files, under
   the documented 25 MB per-file limit, failed with "The file is too large and
   cannot be uploaded", and so did the same files split into 8 MB parts. Oskar
   published them as release assets instead (`gh release create
   eg2-targets-20261020 ...`); each downloaded file matched the sha256 digest
   GitHub records for the asset. For browser-built files over a few MB, ask for
   a release from the start.
3. **The uploading-files skill could not create its branch.** `POST
   /git/refs` returns 403 from a CCotw session. `git push origin
   main:refs/heads/upload-<id>` created it instead.
4. **The ranking-loss A3 variant barely trained and was abandoned.** leaf-mt
   with `--rank-weight 1 --tau 0.01` reached 0.367 real_test after one epoch
   from a 0.364 start, against 0.402 for cosine alone over the same epoch. The KL
   term was about 2.0 against a cosine term of about 0.15, so it dominated the
   loss, and gradient clipping at norm 1.0 shrank every step. Not retuned (a
   weight near 0.05 would balance the terms); the cosine runs went ahead.
   `results/a3_mdbr-leaf-mt_rank1_abandoned.json`.
5. **A container restart stopped leaf-mt A3 after epoch 1.** It resumed from the
   epoch-1 checkpoint, with AdamW's moment estimates reset (they were not saved
   then; they are now, per item 1).
6. **Synthetic-query collection split prompts on blank lines** that post texts
   also contain, and crashed. It now finds the `[i] ` markers with a regex.
7. **The SwiftShader smoke test called a borrowed helper with the wrong
   argument** (`wait_for_server` takes a URL, not a port). One rerun.
8. **Post hydration ran at 12 posts/s sequentially**; six requests in flight
   brought it to about 76 posts/s.
