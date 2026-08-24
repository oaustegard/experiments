# Errors

What was wrong during this survey, how it was caught, and which direction it
pushed the conclusion. The base rate matters more than any single entry: ten
errors, of which one reached a merged commit and none changed a published
number.

## Before the writeup

**Assumed the org API would answer.** `GET api.github.com/orgs/hyparam/repos`
returned 403, because CCotw sessions are bound to their configured
repositories. `mcp__github__search_repositories` with `org:hyparam` works
globally and produced `repos.json`. Cost: one call. Direction: none, the inventory was
correct once fetched.

**Assumed `hypvector` was on GitHub.** The demo README links
`github.com/hyparam/hypvector` and `demos/hypvector/package.json` depends on
`hypvector@0.2.2`, so I ran `git clone` on it along with `hypgrep` and
`hypstore`. All three prompted for auth. Checking npm instead returned the full
unminified `src/`. Direction: without the npm check I would have written the
survey from a demo shell of four React files and the published README, and every
mechanism claim in it would have been a paraphrase of marketing copy.

**Nearly reported the repo inventory from a truncated tool result.** The
`search_repositories` call returned 140 KB and was persisted to a file rather
than shown. The first parse printed 20 rows, then crashed on `KeyError: 'language'` for a
repo with a null language. Twenty of 27 rows plus an exception is not a
listing; re-running with `.get()` gave all 27. Direction: I would have
under-reported the org by seven repos, all of them zero-star and three of them
2026 work: `collectivus`, `squirreling-mcp` and `s3collab`.

## Expectations the sweep overturned

**Expected the default `rerankFactor` to be adequate.** Their README's headline
number is 91% recall at 156k real embeddings, and I set up the sweep expecting
to confirm it. The default measured 50% on this corpus. Their own README already
reports 18% at that setting on a 1M synthetic set, which I had read and not
connected to my own configuration. Direction: had I run only the default and
matched it against their headline, I would have reported a 41-point recall gap as
a finding about hypvector rather than about synthetic Gaussian clusters.

**Ran `probe=1.0` expecting it to help.** It did not: 50% → 49% at
`rerankFactor: 10`, and 86% → 82% at 50, at 2.4x the time. Their `constants.js`
says exactly this and I had read the comment before running the sweep. Direction:
none — the run confirmed their claim rather than mine, which is the useful
outcome, but the honest account is that the sweep tested a claim I had already
been told and half-believed rather than one I derived.

## During the hysnappy benchmark

**Called it "hand-written WASM" in the merged writeup.** The blog says "no
dependencies, not even memcpy, and definitely no emscripten", which I read as
hand-authored WAT. `Makefile:19-25` is one `clang --target=wasm32 -O3 -nostdlib
-Wl,--export-all -Wl,--no-entry` over 674 lines of C. The `memcpy` line means
no *libc* memcpy: `-nostdlib` removes it, so `c/uncompress.c:5-37` defines
`memcpy` and `memmove` as byte loops. Caught by cloning the repo instead of
trusting a blog paraphrase. Direction: the corrected description is more
useful, since "compile C to a freestanding wasm32 target with clang" is a
recipe we could follow and "hand-written WASM" is not.

**Reported a decompression figure from an unwarmed loop.** My first ad-hoc
head-to-head gave 1,232 MB/s for hysnappy on the json-ish corpus, against
4,653 MB/s once each timing was warmed. I nearly wrote the first number down.
Caught by re-running with a checksummed sink and per-sweep printing when a
3.8x swing between two runs of the same code looked wrong. Direction: would
have understated hysnappy's decompression advantage by about 4x, i.e. reported
2.7x when the answer is 8x.

**Measured the warm-up effect in-process, where it cannot show up.** The first
version allocated a fresh `snappyUncompressor()` per trial inside the main
benchmark and reported a 1.2x cold spread. That measures nothing: V8 has
already optimized the shared code paths by then, and a new closure does not
reset the JIT. Caught because the in-process number disagreed with what five
separate `node benchmark.js` invocations had shown. Fixed by spawning
`hysnappy_cold.mjs` in a fresh process per trial. Direction: would have
reported "no warm-up effect" when there is a 1.35x one.

**Ran the warm-up comparison at n=5 and believed the first result.** Two
consecutive 5-trial runs gave opposite orderings — 3.5x cold spread against
1.4x warm, then 1.2x against 1.6x. I had already written the first into
`NOTES.md`. `recheck.py` flagged the figure as unbacked by the artifact, which
is what sent me back to it. At n=15 per arm the medians separate cleanly
(3,201 vs 4,334 MB/s, Mann-Whitney one-sided p≈7e-05) and reproduced at 1.36x
and 1.35x across two independent runs. Direction: at n=5 the sign of the
reported effect was a coin flip. The max/min spread statistic never stabilised
at any n and was dropped in favour of the median shift.


## After the first push

**Nine of eleven cited line numbers in `NOTES.md` were wrong.** I wrote the
notes from memory of files read earlier in the session rather than from a grep,
and every citation was off by 1 to 5 lines: `cluster.js:28-32` for a throw at 26,
`chunks.js:52-62` for a check at 55, `heap.js:38-49` for a tie-break at 36,
`rerank.js:56-64` for a comment at 52, `types.d.ts:24-45` for an interface at 28.
Caught by grepping each anchor string before committing. Direction: none of the
prose claims were wrong, only the pointers — but a pointer that lands four lines
off is worse than no pointer, because it looks checked. This is why
`extract_evidence.py` anchors on literal content rather than line numbers, and
why `recheck.py` verifies every citation lands inside the file it names.

**The first commit shipped a writeup with no artifacts under it.** PR #62's
first push contained `README.md`, a 40-line probe script, `package.json` and a
`.gitignore`. Every number in the README came from stdout that existed only in
the session transcript: no `results.json`, no `run.log`, no `recheck.py`, no
`ERRORS.md`, no source evidence for a library whose repo is private. Oskar
caught it — "Feels like the experiment record is missing a lot of actual work
files?" The repo's own `README.md` names `remex-vs-higgs-ablation/` as the
reference shape and it carries all four. Direction: no number was wrong, and the
rerun reproduced the recall column exactly, which is the only reason the fix was
cheap. Had the probe not been seeded, the committed table would have been
unreproducible and the first push would have destroyed the result.

## Unverified

- **The synthetic corpus is not real embeddings.** 64 isotropic Gaussian
  centers at σ=0.9 is a harder case for sign-bit codes than sentence-transformer
  output. Every recall number here is a lower bound on what real vectors would
  give, and the README says so. Nobody has run this against a real corpus.
- **Nothing here was measured over a network.** All timings are local-file. The
  entire architectural argument is about HTTP range requests, and their published
  numbers are the only evidence for the network behaviour.
- **Their published benchmark table is quoted from their README.** Nothing in
  it was independently checked. The WildChat comparison mixes local compute
  against live cloud round-trips in one latency column, which their footnotes
  disclose; the dollar column is the part worth trusting.
- **The borrows are proposals.** Nobody has built an IVF layer for `remax_kb`.
  The claim that physical cluster reordering lifts its ceiling rests on
  hypvector's measurements; we have none of our own.
