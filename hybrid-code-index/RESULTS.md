# hybrid-code-index

**Started / finished:** 2026-08-05 · **Status:** done — **hybrid wins; ship
`rrf(dense, stored-BM25)`**. Two assumptions I had been carrying were refuted.

**Question.** `repo-index` was a markdown sidecar with a code afterthought and a
single dense arm. The target is a general-purpose *hybrid* code indexer. Which
lexical arm — stored BM25 or ripgrep at query time — and does fusing actually
beat either alone?

## Setup

`hcindex.py` carries no repo-specific constants; this repo's quirks live in
`.repo-index.json`. Corpus: 11,257 chunks / 672 files under the "all text source
+ docs" rule (≤1 MB, code + docs + config). Prose splits on headings, code on
flat 60-line windows — flat because AST-vs-flat was noise in
`bekko-embedding-bench` (+0.063, p=0.424) and flat needs no per-language parser,
which a general-purpose indexer would otherwise need for every extension it
claims to support.

**Code-aware tokenization** is what makes it a code index rather than a text
index pointed at code: identifiers emit the whole token *and* its
`snake_case`/`camelCase` parts, so `_stacked_simhash_encode` is reachable from
"stacked simhash encode" while an exact-identifier query still scores the whole
token highest.

Three query classes, each with an answer key that already existed for another
purpose — none written to make retrieval look good:

| class | n | answer key |
|---|---|---|
| rediscovery | 5 | the repo's own documented rediscovery failures |
| keyword | 10 | what grep over the same file types returns |
| duplication | 9 | `METHODS.md`'s duplication map |

## Results (hit@5)

| arm | rediscovery | keyword | duplication | total |
|---|---|---|---|---|
| **rrf(dense, bm25)** | 5/5 | 10/10 | 9/9 | **24/24** |
| bm25 | 5/5 | 10/10 | 8/9 | 23/24 |
| dense | 4/5 | 9/10 | 9/9 | 22/24 |
| rrf(dense, rg) | 5/5 | 10/10 | 7/9 | 22/24 |
| rrf(all 3) | 5/5 | 10/10 | 7/9 | 22/24 |
| rg | 4/5 | 10/10 | 3/9 | 17/24 |

Identical on the `no-json` corpus except the rg arm (17 → 19).

**1. Stored BM25 beats ripgrep as the lexical arm, decisively.** 23/24 vs 17/24,
and the gap is almost entirely duplication (8/9 vs 3/9). ripgrep returns a *set*;
"find me a file like this one" is a ranking question, and no amount of term
counting recovers a ranking it never produced.

**2. Fusing beats both arms alone** — but only the right pair. dense 22, bm25 23,
fused **24**. The two arms fail on different queries, which is the precondition
for fusion helping at all.

**3. Adding a third arm *hurts*: 24 → 22.** `rrf(all 3)` is worse than
`rrf(dense, bm25)`. RRF is unweighted, so a weak arm votes as loudly as a strong
one, and rg's duplication ranking is close to noise. **More retrieval arms is not
monotonically better** — this is the one result here I would not have predicted.

**4. The dense arm alone was never the right product.** It loses a rediscovery
query and a keyword query that plain BM25 gets. `lexical-kb-phase0` said as much
already ("BM25 on whole documents matches the dense-embedding ceiling"); shipping
dense-only was ignoring a finding this repo had already written down.

## Refuted: the .json dilution

79% of the corpus (8,874 of 11,257 chunks) is generated `.json` results data. I
expected a repeat of the `outputs/` pollution result and pre-wrote a variant to
catch it.

**It is inert.** `rrf(dense, bm25)` scores 24/24 with and without it. Only the rg
arm improves when it is removed (17 → 19). So **no build-time exclusion for
`.json` is warranted**, which supports keeping exclusions at query time.

The two cases differ in a way worth keeping: `run_NN.md` model output is
*topically on-subject prose* competing directly with real answers, while JSON is
lexically alien — the encoder maps it somewhere no natural query goes. Volume
alone does not predict pollution; **similarity to real queries does.**

It is not free, though: BM25 postings inflate to **6.36 MB / 138,685 terms** with
JSON, against ~1 MB without. Storage cost of the stored-lexical arm is strongly
corpus-dependent, not a fixed tax.

## Rebuild cost, and why incremental was necessary

A full rebuild is **537 s** on the all-text corpus (20 chunks/s, 4 threads) —
untenable for a CI job on every push to `main`.

`incremental()` hashes each chunk's text and encodes only the new ones:

| | full | 1-file change |
|---|---|---|
| md+py | 100.3 s | 0.2 s (516×) |
| all text | 537.1 s | 0.2 s (2735×) |

**Verified bit-identical to a full rebuild** (max abs delta 0.000e+00), not an
approximation traded for speed. Two properties make that true, both established
elsewhere in this repo: the encoder is per-chunk independent, and remex
quantization is data-oblivious.

**The lexical arm cannot be incrementalized the same way.** BM25's IDF shifts for
every term whenever any document is added, so it is refit wholesale — affordable
only because fitting is 5.2 s against 537 s. Any component fitted *on* the corpus
(PCA, k-means, ITQ, PQ codebooks, IDF) breaks the equivalence that makes
incremental safe. That distinction is the deciding one for whether a retrieval
component can be incremental at all.

## The cost incremental does *not* fix

The workflow **commits** the index. Binary blobs do not delta-compress, so every
rebuild stores a complete new copy:

| | per rebuild | 200 rebuilds |
|---|---|---|
| dense codes | 1.00 MB | 200 MB |
| BM25 postings | 6.36 MB | 1.27 GB |

Incremental builds make each rebuild cheaper to *produce* and change none of
that. The fix is to stop committing the artifact and publish it as a release
asset — machinery this repo already has and has proven (`repo-index-mirror`,
pinned sha256, round-trip verification). Not done here.

## Cost

580 s dense encode (11,257 chunks), 5.2 s BM25 fit, ~15 min total per full bench
run; two runs. The first run was killed and restarted after adding memoization —
`rg` and query embeddings were being recomputed once per arm, ~1,280 redundant
subprocess calls.

## Reproduce

```bash
python3 hybrid-code-index/bench.py             # arm comparison
python3 hybrid-code-index/bench_incremental.py # rebuild cost + equivalence
```

## Not done

- **Graduating `hcindex` into `repo-index/ask.py`.** The winning arm is measured
  but `ask.py` still ships dense-only.
- **Moving the artifact out of git**, per the table above.
- **A second-repo check.** Every number here is from one repo, and a
  general-purpose indexer that has only been measured on its own corpus is one
  distribution short of the claim.
- **History/tombstone indexing** — searching content that was *deleted*, which no
  current-state index can contain. Not testable here: 73 commits, 1,319 deleted
  lines, and **zero** files deleted-and-never-restored. `remax` would work as a
  testbed (144 commits, 10,042 deleted lines, 21 files deleted and never
  restored) and has an answer key in `bench/results/*.md`, written under its own
  "delete the driver, never the record" convention.

---

## Is it practical? A cost sheet

Measured by building a real persisted index for `oaustegard/remax` (0.99 MB of
source across 102 files) — live tree plus tombstones, 858 chunks. Every number
below is from `build.py`, not an estimate.

| | |
|---|---|
| build, cold | **37 s** (corpus 0.5 s, BM25 fit 0.3 s, dense encode 36 s @ 24 chunks/s) |
| build, one file edited | ~1 s (content-hash reuse; only the changed chunks re-encode) |
| artifact | **353 KB = 35% of source** |
| query, cold process | **~155 ms** (load 70 ms, decode 81 ms, score 5 ms) |
| quality | **12/12**, identical to the fp32 benchmark |
| encoder dependency | **157 MB**, one-time, shared across every repo |

### Artifact breakdown — the lexical arm dominates

| component | size |
|---|---|
| dense codes (remex 2-bit, 384-d) | 93 KB |
| **BM25 postings** | **245 KB** |
| pointers + hashes + meta | 15 KB |

**The lexical index is 2.6x the dense one.** That inverts the usual intuition
that embeddings are the expensive part, and it scales with *vocabulary* rather
than chunk count — on this repo's own JSON-heavy corpus the postings hit 6.36 MB
across 138,685 terms, where the dense side stayed at 1 MB. If an index gets too
big, the lexical arm is the thing to look at first.

### The 200 ms bug I had already written a rule about

First query measurement was ~400 ms, with 270 ms of it attributed to "decode".
Isolating it: **the actual decode of 858 vectors is 22 ms; 200 ms was
reconstructing the Haar rotation** (Householder QR, O(d³) at d=384) on every
query.

That is the same defect diagnosed earlier in this repo's `bekko-embedding-bench`
against `remax_kb` — `_stacked_simhash_encode` rebuilding rotations per query,
87% of query time — and the same one `repo-index` fixed by committing
`rotation.npy`, with a `METHODS.md` entry saying to store the transform rather
than regenerate it. I wrote the rule and then did not apply it in the next
artifact I built.

The fix here is better than storing it: **`rotation="rht"` builds in 6.2 ms
instead of 171 ms and costs 0 bytes**, versus 576 KB to persist a Haar matrix.
remex measured RHT as retrieval-indistinguishable from Haar (−0.0001 ± 0.0013,
this repo's own experiments#11), and that was re-verified rather than assumed —
the persisted 2-bit + rht artifact scores 12/12, matching the fp32 benchmark
exactly. Query dropped to ~155 ms.

### Verdict

**Practical for a repo of this size, with one caveat that dominates everything
else.** 353 KB is committable, 37 s is a CI step, 155 ms is interactive, and
2-bit compression costs nothing measurable.

The caveat is the **157 MB encoder to search 1 MB of source** — 160x the thing
it indexes. That is only reasonable because it is amortized: one download serves
every repo on the machine, and it is already pinned and mirrored. As a
standalone per-repo tool it would be absurd; as shared infrastructure it is fine.

Scaling is linear in chunks at ~24 chunks/s on 4 CPU threads: this repo's 11,257
chunks took 580 s. A 100k-chunk repo would be ~70 min for a cold build, which is
why the incremental path is not optional at that size — and why the artifact
should be a release asset rather than a committed blob that stores a full copy
per rebuild.
