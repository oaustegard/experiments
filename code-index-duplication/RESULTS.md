# code-index-duplication

**Started / finished:** 2026-08-05 · **Status:** done — **positive, shipped**.
`repo-index` now indexes `.py` alongside `.md`, and grew `--file`.

**Question.** `repo-index` indexed only markdown. Should it index code? This
repo's own `bekko-embedding-bench` says dense retrieval does *not* beat grep at
NL→code file localization (r@5 0.656 vs 0.596, n=59, not significant) — so the
usual reason to index code is already measured as a non-reason here.

But localization is not the failure this repo actually has with code.
`METHODS.md`'s **duplication map** records three independent reimplementations of
one bench harness, plus three more near-identical or reuse pairs — all found by
hand, after the fact. That is "I never looked" in code form, and it is the case
`repo-index` exists for.

**Why the answer key is fair:** the duplication map was written before this
experiment existed, for an unrelated purpose (a code-hygiene sweep). It was not
authored to make retrieval look good.

## Setup

- **Corpus:** 831 flat 60-line windows (stride 45) from 190 `.py` files. Flat
  rather than AST because `bekko-embedding-bench` measured AST-vs-flat as noise
  (+0.063, p=0.424).
- **Encoder:** `bekko-embedding-v1-a8m`, the one `repo-index` already pins.
- **Metric:** hit@5 — does a *known sibling* of the query file appear in the top
  5 files? Chance is ~2.6% per sibling over 190 files.
- **Exclusion:** the query file's whole **directory**, matching what `ask.py
  --file` ships (see "What broke" below).

Two query modes, corresponding to different moments:

| mode | the moment |
|---|---|
| `code` | you have a draft and ask "does something like this already exist?" |
| `nl` | you describe what you want in words, before writing |

Two corpus variants, because the path header is a live confound: `bench.py`
appearing in two paths could drive a match on filename alone. This had to be
controlled, not assumed away — `bekko-embedding-bench` measured path-only
retrieval at r@5 0.304 on a related task, real but not dominant.

## Results

| arm | hit@5 |
|---|---|
| dense, `code` query, with path header | **9/9** |
| dense, `code` query, **content only** | **9/9** |
| dense, `nl` query, with path header | 8/9 |
| dense, `nl` query, content only | 8/9 |
| grep, ideal `def` name from the query file | 8/9 |

**The path header does not carry the result.** Content-only scores identically
in `code` mode, so this is content matching, not filename matching. That was the
main thing that could have made the headline number hollow.

**Dense ties grep, it does not beat it.** 9/9 vs 8/9 at n=9 is not a result;
read it as "no worse", and note the arms need different things. Grep's arm was
handed the most distinctive `def` name *out of the query file* — it needs a
draft with a distinctive name in it. The dense `nl` arm needs no draft at all,
which is the only arm that works before you have written anything.

Sibling ranks in the `code` arm are 1–3, so the hits are decisive rather than
scraping in at 5.

### The one `nl` miss is the index being right and the key being stale

Querying "shared helpers … retry with backoff, chunking, atomic checkpoint save
and load" returns `_lib/pipeline.py` and `_lib/textnorm.py` at ranks 1–2 instead
of the recorded sibling. Those *are* the correct answer: the same duplication map
records that the generic parts were extracted into `_lib/` and both files now
re-export. Scored as a miss because the key names the sibling; in use it is the
better answer.

## What broke

1. **The harness retrieved itself.** `run.py` contains all nine NL queries
   verbatim, so it ranked top-5 for 4 of 9 NL queries on the first run.
   Excluding this directory moved content-only NL from 6/9 to 8/9. An evaluation
   script that embeds its queries is part of the corpus it searches.

2. **The first reported number described a configuration the tool does not
   ship.** Excluding only the query *file* scored 9/9 code / 8/9 nl — but in real
   CLI use, same-directory neighbours filled every slot (querying with
   `muninn-embedder-bakeoff/bench.py` returned four files from its own
   directory). `--file` therefore excludes the query file's directory, and the
   harness was changed to match before any number was written down. Same headline
   either way here, but the discipline is the point: report what the tool does,
   not the most flattering configuration of the harness.

3. **The integration benchmark's grep arm was restricted to `*.md`.** After
   adding `.py` to `repo-index`, keyword agreement appeared to fall 10/10 → 7/10.
   All three "regressions" were the index returning the *definition* instead of a
   prose mention — `ascii_fold` → `_lib/textnorm.py`, `GRID_VERSION` →
   `remex-vs-higgs-ablation/grids.py` — which an md-only grep arm cannot score.
   With a matched arm it is 9/10. **A baseline restricted to a narrower corpus
   than the system under test will report improvements as regressions.**

## Effect on `repo-index`

Adding `.py` (0.14 → 0.27 MB index):

| | md only | md + py |
|---|---|---|
| rediscovery hit@5 | 5/5 | 5/5 |
| keyword agreement with grep (matched arm) | 10/10 | 9/10 |
| `ascii_fold` | rank-5 prose mention | **`_lib/textnorm.py:1`** |
| `GRID_VERSION` | METHODS.md prose | **`grids.py:1`** |

Checked for regression deliberately, because the immediately preceding change to
this tool found that 20% of the corpus was junk that the vague-query benchmark
could not see.

## Cost

~4 min per full run (2 corpus variants × 831 chunks encoded on 4 CPU threads),
4 runs. No credentials, no network beyond the pinned encoder. `results.json`
holds per-query detail.

## Reproduce

```bash
python3 code-index-duplication/run.py
```
