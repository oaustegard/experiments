# account-routing-tier

**Started / finished:** 2026-08-06 · **Status:** done — **premature. Routing
reaches 87–90% at k=3 of 9 repos, but the whole account index is 8.54 MB against
a 157 MB encoder the client already downloads, so there is nothing to route
around. Kept as a measurement of when partitioning *would* pay.**

**Question.** A whole-account index partitioned per repo only works if a client
can decide *which* partitions to fetch without downloading them all. That needs a
coarse tier: per-repo summary cards, always loaded, that pick the top-k repos.

The failure mode is unforgiving. A flat index that ranks badly still contains the
answer further down; a coarse tier that routes wrong makes the answer
**unreachable**, and returns a confident result from the wrong repo instead.

## Setup

9 repos on disk, `.json` excluded (measured inert, and 79% of one repo's corpus).

| repo | chunks | share |
|---|---|---|
| sklearn-bench | 12,097 | 46.7% |
| claude-skills | 8,048 | 31.1% |
| experiments | 2,111 | 8.2% |
| muninn-utilities | 1,715 | 6.6% |
| remax / remex / remax_kb | 682 / 594 / 423 | 6.5% |
| claude-workspace / claude-container-layers | 195 / 34 | 0.9% |
| **total fine** | **25,899** | |

**Storage is not the constraint.** The coarse tier is 26–90 chunks — **0.1% to
0.35% of the fine tier**, a few KB of 2-bit codes held permanently to route among
nine repos.

30 queries, all about *internals* (`ascii_fold`, the CSR builder, NVFP4 dequant,
srht construction, sklearn's CSR `indptr`), never about front matter — so a card
has to route on topical similarity to a summary that does not contain the answer.

Gold is an **oracle**: flat RRF over all 25,899 chunks, winning chunk's repo.
Derived rather than hand-labelled because hand-labelled keys had already broken
three times in this line of work. See the caveat on that below — it broke here too.

## Results (routing recall@k)

| card source | @1 | @2 | @3 | @5 | coarse chunks |
|---|---|---|---|---|---|
| front matter (README/CLAUDE.md/AGENTS.md + tree) | 47% | 60% | 73% | 87% | 64 |
| **content (repo-level tf-idf + module paths)** | 50% | 73% | 80% | 93% | **26** |
| **both** | **53%** | **77%** | **87%** | **97%** | 90 |
| both, large repos split into per-directory cards | 47% | 77% | 90% | 97% | 329 |

### A README describes identity; routing needs inventory

Content cards beat front-matter cards at every k **using 2.5x fewer chunks**.

The diagnosis came from a failed fix. Four of eight initial misses wanted
`sklearn-bench`, and its `CARD_FILES` entry was missing because the list only
held `README.md` while scikit-learn ships `README.rst`. That looked like the
cause; it was not. With the README fully in the card, recall@1 went **47% → 43%**
and the same four queries still missed.

scikit-learn's README contains **zero** occurrences of "gradient boosting",
"one-hot", "cross validation", "sparse" or "estimator", while its tree has 296
files matching "sparse" and 311 matching "estimator". It is badges, install
instructions and links. Front matter says what a project *is*; a router needs to
know what it *contains*.

### Splitting large repos backfires at low k

sklearn is 46.7% of the corpus and got the same 400-term card as a repo 0.1% its
size — 0.033 card-terms per chunk against 11.8. Emitting one card per top-level
directory for large repos fixes that imbalance and **makes ranking worse**:
content @1 **50% → 37%**, both @1 **53% → 47%**, for 3.7x the cards, buying only
+3 points at k=3.

The mechanism is visible in the misses, where `experiments` and `claude-skills`
occupy the top-3 almost everywhere. Ranking a repo by its **best** card means
more cards is more draws from the score distribution, so a split repo's maximum
rises for reasons unrelated to relevance. Any per-repo aggregation over a
variable number of cards needs a correction for card count; taking the max does
not have one.

## Verdict: the routing tier solves a problem this account does not have

Measured after the fact, which is the wrong order and is the point of this
section. The **entire** account index — 9 repos, 25,904 chunks, scikit-learn
included — is:

| | |
|---|---|
| dense codes (remex 2-bit, 384-d) | 2.37 MB |
| BM25 postings (2.6x, measured ratio) | 6.17 MB |
| **total** | **8.54 MB** |
| the encoder already required | **157 MB — 18x larger** |
| flat scan over all 25,904 chunks | **7.6 ms/query** |
| decode, once per process | 220 ms |

So routing exists to avoid fetching 8.5 MB in a system that already downloads
157 MB, where scanning everything costs 7.6 ms. **Download the whole account
index and scan it flat.** No partitions on demand, no 13% silent miss, no
confidence gate to tune.

Routing earns its place when the index outgrows what a client will hold —
somewhere past ~1M chunks, roughly 40x this account. That is a monorepo or an
org, not a personal account. The work below stands as a measurement of *when*
partitioning would work, not as something to deploy now.

**The process failure is the reusable part: I measured the accuracy of an
optimization without first measuring whether the thing it optimizes is
expensive.** The partition-size question was answered in one command, after
three 18-minute encodes spent tuning card construction. Cost the baseline
before improving on it.

### Original verdict (routing considered on its own terms)

**87% at k=3 means 13% of queries land in no fetched partition, silently.** That
is not acceptable as a default, and pushing to k=5 (97%) means fetching over half
the partitions — most of the way to downloading everything, which defeats the
architecture at 9 repos. The ratio improves with scale (k=3 of 50 repos is 6% of
partitions, not 33%), but confusability also rises with more repos, and that is
unmeasured here.

The design that makes this usable is **confidence-gated escalation**: route to
top-3, and if the best fine-tier score is weak, widen the fetch. That converts a
silent wrong answer into extra latency, which is the right trade for the failure
mode described at the top.

## What broke

1. **A confidently wrong diagnosis.** I attributed 4 of 8 misses to the
   `README.md`-only card list, fixed it, and recall did not improve — @1 fell 4
   points. The extension bug was real and worth fixing; it was not the cause.
2. **The harness was inside the corpus it measures — the fourth instance here.**
   `run.py` lists all 30 queries verbatim and lives in `experiments`, biasing the
   oracle toward it and invalidating the fine-tier cache on every edit. It had
   already been diagnosed in `code-index-duplication` (harness retrieved itself
   on 4 of 9 queries) and guarded in `hybrid-code-index`. **Knowing a failure
   mode by name did not prevent reproducing it twice more.** The fix that would
   is making self-exclusion a default in `hcindex.build_corpus` rather than
   something each harness remembers — noted, not done.
3. **The oracle is not reliable gold.** "container layer composition and cached
   restore from releases" routed to `claude-container-layers` at rank 1 and was
   scored a **miss** because flat search preferred `claude-skills`. Agreeing with
   flat retrieval is not the same as being correct, so every number here is
   "agreement with a flat baseline", not accuracy.
4. **Self-pollution is structural, not incidental.** `experiments` is both an
   indexed repo and where these writeups live, so each session adds text in the
   exact vocabulary of the next session's queries. "BM25 plus RRF fusion inside
   the kb reader" was aimed at `remax_kb`; the oracle picked `experiments`,
   because `experiments` now genuinely does discuss BM25 and RRF more than
   anything else on disk. Cross-repo evaluation on this account will keep
   drifting this way.
5. One query is invalid here: "tests for the sparse postings list" targets a file
   **deleted** from remax, which is not in the fine tier (no tombstones in this
   run).

## Cost

Three full encodes of ~26k chunks (~18 min each) before the corpus stabilised —
each earlier run changed the corpus and so missed the content-hash cache. With
the harness excluded the cache holds and card iterations cost seconds.

## Reproduce

```bash
python3 account-routing-tier/run.py   # expects the 9 repos under /home/user
```
