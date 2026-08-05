# history-tombstone-index

**Started / finished:** 2026-08-05 · **Status:** done — **positive; tombstones
add what a disciplined working tree structurally cannot hold.**

**Question.** A current-state index cannot contain code that no longer exists.
That sounds like an automatic win for indexing history, but it is only a win if
the *knowledge* left with the code. `remax` is the hard case, because its
CLAUDE.md carries an explicit convention:

> *A measured rejection is an asset — delete the driver, never the record.*

When apparatus is removed there, a prose writeup stays behind in
`bench/results/*.md`. If that convention works, tombstones are redundant and the
honest answer is a negative result. So the question is not "can history find
deleted things" — trivially yes — but **what is the marginal value over a
working tree that already documents its rejections**.

`oaustegard/experiments` could not answer this: 73 commits, 1,319 deleted lines,
and **zero** files deleted-and-never-restored. `remax` has 144 commits, 10,042
deleted lines, 17 true deletions, and an answer key its author wrote for an
unrelated purpose.

## Setup

- **Target:** `oaustegard/remax` @ `edccab5`.
- **Working tree:** 720 chunks / 101 files.
- **Tombstones:** 138 chunks / 17 deleted files, recovered at their last living
  revision (`git show <deleting-commit>^:<path>`). **+19% corpus.**
- Each tombstone chunk is headed
  `<path> [DELETED <date> in <sha>: <commit subject>]`. The subject is often the
  highest-signal part — remax's own read *"Delete the ten Nemotron/NVFP4 bench
  drivers, keep every conclusion"*.
- **Retrieval:** RRF(dense, stored-BM25), the configuration `hybrid-code-index`
  measured best (24/24).

Two query classes, 6 each, predicted to behave differently:

| class | question shape | gold |
|---|---|---|
| existence | "was X ever tried, and what happened?" | the surviving record **or** the deleted file — either answers it |
| mechanism | "how was X implemented?" | **only** the deleted file — "a CSR builder was built" does not tell you how |

## Results (hit@5)

| corpus | existence | mechanism | total |
|---|---|---|---|
| current-only | **5/6** | **0/6** | 5/12 |
| tombstone-only | 4/6 | 6/6 | 10/12 |
| **current + tombstone** | **6/6** | **6/6** | **12/12** |

**1. The convention works — for existence.** 5/6 from prose records alone, with
no access to history at all. `BM25_SKETCH.md` answers "has anyone tried a sparse
BM25 path?" completely. A repo that writes its rejections down really does keep
that knowledge.

**2. It cannot work for mechanism: 0/6.** A record is prose about a *verdict*.
"The encoder, its CSR-builder, an inverted-index streaming variant, and a
BEIR/NFCorpus benchmark were all built" tells you the thing existed and lost —
not what its signature was, how it batched, or what its tests asserted. That
information left with the file, and no amount of writeup discipline retains it
short of pasting the code into the writeup.

**3. The arms are complementary, not competing.** Neither dominates: the working
tree wins existence 5–4, tombstones win mechanism 6–0, and the union is 12/12 —
strictly better than either alone. Tombstones are an *addition* to a current-state
index, not an alternative to one.

For **+19% corpus**, that is a good trade, and the cost is bounded in a way the
full-history alternative is not: only 17 files ever qualified out of 144 commits.

## What broke

**Relocations look exactly like deletions, and name-based detection is wrong.**
The first run reported 21 "deleted" files. Five were *moved*, not deleted — the
`src/remax/bench/*` group relocated to `bench/*` under the commit "Stop shipping
the benchmark harness inside the wheel". They still exist, so indexing them as
tombstones put live content into the "gone" corpus and inflated it by 27%.

It surfaced as an anomaly rather than by inspection: `current-only` scored 1/6 on
mechanism, and the single hit was `bench/crossover.py` — a *live* file matching a
query whose gold was the supposedly-deleted `src/remax/bench/crossover.py`. A
current-state index scoring on a mechanism query was the tell that the answer key
was wrong.

Fixed by detecting relocation **by content**, not path or basename: a dead file
whose non-trivial lines are >50% present in any live file is a move. Basename
matching would have been the obvious implementation and is not sufficient —
`src/remax/bench/__init__.py` shares a basename with `src/remax/__init__.py` but
is genuinely gone, and content correctly keeps it.

The correction *strengthened* the result (mechanism 1/6 → 0/6 for current-only)
while removing an artifact, which is the direction that should increase
confidence rather than decrease it.

## Caveats

- **n=12, one repo, and I wrote the queries knowing the answers.** Same caveat as
  `repo-index`'s original 5-case evaluation. The mechanism/existence *split* is
  the load-bearing claim and it is mechanistic — a prose verdict cannot contain
  an implementation — but the exact scores are a demonstration, not a measurement.
- Gold is matched on filename substrings, so a query mentioning "sparse" scoring
  against `sparse.py` is partly lexical by construction. This mirrors how the
  lexical arm would work in use, but it inflates the apparent difficulty.
- Only whole-file deletions are indexed. Removed *hunks* inside surviving files —
  probably the larger population in most repos — are untested here.
- `remax` is a best case for the negative hypothesis (it documents rejections)
  and possibly a best case for the positive one too (its deletions are coherent
  whole-feature removals with good commit messages). A repo with messy history
  may score worse on both arms.

## Cost

~3 min per run: 858 chunks encoded across three corpora, plus git plumbing for 21
file histories. No credentials. Two runs (the second after the relocation fix).

## Reproduce

```bash
python3 history-tombstone-index/run.py     # expects /home/user/remax
```

## What this suggests for the product

The differentiating claim is narrower and more defensible than "index your git
history": **index what was removed and never came back**. Everything else in
history is a near-duplicate of content already indexed, and near-duplicate
pollution is a measured failure mode in this line of work
(`hybrid-code-index`, `repo-index`).

`git log -S` and hosted commit search already do this *lexically* — they need the
keyword. The gap this whole tool line exists to close is the case where you do
not have it (1/5 vs 5/5 in `repo-index`). Semantic retrieval over removed code is
where those two meet.

Not done: removed-hunk indexing, a per-file cap on tombstone density (proposed to
bound near-duplicate runs, never needed at this scale), and a second target repo.
