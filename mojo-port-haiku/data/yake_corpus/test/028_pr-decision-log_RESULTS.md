# pr-decision-log

**Started / finished:** 2026-08-05 · **Status:** done — **positive; PR bodies
answer a question neither the tree nor the deleted code can.**

**Question.** `history-tombstone-index` showed a working tree cannot hold the
*mechanism* of code that was deleted. This asks the adjacent question about
*rationale*. Code says **what**, commit messages say **what changed**, PR
descriptions say **why** — including what was considered and rejected. None of
that is in the tree.

Proposed as an alternative to indexing hunk-level changes, and the volume
argument alone favours it: 43 merged PRs is 87 chunks, **+12% over the tree**,
where hunks would have been thousands of near-identical neighbours — the same
near-duplicate pollution that cost 20% of the corpus in `repo-index`.

## Setup

- **Target:** `oaustegard/remax`. Tree 720 chunks, tombstones 138, PR bodies 87.
- **Queries:** 8 "why" questions, written from **CLAUDE.md's claims** rather than
  from PR text, so the query side is not copied from the corpus under test.
- **Retrieval:** RRF(dense, stored-BM25).
- Scored as **marginal value** over corpora that already exist — including a
  CLAUDE.md that deliberately documents decisions, with a table of three
  anti-goals overridden by shipped work.

## Results (rationale hit@5)

| corpus | hit@5 |
|---|---|
| tree | 6/8 |
| tree + tombstones | 6/8 |
| **tree + PRs** | **8/8** |
| tree + tombstones + PRs | 8/8 |

**Tombstones add exactly nothing here (6/8 → 6/8), and that is the useful
finding.** The three corpora are orthogonal, each answering its own question
class:

| corpus | answers |
|---|---|
| working tree | *what does the code do* |
| tombstones | *how did the deleted thing work* |
| PR bodies | *why was it done that way* |

Adding a corpus that answers a different question than the one being asked is
inert, not harmful — worth knowing, because it means these can be stacked
without the arms fighting.

### The two the tree could not answer

| query | found in tree+PRs |
|---|---|
| "Why was the benchmark harness removed from the installable package?" | `[PR #65] Consolidation: −4,879 lines of Python, every conclusion kept` (rank 4) |
| "Why was assignment to `rotations_` restored with a write-through setter?" | `[PR #61] Restore rotations_ assignment via write-through setter` (**rank 1**) |

Both are cases where the tree carries the *outcome* but not the *reason*.
`CHANGELOG.md` ranks first for the harness question in both arms and records
that it happened — it does not say why, and the querier asked why.

## Caveats

- **The generalization threat is severe and was recorded before running.** Every
  remax PR is authored through Claude sessions: median body **2,727 chars**, 46
  of 46 non-empty. Most repos have one-line or blank PR bodies, where this
  corpus would be noise with a title attached. **This measures "PR bodies are
  worth indexing when they are written like this", not that they are in
  general.** The honest generalization is conditional: check median body length
  before believing it will transfer.
- n=8, one repo, queries by someone who knew the repo.
- One gold list originally contained the bare substring `"PR #"`, which matches
  *any* PR chunk. Fixed. It was outcome-neutral: that query hits via `CLAUDE.md`
  at **rank 1 in both arms**, verifiable in the recorded top-5, so the totals do
  not move. Noted anyway because it is the third answer-key defect of this shape
  in this line of work — see `history-tombstone-index` (relocations) and
  `code-index-duplication` (the harness retrieving itself).

## The architectural cost, which is the real objection

PR bodies **are not in git**. Every other corpus here comes off the filesystem,
which is why the indexer can run offline, in CI, with no credentials. Indexing
PRs makes network access and a token a hard dependency of a full rebuild, plus
rate limits and staleness between builds.

`prs.json` is committed so this experiment reproduces without either — but that
is a workaround for one repo at one point in time, not a design. A real version
needs a cache with an explicit staleness policy, and should degrade to
tree+tombstones when the network is absent rather than failing the build.

## Cost

~5 min: 4 corpora × ~800-950 chunks encoded, plus one GitHub API page fetch.

## Reproduce

```bash
python3 pr-decision-log/run.py     # uses committed prs.json; no token needed
```
