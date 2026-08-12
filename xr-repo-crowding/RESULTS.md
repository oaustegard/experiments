# xr-repo-crowding — does repo size bury small repos in account-wide search?

**No, and the three obvious corrections all make retrieval worse.** Under the
current ranker, small repos *outperform* large ones. A per-repo size prior —
the intuitive fix — costs 25 points of recall@1.

Run 2026-08-11 against account-index build `2026-08-11T21:33:07Z`
(42,332 chunks / 65 repos, tree corpus only).

## The claim under test

`xr`'s docstring says account-wide search buries per-repo findings, citing one
observation: "about to fan out concurrent LLM calls through a Cloudflare
gateway" returns nothing relevant in the top 20 account-wide, because
`claude-skills` contributes 8,088 chunks thick with API-invocation vocabulary,
while `-r experiments` puts the writeup at #2. The advice that follows — always
scope with `-r` — is not in question. What is in question is the mechanism: if
chunk count drives it, a ranker-side correction should recover the lost results
without needing the user to know which repo to name.

## Query set

145 file-localization queries mined from merged PRs across 19 repos spanning
278 to 11,195 chunks (40x).

- **Query** = PR title + first 400 chars of body.
- **Truth** = the files that PR changed, filtered to files the index carries
  (extension filters, size caps and later deletions mean a changed path may
  have no chunk; scoring against an unreachable target measures the corpus,
  not the ranker).
- **Not circular**: the index under test carries no PR chunks, so no query can
  retrieve its own body.

Ground truth comes from what the PR *did*, not from anyone's judgement about
what should rank — no hand-written queries, per the eval-realism problem that a
test resembling a test draws a grade-targeted answer.

## Arms

All five share one encode, one dense scan and one BM25 pass per query, so they
differ only in how candidates are pooled and fused.

| | |
|---|---|
| **A** baseline | current `xr`: top-25 files per arm, RRF, top-k |
| **B** cap | baseline, then ≤2 files per repo in the final list |
| **C** size prior | RRF contribution scaled by `1/log(1+chunks_in_repo)` |
| **D** diverse pool | ≤2 files per repo while *building* each arm's 25 |
| **O** oracle | baseline restricted to the ground-truth repo (`-r`) |

O is the ceiling. A-to-O is what scoping buys, i.e. the size of the problem.

## Result

recall@k, identifier-rich queries:

| k | A | B | C | D | O |
|---|---|---|---|---|---|
| 1 | **73.8** | 73.8 | 48.3 | 67.6 | 77.2 |
| 3 | **87.6** | 85.5 | 75.9 | 77.2 | 89.7 |
| 5 | **89.7** | 86.9 | 84.8 | 84.1 | 95.2 |
| 10 | **94.5** | 87.6 | 92.4 | 91.0 | 97.9 |

Every intervention is neutral-to-worse at every k. C loses 25 points at k=1
because 41 of 145 queries legitimately belong to a large repo, and a prior that
down-weights large repos pushes those right answers down.

Split by repo size (recall@1, arm A): large 68.3, mid 70.8, small 80.4. Small
repos are not being buried. Large-repo queries are harder — plausibly more
internal confusability, since a big repo contains more near-duplicates of its
own answer.

## Identifier-poor replication

PR text names the files it changed, and identifier-rich retrieval is the regime
where plain lexical matching already wins. The original anecdote was a
*conceptual* query with no such handles. Deleting every token the ground-truth
paths give away turns each query into its identifier-poor twin against the same
ground truth:

| k | A | B | C | D | O |
|---|---|---|---|---|---|
| 1 | **68.3** | 68.3 | 45.5 | 63.4 | 75.9 |
| 10 | **88.3** | 86.2 | 86.9 | 87.6 | 93.8 |

Absolute recall drops ~5 points. The ordering of the arms does not move, and
small still beats large (75.0 vs 65.9 at k=1).

## What survives

`-r` is worth 3.4 points at k=1 and up to 7.6 at k=5. That is the whole real
effect, and it is already the documented advice. The 8,088-chunk observation
stands as a single case, not as a systematic bias with a ranker-side fix.

## Method note — score at multiple k before calling two rankers equivalent

The first run scored only recall@10, where A=94.5 and the arms sit between 87
and 95: saturated, indistinguishable, and one conclusion away from "no
measurable difference, ship whichever." The arms separate at k=1, where C is 25
points down. A saturated metric nearly hid the largest effect in the study.

## Limitations

One account, one index build, n=145. PR-derived queries skew toward "find the
file this change touched" rather than "have we built this anywhere" — the
stripped variant approximates the second but does not fully cover it. A query
set built from real "prior art" questions would test the anecdote's regime
directly; none is logged, which is itself worth fixing.

## Reproducing

```sh
export GH_TOKEN=...                 # needs pull-requests:read
python3 build_queryset.py           # -> queries.json, repo_chunks.json
python3 eval.py                     # -> results.json
STRIP=1 python3 eval.py             # -> results_strip.json
```

`build_queryset.py` and `eval.py` import `xr` from a `claude-workspace`
checkout at `/home/claude/cw`; both read the prepared cache under
`~/.cache/xr`, so run `xr` once first to populate it. Query set and per-query
outcomes are committed, so the tables can be recomputed without the API or the
124 MB encoder.
