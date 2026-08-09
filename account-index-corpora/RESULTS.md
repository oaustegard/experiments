# account-index-corpora — what tombstones and PR bodies cost the account index

`history-tombstone-index` and `pr-decision-log` both measured clear wins on
`remax`: deleted files took mechanism questions 0/6 → 6/6, merged PR bodies took
rationale 6/8 → 8/8. Neither is in the account-wide build.
[`claude-workspace#197`](https://github.com/oaustegard/claude-workspace/issues/197)
asks whether they should be, and names the risk correctly — not answer quality
(an off-class corpus measured *inert*, not harmful) but **size**, in an index
whose known weakness is already crowding.

It also asks about a hardcoded `--depth 50` in `account.py cmd_clone`, justified
by a comment referring to a tombstone corpus that does not exist.

Size is the half of the question that costs seconds instead of a 22-minute
sharded encode, so it was answered first and on its own.

## What was measured

Two scripts, both offline-ish and neither touching the encoder:

- `clone_depth.py` — clone each repo at depth 1, depth 50 and full, recording
  wall time, bytes received, and how much of the deletion record each depth can
  actually see.
- `account.py corpora` (added to `hybrid-code-index/` by this work) — chunk the
  tree, tombstone and PR corpora through the same path a real build uses and
  report the counts. No model, no index written.

The per-corpus analysis below was done on 3 repos, because that is what the
session could clone. It has since been **re-run over all 65** on a runner
([run 31310605478](https://github.com/oaustegard/claude-workspace/actions/runs/31310605478),
2 min 18 s) — see *Account-wide* at the end. Both sets of numbers are kept: the
3-repo run is where the content analysis was done, the account run is the
authority on size.

## Result 1 — `--depth 50` costs nothing, and would not work if it were used

| repo | | secs | MiB | commits | deletions seen |
|---|---|---:|---:|---:|---|
| claude-workspace | depth 1 | 1.4 | 0.12 | 1 | 0 / 644 |
| | depth 50 | 2.8 | 23.03 | 165 | 640 / 644 |
| | full | 2.6 | 25.81 | 478 | 644 / 644 |
| muninn-utilities | depth 1 | 1.0 | 0.79 | 1 | 0 / 18 |
| | depth 50 | 0.9 | 0.93 | 113 | **2 / 18** |
| | full | 0.9 | 1.33 | 241 | 18 / 18 |
| experiments | depth 1 | 4.1 | 29.10 | 1 | 0 / 3 |
| | depth 50 | 3.6 | 31.93 | 108 | 3 / 3 |
| | full | 3.9 | 31.94 | 108 | 3 / 3 |

Summed wall clock: 6.5 s at depth 1, 7.3 s at depth 50, 7.4 s full. The
differences have no consistent sign across repos — depth 50 was *faster* than
depth 1 on two of three — so at this scale they are run-to-run noise on a step
the issue already prices at 89 s against a 22-minute encode. The bytes do move
(claude-workspace transfers 190× more at depth 50 than at depth 1) without
showing up in time.

The interesting column is the last one. `git log --diff-filter=D` sees only the
grafted window, so a shallow clone yields **a sliding recent slice of the
deletion record, not the record**. Depth 50 covered a 2-month window of
muninn-utilities and found 2 of its 18 deletions. Whether that is 11% or 99%
coverage depends entirely on the repo's commit rate, which is not a property
anyone chose.

So there is no cheap middle to buy. `--depth 50` was carrying history nothing
read, and had anything read it, it would have read an arbitrary fraction of it.
Resolved as depth 1 when tombstones are off, full history when they are on —
a full clone measured within noise of both.

## Result 2 — the tombstone corpus is 94% deleted machine-generated data

First run, tombstones enabled account-wide-as-scoped:

```
tree             13,257    100.0%
tombstones       74,822    564.4%      <- 1.7x the entire live account index
```

Not a measurement artifact — a real bug in the corpus builder, and the reason
"measure before shipping" was the right instruction. A deleted file gets no
`stat()` and no `rglob`, so **every filter `hcindex.discover` applies to the
working tree has to be reapplied by hand**: extension, `skip_dirs`,
`skip_names`, the repo's own `exclude` list, and the 1 MiB size cap. Without
them, `phase-a-bridges/data/full_body_embeddings.json` — 767,692 lines of
deleted embedding dump — walks straight into the corpus, even though the live
index refuses it and always did. The tombstone corpus was not surfacing lost
knowledge; it was importing, labelled *gone*, the exact rows the tree already
declines to carry.

With the filters applied (`account.admissible`):

| corpus | chunks | % of tree | chars |
|---|---:|---:|---:|
| tree | 13,257 | 100.0% | 20,008,106 |
| tombstones | 1,570 | 11.8% | 1,899,362 |
| PR bodies | 419 | 3.2% | 591,720 |

47× smaller, and now an affordable-looking 11.8%. But per repo:

| repo | tree | tombstones | PRs |
|---|---:|---:|---:|
| experiments | 11,109 | 3 | 58 |
| muninn-utilities | 1,916 | 83 | 142 |
| claude-workspace | 232 | **1,484** | 219 |

claude-workspace's deletion record is **6.4× its own working tree**, and the top
of it is still data rather than knowledge:

```
770  [deleted] experiments/phase-a-bridges/data/exp3_slot_embeddings.json
258  [deleted] experiments/phase-a-bridges/data/tier0_slot_embs.json
107  [deleted] experiments/phase-a-bridges/tier1/arxiv_cats.json
 67  [deleted] experiments/phase-a-bridges/data/arxiv_cats.json
...
 10  [deleted] experiments/INDEX.md
  5  [deleted] experiments/prior-art-probe/EVAL.md
  4  [deleted] .github/workflows/account-index-seed.yml
  4  [deleted] experiments/prior-art-probe/probe.py
```

Roughly 23 chunks of prose and source against ~1,460 chunks of sub-1 MiB JSON
dumps. Those files *were* in the live index while they existed — they are under
the cap and `.json` is an indexed extension — so the tombstone corpus is
faithfully mirroring a tree that already indexes data. It is not adding a defect;
it is doubling one, in the corpus specifically claimed to answer *how did the
deleted thing work*.

A second account-scale effect the per-repo experiment could not see: the
relocation guard has to compare **across repos**. The 37 research projects that
moved from claude-workspace to experiments on 2026-07-28 are deleted in one repo
and live in another; a per-repo guard skipped 540 files here only because the
check was widened to the whole account. Candidates are restricted by basename to
keep it linear — relocation preserved the filename in every case observed.

## Result 3 — PR bodies are cheap, and this account meets the stated condition

419 chunks, **+3.2%**. `METHODS.md` records the condition for believing the
remax result transfers — median PR body length, because "a repo of `fixes #12`
bodies gets noise with a title attached":

| repo | merged PRs | median body | under `min_chars` |
|---|---:|---:|---:|
| claude-workspace | 154 | 1,577 | 9 |
| muninn-utilities | 84 | 2,241 | 2 |
| experiments | 23 | 3,197 | 0 |

261 merged PRs, 11 of them effectively empty. Comparable to remax's 2,727 and
nowhere near the failure profile.

The other stated condition — *have the build degrade to the offline corpora
rather than fail* — is implemented: a per-repo fetch failure drops that repo's
PRs and continues, and the run emits a `::warning::` naming the count, because
degrading quietly would publish a smaller index that still verifies and still
answers.

## Account-wide (all 65 repos)

| corpus | chunks | % of tree | chars |
|---|---:|---:|---:|
| tree | 42,578 | 100.0% | 57,521,371 |
| tombstones | 3,043 | 7.1% | 4,366,694 |
| PR bodies | 1,953 | 4.6% ⚠ floor | 2,323,975 |
| tree + both | 47,574 | 111.7% | |

The tree count matches the published `manifest.json` (42,578) exactly, which is
the check worth having: `corpora` chunks through the same path the real build
uses, so it is measuring the index rather than an approximation of it.

Both 3-repo estimates were wrong, in opposite directions. **Tombstones 11.8% →
7.1%** (claude-workspace's 1,484 is a smaller share of a bigger tree) and **PR
bodies 3.2% → 4.6%**. The gap narrows but the verdict does not move.

**The PR figure is a floor.** 12 of 65 repos returned `HTTPError` on `/pulls`
while cloning fine, so `ACCOUNT_INDEX_PAT` carries contents read but not
pull-request read on them — claude-workspace, claude-container-layers,
career-search, chats, ADQueryModule, MSDInterview among them. That drops
claude-workspace's 154 merged PRs out of the 1,953 entirely. At the observed
1.35 chunks per PR, the true figure is nearer 5.5%. **Fixing the PAT scope is a
prerequisite to trusting this number**, and it is worth doing before any
answer-quality benchmark, since the missing repos are the ones whose PR bodies
are richest.

Two things that went right and are worth recording as such. The degradation path
behaved exactly as `METHODS.md` requires — the build dropped the unavailable
repos, continued, and emitted `::warning::PR bodies unavailable for 12/65
repos`, which is the only reason the gap was noticed at all rather than shipping
a quietly thinner corpus. And the clone: **65 repos at full depth in 68 s**,
against the 89 s the issue records for the same clone at depth 50. Full history
did not merely land within noise of the shallow clone — it measured faster than
the recorded shallow time, which retires the last reason to keep a depth number.

Per-repo, the shape from the 3-repo run holds and one new case joins it:

| repo | tree | tombstones | PRs |
|---|---:|---:|---:|
| experiments | 11,148 | 3 | 61 |
| claude-skills | 8,092 | 539 | 538 |
| oaustegard.github.io | 2,633 | 364 | 366 |
| sage | 2,899 | 20 | 0 |
| muninn-utilities | 1,922 | 82 | 144 |
| **claude-workspace** | **237** | **1,484** | 0 |
| aeyu.io | 662 | 89 | 217 |
| tree-sitter-mojo | 554 | 209 | 42 |

claude-workspace stays the inversion — 6.3× more deleted than live. Nothing else
on the account comes close, which is the point: it is not a general property of
tombstones, it is one repo that shed a research directory and kept its data files
in history.

## Where this leaves it

Both corpora are implemented and **off by default**. The size question is
answered; the answer-quality question at account scale is not, and it needs a
full encode plus a benchmark that does not exist yet.

- **PR bodies**: **+4.6% account-wide and that is a floor** — nearer 5.5% once
  the PAT can read pull requests on the 12 repos it currently cannot. For the
  corpus that answers *why*, on an account whose PR bodies match the profile the
  win was measured on. Cheapest thing on the table, and the next step is a PAT
  scope fix rather than more measurement.
- **Tombstones**: **+7.1% account-wide**, but ~94% of it is deleted data files.
  Not worth turning on as it stands. What would change that is not more history —
  it is excluding machine-generated data from *both* corpora, which is the same
  crowding problem `#197` files under "related, not in scope"
  (`code-index-duplication`).
- **`--depth 50`**: resolved to depth 1, since tombstones are not shipping — and
  the account-wide clone confirms there was never a cost to trade: full history
  for 65 repos took 68 s against the 89 s on record at depth 50.

Re-run the size half any time with the `account-index-corpora` workflow in
claude-workspace — dispatch-only, no encode, 2 min 18 s for the run above.

## Cost

No GPU, no encoder, no runner. The only components timed directly: nine clones
totalling 21.2 s (3 repos × 3 depths, table above), and the chunking pass at
**3.7 s** for all three corpora over three repos — which is the number that says
this measurement did not need to be expensive to be decisive.
