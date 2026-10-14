# nearness-lanes

Does a "nearness" score between parallel agents' changes predict which ones need
coordination, and is it worth adding to the swarm harness? This is an offline
backtest on the haiku-cc swarm's history, with no agents run.

## Answer

Nearness predicts textual conflicts almost perfectly once the diffs exist. In 2,661
replayed stale-base merges of hcc's own history, a ±3-line hunk overlap flagged
every conflict but one (a binary file), and it flagged half as many patches as
file overlap did. At that point, though, git's trial merge answers the same question
for free, so nearness adds nothing at the gate.

Before a worker starts, almost nothing predicts its files. Its failure signature
did not separate conflicts (61% with a same-stage live peer, 48% without). The one
exception is a rebase task: its files are known from the conflicted patch. Among 54
rebases that reached the gate, 31 of 43 launched next to a live worker editing those
files conflicted again (72%), against 1 of 11 without such a peer (9%). The other
predictor is how many merges land during a worker's run: 4% conflicted with 1–5
merges, 80% with 21–60.

So the harness gained one rule: hold a rebase while a near peer is live, for at most
20 minutes. The rule is in `swarming-failing-tests` 0.6.0. The counterfactual is
not measured. A live swarm with the hold on is what would change this answer: if
held rebases still conflict at near 72%, the rule is wrong.

## Findings

1. **Hunk overlap at ±3 lines catches every textual conflict except a binary-file
   one.** It flags 2.6% of patches at window 2 and 47.8% at window 32. File overlap
   flags 17.4% and 85.6% at the same windows. (replay, `score.py`)
2. **File overlap on the real ledgers: recall 0.99, precision 0.69.** It flags 77%
   of 145 candidates whose target is known. (ledger, `ledger.py`)
3. **The number of merges during the run drives the conflict rate.** 4% at 1–5
   merges, 47% at 6–20, 80% at 21–60. (ledger)
4. **Rebase tasks next to a near live peer conflict again 72% of the time (31/43),
   against 9% (1/11) without one.** (`dispatch.py` §3)
5. **A first attempt's signature carries no file signal.** 61% against 48%
   conflicted, and 178 of 201 tasks had a same-stage peer anyway. Even with oracle
   files the split is only 64% against 47%. (`dispatch.py` §1, §4)
6. **A prior attempt's files are known for 114 of 280 tasks**, mostly rebases. Mean
   Jaccard similarity with the files finally changed is 0.62. (`dispatch.py` §2)
7. **The hold costs a median wait of 506 s (p90 1,043 s) per held rebase.** The slot
   goes to other work meanwhile, and a median rebase runs for 265 s. (`dispatch.py` §5)
8. **Function-level overlap is worse than hunk:3.** It flags about 2× as many
   patches at the same recall, because hcc's parser functions are long. (replay)

## Method

**Data.** `hcc/main`, 536 commits under `hcc/`. The sibling ledgers on
`hcc/stats/s1..s3` hold 366 tasks with base commit, times, `files_changed` and
outcome. Rejected patches were not kept, so ledger analysis is file-level only.

**Replay** (`replay.py`). For each code commit c and window W ∈ {2,4,8,16,32}, the
script pretends c was developed on `start`, the commit W−1 steps before c^. Ground
truth is `git merge-tree --merge-base=c^ start c` (exit 1 = conflict), the same
question the gate asks when a stale candidate meets a moved main. The predictors
compare c's changed lines with the lines changed between c^ and start, both in c^
coordinates:
- file: a shared file;
- hunk:k: changed lines within k lines of each other;
- func: the same innermost def or class, found by Python `ast`.

Merged commits are survivors, so the base rate is below the swarm's.

**Ledger** (`ledger.py`). The target is the hub commit named in a rebase conflict,
else the last sibling merge before the gate. The 41 rows with no known target are
dropped: s2 kept no merges log, and federation restacks went unrecorded.

**Dispatch** (`dispatch.py`). Uses only what is known when a worker launches:
- the live peers on the same node;
- their eventual files, as a proxy for the worktree diff the harness reads live;
- the conflicted patch's files, for a rebase.

**Prior art.** Accioly et al. measured same-method edits as a conflict signal
(precision 58%, recall 83%). Owhadi-Kareshk et al. (2019) predicted safe merges from
Git features with F1 0.95–0.97, and conflicting merges only at 0.57–0.68. Cassandra
scheduled tasks to minimise simultaneous edits to the same files, which is the
dispatch-hold idea for human developers. Braidkeep and "claim plane" posts (2026)
do the same for coding agents with merge-tree probes and declared file claims. What
this experiment adds is a measured split on agent swarm data, and the result that
for first attempts the files are not knowable at dispatch.

## Log

### Round 1, 2026-10-14

Asked: test the "nearness and parallel work" infographic at small scale and add it
to the framework if it holds up. The infographic's proposal is to estimate
interaction risk from edit overlap, dependencies, history and CI evidence, then
coordinate only where it is high.

Ran: replay over `hcc/main` (17 s), then ledger and dispatch analysis on the three
sibling ledgers. No model calls.

Replay, precision P and recall R per predictor, with the share of patches flagged:

| W | n | conflicts | file flag/P/R | hunk:3 flag/P/R | hunk:10 flag/P/R | func flag/P/R |
|---|---|---|---|---|---|---|
| 2 | 536 | 6 | 17.4% .06 1.00 | 2.6% .43 1.00 | 5.8% .19 1.00 | 10.3% .11 1.00 |
| 4 | 535 | 16 | 32.5% .09 .94 | 6.7% .42 .94 | 12.9% .22 .94 | 18.9% .15 .94 |
| 8 | 535 | 39 | 51.8% .14 .97 | 14.8% .48 .97 | 25.4% .28 .97 | 33.3% .21 .97 |
| 16 | 534 | 86 | 72.5% .22 .99 | 28.3% .56 .99 | 41.9% .38 .99 | 52.1% .31 .99 |
| 32 | 521 | 178 | 85.6% .40 1.00 | 47.8% .71 1.00 | 62.0% .55 1.00 | 71.0% .48 1.00 |

The only conflict with no shared text was `b454fe5d98`, which deleted `.pyc`
files: a binary file, so no hunks to compare.

The predictions held. Overlap at some granularity was necessary for a conflict,
and finer granularity raised precision. The prediction that failed was my own,
made before the run: that the signature's stage would stand in for files at
dispatch. It does not.

Added to `swarming-failing-tests` (0.6.0):
- `scripts/swarmkit/nearness.py`;
- a `HOLD` step in the dispatch loop;
- the `queue.near_hold_s` setting (default 1200);
- two selftest checks.

Selftest: 47 of 47 pass, including the refactored rebase path. Not built:
- hunk nearness at the gate, because git's trial merge already answers it there;
- dependency and history scores, because there was no first-attempt signal for
  them to improve on;
- a live A/B.

Prior-art sources: [Owhadi-Kareshk et al. 2019](https://arxiv.org/pdf/1907.06274),
[Agent deconfliction](https://pub.towardsai.net/ai-agent-deconfliction-architecture-keep-parallel-coding-agents-from-fighting-over-the-same-work-990beaeb88e8),
[Claim plane](https://codex.danielvaughan.com/2026/07/28/claim-plane-pre-write-admission-control-parallel-coding-agents-codex-cli-changeintent-scope-promotion/),
[Concurrent agent PR conflicts](https://www.agentpatterns.ai/workflows/concurrent-agent-pr-merge-conflicts/).
