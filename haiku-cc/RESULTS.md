# RESULTS: a C11 compiler built by a Haiku 5.5 swarm

From Oskar's "Go" at 02:56 UTC on 2026-10-12 to the final measurement at 02:52 UTC on
2026-10-13. The compiler (`hcc/`), the test harness (`tests/`), the swarm orchestrator
(`swarm/`) and the statistics script (`analysis/stats.py`) are mirrored here from the working
repo; `SNAPSHOT.md` says which commit. Final numbers are at commit 0ea26d7e, measured once
across all 22 suites by the hub and three measurement sessions.

## Answer

Can a swarm of Claude Haiku 5.5 workers build a C11 compiler, and at what cost next to
Nicholas Carlini's Opus 4.6 compiler? Yes, for a narrower compiler. Three Opus 5.5 agents
wrote a 7K-line Python foundation, the test harness and the orchestrator in 27 minutes for
$14.42. 1,436 Haiku workers then fixed one cluster of failing tests at a time behind a serial
merge gate: 518 merges for $40.50 in 21.9 hours. At 0ea26d7e hcc is 22,644 lines of
standard-library Python passing all 38,451 tests in 22 suites, including GCC torture,
CPython 3.12, SQLite, Mbed TLS, Redis, QuickJS and Doom.

Carlini's compiler cost just under $20,000 over two weeks. It is 100K lines of Rust with its
own code generators for x86, ARM and RISC-V, and it builds Linux 6.9. hcc emits x86-64 only,
uses the GNU assembler and linker, and has not attempted Linux.

In three controlled runs on 54 failing clusters, Sonnet 5.5 cost about 7 times as much per
solve as Haiku ($0.075 to $0.087 against $0.011 to $0.012). The swarm's unsolved tail was
mostly made by harness faults, and the final measurement found regressions the swarm's
sampled gate had let through, including one that broke 3,340 Mbed TLS tests.

What would change this: a Linux kernel attempt, or clusters only Sonnet solves.

## Findings

1. **The build cost $54.92 metered for 518 merged fixes in 21.9 hours.** Haiku workers
   $40.50 (hub $26.84, siblings $13.66) and the Opus foundation $14.42, at API list price.
   Not metered: the Opus sessions that coordinated the hub, siblings and measurement nodes,
   the Sonnet agents that built test suites, and the seven Haiku Agent-tool fixers of the
   last stretch. Hub tokens: 613M cache-read, 12.1M output, against Carlini's 2B input and
   140M output. (`figures/comparison.md`, `figures/metrics.json`)
2. **Sonnet 5.5 solved slightly more runs than Haiku 5.5 at about 7 times the cost per solve.** Run 1 (40
   random failing clusters): Sonnet 80/80 runs, Haiku 76/80, $6.01 against $0.81. Run 2
   (4 clusters Haiku had failed in the swarm and that still failed at the pinned base):
   8/8 for Haiku, Haiku at high effort and Sonnet. Run 3 (10 abandoned clusters that still
   failed): Haiku 19/20, Haiku-high 20/20, Sonnet 20/20, $0.012 against $0.087 per solve.
   The two clusters only Sonnet solved in run 1 (unwind cleanup handlers, `_Decimal`
   types) were later fixed by Haiku feature-mode workers for $0.054 and $0.022. Run 1
   drew from the live failing set, which the swarm was fixing while the comparison ran, so
   it saturated; runs 2 and 3 re-checked their clusters at the current base first.
   (`figures/comparison.md`, `analysis/escalation.py`)
3. **The final measurement found what the sampled gate had let through.** The gate re-ran a
   sample of passing tests, and the last full refresh before the queue drained was 45
   merges old. At the swarm's end state (0845a36b) an enum change had made the constants of
   ordinary enums unsigned, so the hcc-built TinyCC passed 35 of 333 tests; 8 of 710
   compiler selftests failed because the gate never ran them; and Mbed TLS failed 3,340 of
   11,970 tests. The Mbed TLS bug came from 95980b19 (11:50Z), which added rbx to the inline
   asm register pool while the `"b"` constraint still meant any register; Mbed TLS's bignum
   asm reads `%rbx` directly. No late round dispatched Mbed TLS, so nothing sampled it for
   14 hours. A 4-way parallel bisect over 212 commits found it in 6 minutes; the fix is one
   table entry (0ea26d7e). The loop now refreshes when its queue drains, the gate runs the
   selftests, and workers can report a cluster `stale`. (`swarm/run.py`, `swarm/merge.py`,
   `hcc/codegen/asm_ext.py`)
4. **The swarm's unsolved tail was mostly made by the harness.** Of 33 clusters no node
   was dispatching any more, 23 already passed at the current base. The other 10 were
   solved by every arm in run 3. Tests had been parked (taken out of the queue after 3
   attempts without a merge) on attempts the harness had wasted: stale state, skipped
   tests counted as regressions, and conflicts counted as failed attempts. The hub's
   relaunch at 18:12Z sets an attempt epoch, so attempts before it no longer count toward
   parking. (`swarm/queue.py` `SWARM_ATTEMPT_EPOCH`)
5. **Merge conflicts were the largest loss.** 176 hub rejections were conflicts applying
   onto main, and sibling s1 lost 29 of its 101 local merges when its branch was rebased
   onto hub main. Handing a conflicted diff to the next worker as a rebase task recovered
   77 merges from 131 rebase workers. Moving shared lists (builtins, type specifiers,
   mixins) to self-registering modules at 13:50Z, where a new entry is a new file, was
   followed by a fall in the conflict share of gated hub candidates from 35% to 13%; other
   harness changes landed in the same window. The run's failure review traced at least 58
   of 128 conflicts to those shared lists. (`hcc/registry.py`, `figures/metrics.json`)
6. **Stale gate state sent about 100 workers after tests that already passed.** The gate
   re-checks at most 20 tests of a cluster, so when one merge fixed the QuickJS build,
   about 3,900 QuickJS tests stayed FAIL in the known state (the gate's record of each
   test's last status). The fix re-runs every known-failing member of a merged cluster.
   (`swarm/merge.py` `recheck_clusters`). At the end, 104 of the 110 failures still on
   record were tests that already passed, in suites no late round re-measured.
7. **A test that stopped running was treated as a regression.** When a suite builder
   added tests to a skip list, the gate counted them as newly failing: 27 good patches
   were rejected for the same 42 tests. (`swarm/merge.py`)
8. **A regression outside a node's own partition went unseen and stopped a sibling.**
   Hub merge d31ba839 broke `gcc-dg-compile/pr112488-2.c`, a test only sibling s1 ran.
   s1's gate then rejected 57 of 63 candidates for "regressing" it, and s1 stopped at
   13:24Z. Every node's gate sample now includes all cheap suites.
   (`swarm/merge.py`, `figures/merges_timeline.png`)
9. **One container's CPU capped the swarm for 8 hours.** Each worker compiles and runs
   tests, so 4 vCPU held the hub at load 10 to 22 whatever the launch mechanism. Three
   sibling containers from 11:27Z added 366 workers and 81 of the 460 merges for 36% of
   the worker spend; s2 landed none. (`figures/outcomes_by_kind.png`, `figures/spend.png`)
10. **The swarm filled the disk at 12:20Z after about 680 workers.** Worker scratch in
   `/tmp/<task id>/` (up to 170 MB each, 14 GB in all) was never deleted, and the
   symptom was the loop crashing on `git worktree add`. (`swarm/run.py` reap)
11. **The conformance suites lasted 70 minutes.** The first Haiku round took c-testsuite,
    chibicc and GCC torture to saturation (67 merges, $2.62) and the queue drained; the
    run continued only because Sonnet agents had been building larger suites.
    (`figures/merges_timeline.png`)
12. **Shell and session faults cost runs.** A `cmd | tail && git commit` chain committed
    after failed commands, because only `tail`'s exit status reaches the shell. Bash
    background jobs end at a 30-minute default timeout or the 2-hour cap, so the swarm loop
    has to be resumable and relaunched. A validation started in a subagent's background
    died with the subagent. (`METHODS.md`)

## Method

**Compiler and contract.** hcc is a C11 compiler in Python 3.11 with no dependencies. It
emits x86-64 System V assembly; the system `as` assembles it and `gcc` serves only as the
linker driver. `docs/CONTRACT.md` fixes the module layout and the stable interfaces that
every worker codes against.

**Tests.** `tests/run.py` runs 22 suites. A test passes when the hcc-built program behaves
like the gcc -O0 build (exit code, stdout, stderr), or, for compile-only suites, when `as`
accepts hcc's output; for `reject-gcc-dg`, when hcc rejects a program gcc rejects. The
suites run from conformance tests (c-testsuite, chibicc, GCC torture, gcc.dg, LLVM
SingleSource) through differential csmith programs to whole programs built by hcc and run
under their own test suites (SQLite, Lua, QuickJS, CPython 3.12, Redis, Mbed TLS, TinyCC,
zlib, bzip2, cJSON, Doom). Skip lists name 8,491 tests the runner does not run: other
targets, unsupported options, tests gcc itself rejects, and in gcc-dg-compile a fixed
sample of passing tests dropped to bound the run time. The final count (38,451) is
every test the runner runs at 0ea26d7e, measured in one pass; counts in the log before
then were taken from the known state (`swarm/state/known.json`, the last observed status of
each test) and included 4,619 sampled-out gcc-dg-compile tests with the PASS they had when
they were dropped. Upstream sources are fetched by `tests/scripts` and
`tests/drivers` and are not mirrored here.

**Swarm.** `swarm/run.py` clusters failing tests by a normalised failure signature (a
"cluster") and ranks clusters by suite level, size and earlier failed attempts. Each Haiku
5.5 worker is one headless `claude -p` session in its own git worktree with a $2 and
1,500 s cap, given one cluster and the notes of earlier attempts. A serial gate stacks 4
to 8 candidates, runs smoke tests, each candidate's targets and a fixed sample of passing
tests, and merges only when no test regresses against the known state and each candidate
flips at least one target. Full refreshes bisect and revert regressions. Feature mode
gives one worker a whole feature after two failed attempts; a rebase task hands a
conflicted diff to the next worker.

**Federation.** From 11:27Z three sibling sessions (s1 to s3), each its own 4-vCPU
container, ran the same loop on a partition of the suites and pushed candidate branches;
the hub gated them again ("ingest") before merging to main.

**Escalation runs.** `analysis/escalation.py` gives Haiku 5.5 at default effort, Haiku
5.5 at high effort and Sonnet 5.5 at medium the same cluster, prompt, base commit and
caps, two replicates each. A run is solved when a representative test flips to PASS with
no regression in smoke plus 120 sampled passing tests, graded in isolation. Nothing is
merged.

**Metrics.** Spend is the runner's per-worker cost at API list price. `analysis/stats.py`
reads the hub's state, the siblings' ledgers and the escalation results; it counts a
sibling candidate once, under its sibling.

## Log

### 02:56–03:25Z: foundation

Three Opus 5.5 agents, headless, wrote the contract, a 7K-line hcc, `tests/run.py` with
smoke, c-testsuite and chibicc, and `swarm/` with its gate and a fake worker for testing.
27 minutes, $14.42. Start: c-testsuite 213/220, chibicc 30/41, torture 1,442/1,632.

### 03:25–04:33Z: round 1, conformance suites

24 worker slots, held to about 6 by the load throttle on 4 vCPU. 166 workers, 67 merges,
$2.62. c-testsuite 220/220, chibicc 40/41, torture 1,613/1,632, Lua 91/91, SQLite
183/184. The queue drained. Merge conflicts were 38 of the first 100 rejections.

### 04:33–05:34Z: stale state

A merge fixed the QuickJS whole-program build; about 3,900 QuickJS tests stayed FAIL in
`known.json` and about 100 workers launched on them, each reporting "does not reproduce at
HEAD" for about $0.005. Fix 9a0c2bc re-runs every known-failing member of a merged
cluster; the loop refreshes state when `tests/` changes and waits for new suites instead
of exiting.

### 05:34–08:34Z: feature mode, skips, rebase tasks

Workers refused half-features (a `_Complex` suffix without a complex type) and quit in 3
to 6 turns; feature mode (07:06Z) gives one worker the whole feature. A skip-list commit
turned 42 tests into "regressions" and 27 good patches were rejected for them; fixed at
07:58Z. From 08:34Z a conflicted diff becomes a rebase task and no longer counts as a
failed attempt.

### 08:34–11:27Z: throughput

Gate retune (larger batch, smaller sample) took merges from about 15 to about 41 per hour.
Whole-program build failures were ranked first: SQLite's `__sync_synchronize` fix cost
$0.007 and unblocked hundreds of tests. PyPy runs hcc on large inputs (2.1 times faster
there, 1.8 times slower on small tests), chosen per input by `tests/bin/hcc-auto`. 141
merges and $11.40 by 11:05Z.

### 11:27–15:25Z: federation, disk, cross-partition regression

Three siblings started at 11:27Z. The disk filled at 12:20Z (worker scratch, 14 GB); fix
d880e97 deletes `/tmp/<id>` at reap. At 13:13Z: 745 hub ledger rows (ingests included),
291 merges, 41,772/42,391 tests, $20.23 of Haiku. Hub merge d31ba839 broke a test only
s1 ran, and s1 stopped at 13:24Z after rejecting 57 of 63 candidates; fix 7f2cbf26 puts
every cheap suite in every node's sample. Self-registering modules landed at 13:50Z
(78a529bb). By 15:20Z the hub, s2 and s3 had run out of work; s2 and s3 were stood down.

| node | workers | merged | rejected | failed | spend | landed on hub main |
|---|---|---|---|---|---|---|
| s1 | 271 | 72 | 165 | 34 | $7.98 | 72 |
| s2 | 40 | 0 | 27 | 5 | $0.34 | 0 |
| s3 | 55 | 9 | 19 | 27 | $5.34 | 9 |

### Escalation runs 1 to 3, finished by 17:00Z

Run 1 (base f0b6e3f) sampled 40 failing clusters the swarm had not reached; both models
solved 38, only Sonnet 2. Run 2 selected 18 clusters Haiku had failed at least once; 14
were already fixed on main and all arms solved the other 4. Run 3 took the 33 clusters in
suites no node was dispatching; 23 passed at base 2eefd0e and all arms solved the other
10. The 10 Haiku patches were injected as candidates (`esc3-c00` to `c09`).

| run | design | arm | runs | solved | $ total | $ per solve | median turns | median s |
|---|---|---|---|---|---|---|---|---|
| 1 | 40 random failing clusters | haiku | 80 | 76 | 0.81 | 0.011 | 14 | 63.5 |
| 1 | 40 random failing clusters | sonnet | 80 | 80 | 6.01 | 0.075 | 9 | 45 |
| 2 | 4 clusters Haiku had failed | haiku | 8 | 8 | 0.10 | 0.012 | 14.5 | 125 |
| 2 | 4 clusters Haiku had failed | haiku-high | 8 | 8 | 0.18 | 0.023 | 23 | 108 |
| 2 | 4 clusters Haiku had failed | sonnet | 8 | 8 | 0.61 | 0.077 | 12 | 91 |
| 3 | 10 abandoned clusters | haiku | 20 | 19 | 0.22 | 0.012 | 14 | 77.5 |
| 3 | 10 abandoned clusters | haiku-high | 20 | 20 | 0.35 | 0.018 | 18 | 87 |
| 3 | 10 abandoned clusters | sonnet | 20 | 20 | 1.74 | 0.087 | 9 | 47 |

The run-1 sample saturated because the swarm was fixing its own failing set while the
comparison ran.

### 18:12Z: hub round 19, as of 18:19Z

The hub relaunched with `SWARM_ATTEMPT_EPOCH` at 18:12Z, unparking tests parked on harness
failures, to finish reject-gcc-dg (835/888) and the re-opened tail. Numbers from
`analysis/stats.py` at 18:19Z, hub HEAD 9f2d24a1, hub still running:

| | Carlini (Opus 4.6) | haiku-cc |
|---|---|---|
| model | Opus 4.6 | Haiku 5.5 workers (Opus 5.5 for the 3 foundation agents and coordination) |
| parallel agents | 16 | 6-24 per container; hub + 3 sibling containers from 11:27 |
| sessions | nearly 2,000 | 1,235 Haiku workers (869 hub + 366 on 3 siblings) |
| cost | just under $20,000 | $37.99 Haiku workers (hub $24.33, siblings $13.66) + $14.42 Opus foundation, API list price; Opus coordination not metered |
| tokens | 2B input, 140M output | hub only: 535M cache-read, 10.4M output |
| wall clock | two weeks | 15.4 h so far |
| language | Rust (std only) | Python 3.11 (stdlib only), PyPy for large inputs |
| size | 100,000 lines | 21,536 lines + 630 selftests |
| assembler/linker | GCC's (own ones "still somewhat buggy") | system as + gcc as linker driver |
| headline result | builds Linux 6.9 on x86, ARM, RISC-V; QEMU, FFmpeg, SQLite, postgres, redis, Doom | CPython 3.12, SQLite, Mbed TLS, Redis, QuickJS, Lua, Doom, TinyCC; no Linux attempt |
| test suites | "99% pass rate on most compiler test suites" incl. GCC torture | 42,181/42,358 tracked tests pass |
| coordination | lock files in current_tasks/, agents pull/merge/push, resolve own conflicts | orchestrator queue by failure signature, per-worker worktree, serial no-regression gate; 460 merges on main |

Per suite: bzip2 48/48, c-testsuite 220/220, chibicc 40/41, chibicc-self 101/101, cJSON
183/183, CPython 5,723/5,723, csmith 239/239, Doom 93/93, gcc-compile 1,926/1,926, gcc-dg
2,269/2,269, gcc-dg-compile 7,918/7,919, GCC torture 1,632/1,632, llvm-single 278/278, Lua
122/122, Mbed TLS 11,950/11,970, QuickJS 4,508/4,510, Redis 2,375/2,475, reject-gcc-dg
835/888, smoke 34/34, SQLite 1,273/1,273, TinyCC 333/333, zlib 81/81. Redis and QuickJS
are outside round 19's dispatch set. The gcc-dg-compile count includes 4,619 tests that
were sampled out at 06:02Z and carry the PASS they had then.

Hub worker outcomes: 369 merged, 245 rejected, 239 failed, 1 reverted, 15 in flight.
Losses by reason: worker reported failed 194, conflict applying onto main 176, empty diff
64, worker reported partial 43. The 16:53Z draft counted 950 hub workers because it
included ingested sibling candidates; `stats.py` now counts those under their sibling.

### 18:19–21:37Z: rounds 19 to 22, quick refresh, queue empty

At 18:15Z the hub merged 9 of the 10 run-3 Haiku patches (`esc3-c00` to `c09`); `c00` fixed
Doom's three demo replays, and `c03` conflicted with another candidate in its batch and then
with main. At 18:37Z the hub
ran a full refresh after about 200 merges. It finished at 19:19Z with 82 regressions that
the gate's 80-test sample had missed. The bisect named 73daae00 (`__attribute__((cleanup))`),
256 commits back; its revert conflicted, and the gate accepted the regressions into the
known state so that workers would pick them up as ordinary failures. Commit 2cddfd3d at
19:49Z added a quick refresh: the fast suites (smoke, c-testsuite, chibicc, GCC torture)
every 25 merges, bisecting only over those merges, so a culprit is still recent when found.
It is covered by the orchestrator selftest; round 21 merged only 7 times, so it never fired
in production.

Round 21 ran out of work at 20:12Z: its last 7 workers fixed C23 enum, constexpr and typeof
cases in gcc-dg and gcc-dg-compile, which left no failing test in its dispatch set (`--only`:
the conformance suites, reject-gcc-dg and gcc-dg-compile). The 122 failures still in the
known state were Redis 100, Mbed TLS 20 and QuickJS 2, outside every late round's
dispatch set. It idled until it was stopped at 21:37Z.

Two harness faults from this stretch: a worker's background task wrote 4.3 GB under Claude
Code's per-directory temp directory (`/tmp/claude-<uid>/<cwd>`), which the scratch cleanup
did not cover, and a kill at 16:58Z between pushing the federation state commit and saving
its parent made every later `hcc/state` push non-fast-forward until 0845a36b chained the
commit onto the fetched remote tip. Both are fixed in `swarm/run.py` reap and
`swarm/federation.py`.

### 21:37Z to 02:52Z: final measurement, late regressions, cleanup

The first full measurement at the swarm's end state (0845a36b) found GCC torture 1630/1632
and TinyCC 35/333: 8a2a89ea (20:04Z) had given ordinary enum constants an unsigned type.
2dc34f59 fixed it by hand; 8 of 710 compiler selftests also failed at that head. Four Haiku
Agent-tool workers took the selftests: four compiler regressions fixed (setjmp operand order
from the Doom-demo escalation patch, a wide-literal constant fold, a dead struct-tag guard,
struct/scalar assignment reported at the wrong stage) and three selftests GCC rejects as
written corrected (duplicate case values, `u8''` before C23, and a non-static weakref, the
last by the hub), plus one selftest that expected the wrong diagnostic wording. The gate now runs the selftests (no new
failures) and the loop runs a full refresh when its queue drains.

A cleanup round (r23/r24) found 104 of the 110 failures on record already passing: Redis,
Mbed TLS and QuickJS records from before later merges, in suites no late round dispatched.
Workers said so in their notes; the `stale` status now makes the loop re-measure such a
cluster instead of counting a failed attempt. With every suite in scope each gate rebuilt
the whole-program suites (about 15 minutes per gate), so the remaining five regressions
(qual-return-3/7, transparent-union-2, Wshadow-3, ucnid-15-utf8) were fixed by Agent-tool
workers and the hub directly; `c99-intconst-1` compiles in 43 s at 3844edfc and at HEAD
alike and times out only under load.

The measurement at aa6426e9 ran on the hub plus three sibling sessions (m1 to m3), each
measuring a share of the suites and pushing JSON to `hcc/final/<node>`. It found Mbed TLS at
8,630/11,970: 95980b19 (11:50Z) had put rbx in the inline-asm register pool while `"b"`
still meant any register, and Mbed TLS's MULADDC reads `%rbx`. A 4-way parallel bisect over
212 commits found it in 6 minutes; 0ea26d7e maps `"b"` to rbx. The re-measurement at
0ea26d7e: 38,451/38,451 in 22 suites. Mbed TLS's three shards ran in 4 to 5 minutes each
instead of over 30.

| suite | pass/total | suite | pass/total |
|---|---|---|---|
| bzip2 | 48/48 | lua | 122/122 |
| c-testsuite | 220/220 | mbedtls | 11,970/11,970 |
| chibicc | 41/41 | quickjs | 4,510/4,510 |
| chibicc-self | 101/101 | redis | 2,475/2,475 |
| cjson | 183/183 | reject-gcc-dg | 888/888 |
| cpython | 5,723/5,723 | smoke | 34/34 |
| csmith | 239/239 | sqlite | 1,985/1,985 |
| doom | 93/93 | tcc | 333/333 |
| gcc-compile | 1,926/1,926 | zlib | 81/81 |
| gcc-dg | 2,269/2,269 | gcc-dg-compile | 3,300/3,300 |
| gcc-torture | 1,632/1,632 | llvm-single | 278/278 |
