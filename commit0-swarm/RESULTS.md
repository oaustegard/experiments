# commit0-swarm results — 2026-10-10

## Answer

Can one Haiku 5.5 agent rebuild a whole Python library from a Commit0
skeleton (signatures and docstrings kept, bodies emptied, the library's own
tests as target)? Yes, every time it was asked. Solo, it passed 11,208 of
11,209 graded tests across the 15 admissible Commit0-lite libraries for $16.36,
including babel (6,699 tests, 65 minutes); the one miss is an
expected-to-fail chardet test the reference passes.

On eight libraries first released after 2026-07-01, which it cannot have
trained on, it also scored 100% (1,589 of 1,589 tests, $7.50). It remembers
the famous libraries (a median 55% of its informative lines are verbatim
original lines, against 17% on the unseen ones) but does not need to.

A swarm of up to 8 builders per library plus one round of shard fixers reached
11,205 of 11,209 at 1.2× the cost, two to three times faster on babel and
jinja. Its build pass alone reached 9,402: the swarm fails where files meet.

On statsmodels, the largest full-Commit0 library (3,496 stub bodies, 17,667
graded tests), neither arm finished in one night. The swarm (19 builders,
then 19 fixers) reached 14,315 (81%) for $138.70; a chain of three solo agents
reached 6,303 (36%) for $17.29. More fixer rounds on a larger machine would
move this.

## Findings

1. **One Haiku agent rebuilds a whole library.** Fifteen Commit0-lite
   libraries, solo, one attempt each (chardet also got a continuation agent,
   which changed nothing) (Round 2):

   | library | graded tests | solo | swarm build | swarm + fixers | builders | solo $ | swarm $ |
   |---|---|---|---|---|---|---|---|
   | babel | 6699 | 6699 | 5599 | 6699 | 8 | 5.77 | 8.76 |
   | marshmallow | 1228 | 1228 | 1228 | 1228 | 7 | 0.71 | 0.85 |
   | jinja | 850 | 850 | 848 | 850 | 8 | 5.65 | 2.88 |
   | chardet | 376 | 375 | 31 | 376 | 5 | 0.66 | 2.00 |
   | cookiecutter | 367 | 367 | 367 | 367 | 7 | 0.99 | 0.94 |
   | imapclient | 267 | 267 | 266 | 267 | 8 | 0.77 | 1.22 |
   | pyjwt | 258 | 258 | 0 | 258 | 2 | 0.31 | 0.57 |
   | cachetools | 215 | 215 | 215 | 215 | 2 | 0.02 | 0.03 |
   | parsel | 205 | 205 | 203 | 205 | 4 | 0.23 | 0.26 |
   | tinydb | 201 | 201 | 199 | 201 | 4 | 0.19 | 0.20 |
   | deprecated | 171 | 171 | 171 | 171 | 1 | 0.07 | 0.02 |
   | simpy | 149 | 149 | 145 | 145 | 7 | 0.23 | 0.25 |
   | voluptuous | 148 | 148 | 59 | 148 | 3 | 0.39 | 0.84 |
   | portalocker | 38 | 38 | 34 | 38 | 2 | 0.33 | 0.36 |
   | wcwidth | 37 | 37 | 37 | 37 | 1 | 0.04 | 0.06 |

   Graded tests are those the reference implementation passes in this
   harness. portalocker and simpy are graded best of three on an idle machine,
   because their multiprocess and real-time tests carry 0.2–1.1 s deadlines
   that fail under the load of 20 concurrent agents (the same solo tree scored
   34/38 loaded and 38/38 idle). The simpy swarm's 145 holds idle too: its
   `rt.py` has a real bug. minitorch was excluded because its reference commit
   is itself an unfinished course skeleton (reference 15/230).

2. **The control: libraries Haiku cannot have seen, same result.** (Round 3)

   | control library | first PyPI release | graded tests | solo | $ | verbatim rate |
   |---|---|---|---|---|---|
   | catraca | 2026-09-25 | 428 | 428 | 2.24 | 0.14 |
   | aseprite | 2026-09-05 | 260 | 260 | 0.56 | 0.27 |
   | verifactu-lint | 2026-08-07 | 254 | 254 | 0.61 | 0.20 |
   | bslfmt | 2026-09-26 | 214 | 214 | 3.28 | 0.05 |
   | africa-g2p | 2026-08-06 | 197 | 197 | 0.43 | 0.12 |
   | vstg | 2026-07-27 | 117 | 117 | 0.11 | 0.42 |
   | laga | 2026-07-24 | 67 | 67 | 0.11 | 0.23 |
   | agent-self-edit-gate | 2026-08-29 | 52 | 52 | 0.16 | 0.12 |

   A Sonnet 5.5 agent found them through the ecosyste.ms package index (web
   and GitHub search were blocked in its sandbox): 1,530 repositories checked
   for a first PyPI upload and first commit after the cutoff, 40+ tests, 500–8000
   pure-Python source lines, no lineage from an older project; 12 chosen, 8
   certified. Stubs were made the way Commit0 made its own (`ast.unparse` with
   every function body reduced to its docstring and `pass`), by
   `harness/control_setup.py`. Three candidates were dropped because their
   reference fails its own suite here (nodrill collected nothing, adaptkit
   and epub-blocks errored), one because its package layout was not found
   (pathbase).

3. **Haiku remembers the famous libraries, and does not need to.**
   `harness/memorization.py` compares the lines an agent added to the
   skeleton with the lines the reference adds, counting only lines of 25+
   characters that appear nowhere in the stub tree (tests, docs, skeleton), so
   no test or docstring could have dictated them. Median verbatim rate: 0.55
   on Commit0-lite (range 0.10 imapclient to 0.72 chardet), 0.17 on the
   control (0.05 bslfmt to 0.42 vstg). Two independent Haiku runs of one
   library agree with each other at 0.29–0.80. A rate of 0.2 is the floor for
   code written to the same tests and docstrings. The three-fold excess on the
   old libraries is recall. Agents also named upstream specifics unprompted
   ("upstream uses `get_spontaneous_environment.cache_clear()`").
   I first read the Commit0-lite rates alone as evidence that recall drove the
   scores. The control overturned that reading. (Round 3, `data/runs/*/memorization.json`)

4. **The swarm's build pass fails at the seams, and fixers close them fast.**
   Builders finish their own files in 1–7 minutes and then cannot check them,
   because the package will not import until the core file's owner lands
   (`schema.py`, `imapclient.py`, `utils.py`). They verify in isolation with
   in-memory shims and report "blocked by another builder's file". The build
   pass was below 90% on four libraries:
   - pyjwt 0/258 and voluptuous 59/148 in the pilot: files whose members the
     stubber deleted outright, leaving no empty body, were assigned to nobody
     (`utils.py`, `error.py`). Fixed by giving every source file an owner.
   - chardet 31/376: four prober files had lost their `charset_name` and
     `language` properties, and each owner saw no `pass` body and reported
     the file "already complete".
   - babel 5599/6699: 813 smoke-test failures and 272 date failures traced to
     three missing fallbacks.

   Fixers given contiguous shards of the failing test ids closed every gap in
   one round: chardet 31 → 376 with five fixers in 10 minutes, babel 5599 →
   6699 with eight in 6. When failures shared one root cause, the fixers
   converged on it: two of five chardet fixers and two of eight babel
   fixers made no edit, because another fixer's fix landed mid-run. The
   Edit tool's stale-file check served as the concurrency control, and no
   fixer clobbered another's edit. (Rounds 1–2)

5. **Cost and time.** Solo totals $16.36 for Commit0-lite; the swarm $19.24
   (build $18.19, fixers $1.05). Agent-active wall time, solo vs swarm (build +
   fix): babel 65 vs 36 min, jinja 58 vs 18, cookiecutter 14 vs 7, imapclient
   21 vs 19, chardet 13 vs 32 (the integration failure). On libraries under
   ~1,000 tests the two are within a few minutes. The swarm figures exclude
   the queueing a 20-agent session cap imposes between waves. (Round 2, transcripts)

6. **Commit0-lite and the control cost $43.10 input-side** across 116 Haiku 5.5 spawns
   (pilot $2.50, Commit0-lite solo $15.45, swarm $16.68, fixers $0.96,
   continuation $0.01, control $7.50), priced per turn with Haiku 5.5's long
   rate card (5x every rate) on prompts over 100K tokens (`harness/cost.py`).
   The figures first published here (2026-10-10, $18.41 in all) used 2x for the
   long rate and under-priced every long turn.

7. **Statsmodels, one night, neither arm finished.** Swarm build 3,106 of
   17,667 (19 builders, 83 min, $90.11), then 19 fixers on contiguous shards
   of all targets: 14,315 (81%, up to 112 min, $48.59). Solo: 1,303 (79 min,
   $4.28), then two continuations to 6,303 (36%) ($13.01 more). The swarm ran
   8,241 turns to the solo chain's 898; about 90% of turns in both arms were
   over 100K tokens and paid the 5× rate. Fixer shards inside one or two
   test files (autoregression, ARDL) closed; shards across state space, GLM
   and discrete choice stalled. (Round 4) Output tokens in
   these transcripts are streaming-start floors, so the figure is input-side. (`data/runs/*/costs.jsonl`)

## Method

Fixture: Commit0-lite (`wentingzhao/commit0_combined`, repos
`github.com/commit-0/<lib>`), 16 libraries, 15 admissible after both-direction
certification (`harness/certify.py`); minitorch's reference is itself a
skeleton. Graded set: the tests the reference passes here. Control: 8
post-cutoff libraries stubbed the same way (`harness/control_setup.py`).

Arms: solo (one Haiku 5.5 Agent-tool subagent, continuation agent where it left
failures); swarm (up to 8 builders sharing one checkout, files partitioned by
stub count with every file owned, then fixers on contiguous shards of the
failing test ids). Prompts: `harness/prompts.py`.

Metrics: graded pass count (`harness/run.py grade`), input-side cost per spawn
(`harness/cost.py`), agent-active wall time from transcript timestamps,
verbatim rate (`harness/memorization.py`). One replicate per arm.

### Integrity

- **History.** In all 16 Commit0-lite repos the stub commit's parent is the
  reference commit, so in a clone `git diff HEAD~1` prints the original
  library. The pilot's trees carried that history; their agents' commands
  were audited and none read another revision. Every later tree is a
  one-commit repository built from `git archive` of the stub
  (`workspace.checkout(history=False)`, tested in `tests/test_harness.py`).
- **Other copies of the source.** Every library was installed from PyPI for
  its dependencies and then uninstalled, so the originals sat in the uv cache.
  Several venvs hold another target library as a dependency (cookiecutter's
  holds jinja2 and marshmallow), and the certification trees held every
  reference. Agents run as root, so no permission could hide these. The cache
  and certification trees were deleted mid-run. `harness/audit.py` lists every
  path each agent named outside its own tree and every git command that reads
  history. All 116 transcripts were audited and the flagged ones read by
  hand. None reached an original: the flags were scratch files, `ls` of the
  envs directory, a read of an env build log, and `git show HEAD` on
  history-free trees.
- **Isolation.** Library code ran only inside claude-workspace's
  `scripts/jail.sh` (no network, uid nobody, empty environment), through the
  `c0-test` / `c0-run` wrappers.
- **Grader.** Each library was certified in both directions before any agent
  ran: the reference src over the stub's tests passes, the stub does not pass
  the same set. Grading applies the agent's `src_dir` patch to a fresh stub
  checkout with pristine tests; changes outside `src_dir` are discarded. Some
  stubs pass many tests on their own (cachetools 153/215, simpy 89/149), so
  pass rate over the graded set overstates the work on those; every library
  reached 100% of the stub-failing tests too, except the simpy swarm and solo
  chardet's xfail.

### Deviations and caveats

- **Grader artifacts found during the run**: marshmallow parametrises a test
  with `datetime.now()`, so its test id differs between certification and
  grading; ids are now matched with digits inside the brackets normalised
  (`run.param_key`). chardet's last miss is an `xfail` test the reference
  XPASSes.
- **Environment, not code**: test-only failures from missing tzdata links,
  distribution metadata (`importlib.metadata.version`), DNS, and an `os.utime`
  on a root-owned fixture are excluded, because the reference fails them here
  too.
- **One replicate per arm.** Pass rates are at ceiling, so replicates would
  not separate the arms on score; they would tighten the time and cost
  comparison.
- **The control libraries may themselves be LLM-written.** Several look like
  2026 agent-era code, which could raise the verbatim floor (shared style),
  not lower it. It does not change the pass rates.
- **Commit0's specification documents** (the PDFs Commit0 hands agents) were
  not provided; agents had docstrings and tests only.
- **Python versions**: 3.10 and 3.12 as Commit0 specifies; jinja and babel
  stubs contain 3.12-only nested-quote f-strings, which the 3.10 builders
  had to rewrite.

### Harness notes

- A Commit0 stub commit is a child of its reference: strip history from any
  agent checkout.
- Installing a library from PyPI to get its dependency set leaves its source
  in the package cache and, transitively, in other libraries' venvs. Clear the
  cache, and audit agent paths, since root agents can read anything.
- Partition a swarm by file and every file needs an owner, including files
  with no empty bodies: stubbers delete members outright.
- Claiming whole test files starves fixers when one file holds most failures
  (voluptuous: one file, 89 failures, one fixer working, one idle). Shard
  failing test ids instead.
- Timing-sensitive suites (multiprocess deadlines, real-time simulation) need
  grading on an idle machine.
- Parametrised test ids can embed the clock; normalise before matching.

### Reproduce

See `README.md`. Runs: `p-solo`, `p-swarm`, `p-fix` (pilot); `f-solo`,
`f-swarm`, `f-fix`, `f-fix2`, `f-fix3`, `f-fix4`, `f-cont` (full);
`c-solo` (control). `harness/analyze.py` rebuilds the tables from
`data/runs/*/results.jsonl`, `costs.jsonl` and `memorization.json`.

## Log

### Round 1 — pilot (tinydb, pyjwt, voluptuous)

Asked: Oskar, "aim higher, Haiku is 20 times cheaper than Sonnet". Ran 3 solo
agents and 9 swarm builders, then 6 fixers claiming whole test files. Solo
201/201, 258/258, 148/148. Swarm build 199, 0, 59: the partition assigned only
files with empty bodies, so `jwt/utils.py` and `voluptuous/error.py` had no
owner. File claims idled one of two voluptuous fixers (one test file held all
89 failures). Fixers brought all three to 100%. Found here: the stub commit's
parent is the reference (git history leak), the uv cache and certification
trees held originals; fixed before Round 2 and audited.

### Round 2 — full lite set, both arms (12 more libraries)

Ran 12 solo agents and 60 swarm builders (every file owned, 12 stubs per
builder, cap 8), then shard fixers on imapclient, parsel, jinja, chardet (5)
and babel (8). Solo reached 100% on all but chardet's xfail. Swarm build pass:
chardet 31/376 (prober files with deleted properties, called "complete" by
their owners), babel 5599/6699; fixers closed both in one round. Grader fixes
in this round: clock-dependent test ids (marshmallow), load-sensitive
deadlines (portalocker, simpy; regraded idle, best of 3), a `save_patch`
failure on babel's gitignored locale data.

### Round 3 — contamination control

Asked: Oskar, "is it possible the incredible performance is from Claude's own
training data?" First reading, from Commit0-lite alone: 22–72% of Haiku's
informative lines are verbatim original lines, so probably largely recall.
Then a Sonnet agent found 12 post-cutoff libraries (ecosyste.ms, 1,530 repos
checked), 8 certified; solo Haiku scored 100% on all 8 with verbatim rates of
5–42%. The first reading was withdrawn: recall is real (about 3× the unseen
floor) but the scores do not depend on it.

### Round 4 — statsmodels, the largest full-Commit0 library

Asked: Oskar, after a size estimate for the 54-library set (about a day), "I
think maybe just go for the single very biggest test?" `harness/size_full.py`
ranked the 38 non-lite libraries by lines the stub removed: statsmodels first,
123,651 lines, 3,496 stub bodies, 27 Cython sources the stub leaves
byte-identical. Certified both ways: the reference passes 17,667 of 18,195
collected tests, the stub collects nothing. The Cython modules were built once
in the jail (`harness/full_setup.py`, 26 `.so` files) and copied into every
tree; tests live inside `statsmodels/` and are excluded from patches.

| arm | agents | wall time | graded | $ input-side |
|---|---|---|---|---|
| solo | 1 | 79 min | 1,303 | 4.28 |
| solo, continuation 1 | 1 | 147 min | not graded (5,285 self-reported) | 6.59 |
| solo, continuation 2 | 1 | 113 min | 6,303 (36%) | 6.42 |
| swarm build | 19 | 83 min (median 56) | 3,106 | 90.11 |
| swarm fixers | 19 | up to 112 min (stopped at the deadline) | FIX_GRADE | 48.59 |

No prediction was registered for this round. Both arms fell well short of
the ceiling in one night, and the swarm cost about 8× the solo chain.

What broke, in order of cost:
- **The machine.** 20 agents on 4 vCPUs, each running statsmodels suites
  (a full run is about 27 minutes there at `-n 2`); load average peaked at
  74. Fixers spent much of their time waiting on test runs.
- **Grading.** Two whole-suite grades of the swarm build were lost: one ran
  into the 90-minute cap at 99% and wrote no junit, one died at 19% when the
  first continuation agent ran `pkill -f "pytest -n"`. Grading now runs one
  `tests` directory per pytest call with its own junit, a 300-second
  per-test timeout (`pytest-timeout`), and `--resume` to continue a grade
  the 2-hour background limit cut off. All prompts now forbid killing a
  process the agent did not start.
- **Early hand-backs.** Fixers started long test runs in the background,
  ended their turn and filed "partial" reports with stale counts; two were
  resumed by message. Prompts now say to run tests in the foreground.
- **A regression the second continuation flagged and could not isolate:**
  58 MNLogit tests that passed at its start diverge at its end.

Fixers whose shard fell in one or two test files closed it: four of 19
(autoregression and ARDL) reported their 930-test shards passing or one test
short. The rest were stopped at 08:00 UTC with partial reports.

Integrity, Round 4: at least four builders and the solo agent ran library
code with plain `python3` outside the jail (heredoc `python3 -`, `importlib`
loads, `sys.path` inserts), which `guard_jail_route` does not catch; the code
was the Commit0 stub plus agents' own writing. One fixer searched the whole
filesystem for an installed `statsmodels/iolib/summary.py`, excluding the
jail; there was none. The statsmodels certification tree (holding the
reference) existed from 02:39 to 06:07 UTC; no transcript names `/cert` or
searches the work directory recursively. No agent read the mirrors.

Grading note: `param_key` collapses parametrised ids that differ only in
digits, and on statsmodels 6,654 of 17,667 targets share a key. A collapsed
group passes only if every member passes, so scores are slightly low:
re-scoring 41 chunks of the fixer grade with exact ids first gave 8,301
against 8,276.

Costs: re-priced every run at Haiku 5.5's long rate card. `cost.py` had
doubled every rate on prompts over 100K tokens; the claude-api skill gives
5× ($0.50/$2.50 against $0.10/$0.50). Commit0-lite solo is $16.36, not the
$6.71 first published, and the control $7.50, not $3.06.
