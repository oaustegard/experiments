# commit0-swarm results — 2026-10-10

## Answer

Haiku 5.5 rebuilt every library it was given from Commit0-style skeletons:
modules, classes, signatures and docstrings kept, function bodies emptied,
with the library's own test suite as the target. One agent working alone
passed 11,208 of 11,209 graded tests across the 15 admissible Commit0-lite
libraries, including babel (6,699 tests, 65 minutes, 344 tool calls) and
jinja (850 tests, 58 minutes, 287 tool calls). The one miss is a chardet test
marked expected-to-fail that the reference happens to pass. Input-side cost
for the 15 libraries was $6.71.

Haiku 5.5 cannot have trained on eight libraries first released after
2026-07-01. It also rebuilt all eight to 100% (1,589 of 1,589 tests, $3.06),
including catraca (428 tests, 5,700 source lines), aseprite (a binary file
format, 260 tests) and verifactu-lint (254 tests). Haiku recognises the famous
libraries: a median 55% of its informative lines are verbatim lines of the
original, against 17% on the unseen ones. The scores do not depend on that,
because the unseen libraries reached the same ceiling.

A swarm of up to 8 Haiku builders per library, each owning a set of files in
one shared checkout, followed by a wave of fixers that each took a shard of
the failing tests, reached 11,205 of 11,209. It cost 1.3× the solo arm
($8.64). It finished the two largest libraries two to three times sooner:
babel in 36 agent-minutes of wall time against 65, jinja in 18 against 58.
The build pass alone reached only 9,402 of 11,209. The gap is integration
failure, and the fixer wave closed it.

## Findings

1. **One Haiku agent rebuilds a whole library.** Fifteen Commit0-lite
   libraries, solo, one attempt each (chardet also got a continuation agent,
   which changed nothing):

   | library | graded tests | solo | swarm build | swarm + fixers | builders | solo $ | swarm $ |
   |---|---|---|---|---|---|---|---|
   | babel | 6699 | 6699 | 5599 | 6699 | 8 | 2.33 | 3.68 |
   | marshmallow | 1228 | 1228 | 1228 | 1228 | 7 | 0.29 | 0.41 |
   | jinja | 850 | 850 | 848 | 850 | 8 | 2.28 | 1.23 |
   | chardet | 376 | 375 | 31 | 376 | 5 | 0.28 | 0.91 |
   | cookiecutter | 367 | 367 | 367 | 367 | 7 | 0.41 | 0.44 |
   | imapclient | 267 | 267 | 266 | 267 | 8 | 0.31 | 0.58 |
   | pyjwt | 258 | 258 | 0 | 258 | 2 | 0.14 | 0.28 |
   | cachetools | 215 | 215 | 215 | 215 | 2 | 0.02 | 0.03 |
   | parsel | 205 | 205 | 203 | 205 | 4 | 0.10 | 0.15 |
   | tinydb | 201 | 201 | 199 | 201 | 4 | 0.09 | 0.14 |
   | deprecated | 171 | 171 | 171 | 171 | 1 | 0.03 | 0.02 |
   | simpy | 149 | 149 | 145 | 145 | 7 | 0.10 | 0.20 |
   | voluptuous | 148 | 148 | 59 | 148 | 3 | 0.17 | 0.38 |
   | portalocker | 38 | 38 | 34 | 38 | 2 | 0.14 | 0.15 |
   | wcwidth | 37 | 37 | 37 | 37 | 1 | 0.03 | 0.04 |

   Graded tests are those the reference implementation passes in this
   harness. portalocker and simpy are graded best of three on an idle machine,
   because their multiprocess and real-time tests carry 0.2–1.1 s deadlines
   that fail under the load of 20 concurrent agents (the same solo tree scored
   34/38 loaded and 38/38 idle). The simpy swarm's 145 holds idle too: its
   `rt.py` has a real bug. minitorch was excluded because its reference commit
   is itself an unfinished course skeleton (reference 15/230).

2. **The control: libraries Haiku cannot have seen, same result.**

   | control library | first PyPI release | graded tests | solo | $ | verbatim rate |
   |---|---|---|---|---|---|
   | catraca | 2026-09-25 | 428 | 428 | 0.90 | 0.14 |
   | aseprite | 2026-09-05 | 260 | 260 | 0.23 | 0.27 |
   | verifactu-lint | 2026-08-07 | 254 | 254 | 0.25 | 0.20 |
   | bslfmt | 2026-09-26 | 214 | 214 | 1.32 | 0.05 |
   | africa-g2p | 2026-08-06 | 197 | 197 | 0.18 | 0.12 |
   | vstg | 2026-07-27 | 117 | 117 | 0.05 | 0.42 |
   | laga | 2026-07-24 | 67 | 67 | 0.05 | 0.23 |
   | agent-self-edit-gate | 2026-08-29 | 52 | 52 | 0.07 | 0.12 |

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
   scores. The control overturned that reading.

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
   fixer clobbered another's edit.

5. **Cost and time.** Solo totals $6.71 for Commit0-lite; the swarm $8.64
   (build $7.95, fixers $0.69). Agent-active wall time, solo vs swarm (build +
   fix): babel 65 vs 36 min, jinja 58 vs 18, cookiecutter 14 vs 7, imapclient
   21 vs 19, chardet 13 vs 32 (the integration failure). On libraries under
   ~1,000 tests the two are within a few minutes. The swarm figures exclude
   the queueing a 20-agent session cap imposes between waves.

6. **The whole experiment cost $18.41 input-side** across 116 Haiku 5.5 spawns
   (pilot $1.20, Commit0-lite solo $6.31, swarm $7.24, fixers $0.60,
   continuation $0.01, control $3.06), priced per turn with Haiku 5.5's
   double rate above 100K prompt tokens (`harness/cost.py`). Output tokens in
   these transcripts are streaming-start floors, so the figure is input-side.

## Integrity

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

## Deviations and caveats

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

## Harness lessons

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

## Reproduce

See `README.md`. Runs: `p-solo`, `p-swarm`, `p-fix` (pilot); `f-solo`,
`f-swarm`, `f-fix`, `f-fix2`, `f-fix3`, `f-fix4`, `f-cont` (full);
`c-solo` (control). `harness/analyze.py` rebuilds the tables from
`data/runs/*/results.jsonl`, `costs.jsonl` and `memorization.json`.
