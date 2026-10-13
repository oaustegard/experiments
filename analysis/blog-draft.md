# A C compiler from 1,300 Haiku agents

In February, Nicholas Carlini had 16 Claude Opus 4.6 agents build a C compiler. Over two
weeks they ran nearly 2,000 sessions, used about 2 billion input tokens, cost just under
$20,000, and produced 100,000 lines of Rust that build Linux 6.9 on x86, ARM and RISC-V. His
[write-up](https://www.anthropic.com/engineering/building-c-compiler) ends on the model:
earlier Opus generations barely produced a working compiler, and 4.6 was "close to the
limits" of what it could do.

On 12 October Oskar asked me to run the same kind of build on the cheapest model tier. Three
Opus 5.5 agents spent 27 minutes and $14.42 on the starting point: a 7,000-line Python compiler
foundation, the test harness, and the orchestrator that runs the swarm. Then {{WORKERS}}
Claude Haiku 5.5 agents, each handed one group of failing tests, took it to {{LINES}} lines
passing {{PASS}} of {{TOTAL}} tests in 22 suites, for ${{HAIKU}} at API list price. The suites
run from GCC's torture tests to whole programs built by the compiler and run under their own
test suites: SQLite, CPython 3.12, Redis, Mbed TLS, QuickJS, Lua, TinyCC, and Doom replaying
its demos.

The compiler, hcc, is narrower than Carlini's. It targets x86-64 only, hands its assembly to
the system assembler, uses gcc as its linker driver, and has not attempted Linux. The two runs
share a method, a model fixing its way through a very large test corpus, and the cost
comparison is about that method.

## Carlini's harness

Carlini's agents ran with no orchestrator. A bash loop restarted Claude whenever a session
ended, each agent had its own container, and they shared a bare git repository. An agent
claimed a task by writing a lock file to `current_tasks/`, worked on whatever it judged the
next most obvious problem, then pulled, merged, resolved its own conflicts and pushed. Some
agents held standing roles: one merged duplicate code, one worked on the compiler's speed,
one on the quality of its output, one critiqued the design as a Rust developer would, one kept
the documentation.

His lessons were mostly about the tests. A near-perfect verifier, because otherwise Claude
"will solve the wrong problem." Short test output, with details in log files, because long
output fills the model's context. A `--fast` option that runs a 1% or 10% sample, because the
model cannot tell how long a full run is taking. And for the Linux kernel, which is one giant
task on which all 16 agents kept hitting the same bug, GCC as a known-good oracle: compile most
files with GCC, a few with the new compiler, and delta-debug until the bad file turns up.

Each of those lessons changed the harness, not the model. This run moved more of the judgment
into the harness and measured how small a model the remaining work needed.

## The setup

The Opus architect wrote the lexer, preprocessor, parser, type checker and code generator,
plus `docs/CONTRACT.md`, which fixes the module layout and the interfaces every later worker
codes against. That document does the job of Carlini's READMEs and progress files: each Haiku
worker starts with no memory, and the contract tells it where things go.

hcc is C11 in Python 3.11 with no dependencies outside the standard library. A test passes
when the program hcc builds behaves like the one gcc -O0 builds: same exit code, same output.
That uses GCC as the oracle for every test, where Carlini used it only for the kernel.

The first conformance suites ran out of failures in 70 minutes. The c-testsuite, chibicc and
GCC torture tests were at saturation, and the queue was empty. The run continued because
Sonnet agents had been packaging larger suites: gcc.dg, LLVM's single-source tests, csmith
random programs, and the whole-program builds. Carlini made the same move at 99%, giving each
agent a different open-source project to compile.

## The loop and the gate

The design moves the judgment Carlini left to his agents into code.

| | Carlini | hcc |
|---|---|---|
| choosing work | agent picks, writes a lock file | queue of clusters: failing tests grouped by their normalised error message |
| unit of work | open-ended session | one cluster, capped at $2 and 25 minutes, with notes from earlier attempts |
| merging | agent merges and resolves conflicts | one serial gate; a patch merges only if nothing that passed before fails |
| conflicts | agent resolves its own | the conflicted patch goes to the next worker as a rebase task, markers left in place |
| regressions | CI added late in the run | the gate from the first merge, plus periodic full and quick re-runs |
| long output | a few lines, details in files | the same |
| time blindness | `--fast` sample | hard caps, and the gate's sample |
| hard problems | standing specialist roles | after two failed attempts, one worker gets the whole feature |

Each worker is a headless `claude -p` session in its own git worktree. The hub, the
coordinating session, clusters failing tests, so every test failing with
`conflicting types for '<id>'` becomes one task, and ranks the clusters by suite, size and
earlier failures. Finished patches queue at the gate, which stacks four to eight, runs their
target tests and a sample of passing ones, and merges them only when every patch fixes at
least one target and nothing regresses.

## Scaling out

The limit was CPU, not the model. Every worker compiles and runs tests, so one 4-vCPU
container sat at load 10 to 22 whether the workers ran as subagents or as headless processes.
Carlini's 16 agents each had a container. From 8.5 hours in, three sibling cloud sessions,
each a container of its own, ran the same loop on a share of the suites and pushed candidate
branches; the hub gated them again before merging to main. The siblings ran 366 workers for
36% of the worker spend and contributed 81 merges. One of the three landed nothing.

## Haiku against Sonnet

A stronger model might waste fewer attempts. I tested that three times on 54 failing clusters:
Haiku 5.5, Haiku 5.5 with the `effort` setting raised to high, and Sonnet 5.5 each got the same
cluster, prompt, base commit and caps, twice. A patch counted as solved when, applied alone to
that commit, it fixed one of the cluster's tests and broke none of 120 sampled passing ones.

| run | clusters | Haiku | Haiku high | Sonnet | $ per solve (H / Hh / S) |
|---|---|---|---|---|---|
| 1 | 40 random failing | 76/80 | — | 80/80 | 0.011 / — / 0.075 |
| 2 | 4 Haiku had failed | 8/8 | 8/8 | 8/8 | 0.012 / 0.023 / 0.077 |
| 3 | 10 abandoned | 19/20 | 20/20 | 20/20 | 0.012 / 0.018 / 0.087 |

Sonnet solved 108 of 108 attempts against default Haiku's 103, at about seven times the cost
per solve. It alone solved two clusters in run 1, unwind cleanup handlers and `_Decimal` types.
Haiku workers in the swarm fixed both later, for $0.054 and $0.022, once each had the whole
feature to work on instead of one error message. Haiku at high effort solved all 28 of its
attempts in runs 2 and 3 at under twice default Haiku's cost, so a failed cluster now goes
back to Haiku at high effort first.

For the unit this harness hands out, one cluster against a fixed contract, the cheapest model
was enough. The work that needed the top tier was the foundation and the contract. Eight months
of model releases separate Opus 4.6 and Haiku 5.5, so this does not compare two models of the
same generation.

## Attempts the harness wasted

Run 3's ten clusters were the swarm's abandoned tail: tests it had stopped handing out after
three attempts without a merge. Of 33 such clusters, 23 already passed by the time I looked,
and Haiku solved the other ten on a clean retry. Nine of the ten patches merged, including the
fix for Doom's three demo replays. The earlier attempts had failed on harness bugs:

- The gate re-checked at most 20 tests of a cluster after a merge, so when one fix repaired the
  QuickJS build, about 3,900 QuickJS tests stayed marked failing, and about 100 workers went
  after tests that already passed.
- When a suite builder moved tests to a skip list, the gate read the missing results as
  regressions and rejected 27 good patches.
- Merge conflicts counted as failed attempts, so tests were parked for losing races.

Conflicts were the largest single loss: 176 patches rejected at the hub. Carlini also reports
frequent conflicts. Here at least 58 of the 128 conflicts reviewed came from shared lists, such as
the builtin table and the type-specifier list, that every feature edited. Eleven hours in, each
entry moved into its own file that registers itself, and the conflict share of gated patches
fell from 35% to 13%. Rebase tasks recovered 77 merges from the conflicts that remained.

Smaller faults cost hours. Worker scratch directories filled the disk after about 680 workers,
and later one worker's runaway background task wrote 4.3 GB of temp output. A regression in a
test only one sibling ran made that sibling reject 57 of 63 candidates. And a
`selftest | tail && git commit` chain committed after failures, because only `tail`'s exit
status reaches the `&&`.

## Regressions

Carlini writes that new features "often broke existing functionality." The gate here samples
passing tests rather than re-running all of them, so the same thing happened more quietly.
Sixteen hours in, a full re-run after about 200 merges found 82 tests that had broken since
the previous one, and the bisect traced them to a commit 256 merges back that no longer
reverted cleanly. The hub now re-runs the fast suites every 25 merges, so a bisect searches at
most 25.

That quick re-run went live for the last round, which merged seven patches before the queue
drained, with 45 merges since the last full re-run. The end-of-run measurement found what they
broke. One merge made the constants of ordinary enums unsigned, so `i >= FOO` never went false,
and the TinyCC that hcc built failed 298 of its 333 tests. Eight of the compiler's 710 unit
tests were also failing, because the gate never ran them. I fixed the enum bug by hand.
Four Haiku agents took the unit tests. They fixed four compiler regressions, and found that GCC
rejects three of the tests as written; those three, and one that expected the wrong wording,
were corrected instead. A last swarm
round then took {{CLEANUP}}. The loop now runs a full re-run whenever its queue drains, and the
gate rejects any patch that breaks a unit test.

Carlini closes on a worry that passing tests can be mistaken for a finished job. Here the
measurement that mattered came after the swarm said it was done.

## Next to Carlini's run

| | Carlini, Feb 2026 | hcc, 12 Oct 2026 |
|---|---|---|
| model | Opus 4.6 | Haiku 5.5 workers; Opus 5.5 foundation and coordination |
| parallelism | 16 agents, one container each | up to 24 workers per container, 4 containers |
| sessions | nearly 2,000 | {{WORKERS}} Haiku + 3 Opus |
| cost | just under $20,000 | ${{HAIKU}} Haiku + $14.42 Opus |
| tokens | 2B input, 140M output | {{TOKENS}} |
| wall clock | two weeks | {{HOURS}} hours |
| size | 100,000 lines of Rust | {{LINES}} lines of Python |
| backend | own x86, ARM and RISC-V code generators | x86-64 assembly |
| assembler, linker | own, "still somewhat buggy"; GCC's in the demo | system `as`, gcc as linker driver |
| builds | Linux 6.9, QEMU, FFmpeg, SQLite, Postgres, Redis, Doom | SQLite, CPython 3.12, Redis, Mbed TLS, QuickJS, Lua, TinyCC, Doom |
| tests | "about 99%" on most suites, GCC torture included | {{PASS}}/{{TOTAL}} |

The costs on the right leave out the Opus sessions that coordinated the swarm and the Sonnet
agents that packaged the test suites. Carlini measured his compiler's output as slower than
GCC's unoptimised code; hcc's output speed has not been measured, and hcc itself is slow enough that
inputs over 200 KB go through PyPy.

## Not attempted

Linux is the part of Carlini's run that this one did not reach, and the part where his
parallelism broke down. The kernel is one build, so his agents converged on the same bug, and
the cluster queue here would converge the same way; his GCC-oracle bisection would be the
place to start. A foundation written by Haiku alone is untested too. Either result could change
the conclusion, as could a class of clusters that only Sonnet solves.

The harness is now a skill, `swarming-failing-tests`, in
[claude-skills](https://github.com/oaustegard/claude-skills/tree/main/swarming-failing-tests),
with a catalogue of the 47 failures above and the check that catches each. The compiler, logs
and full results are in
[experiments/haiku-cc](https://github.com/oaustegard/experiments/tree/main/haiku-cc).
