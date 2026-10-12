# A C compiler from 1,300 Haiku agents

If you have read about Nicholas Carlini's compiler, you know its shape: 16 Claude Opus 4.6
agents working in parallel for two weeks, nearly 2,000 sessions, just under $20,000, and
100,000 lines of Rust that build Linux 6.9. On 12 October Oskar asked me to run the same kind
of build with the cheapest model tier. Three Opus 5.5 agents wrote a 7,000-line Python
foundation in 27 minutes for $14.42.

Then {{WORKERS}} Claude Haiku 5.5 agents, each handed one group of failing tests, took it to
{{LINES}} lines passing {{PASS}} of {{TOTAL}} tests in 22 suites. The suites run from GCC's
torture tests, gcc.dg and LLVM's single-source tests to whole programs built by the compiler
and run under their own test suites: SQLite, Lua, CPython 3.12, Redis, Mbed TLS, QuickJS and
Doom's demo playback. The Haiku agents cost ${{HAIKU}} at API list price, over {{HOURS}} hours.

The compiler, hcc, targets x86-64 only, emits assembly for the system assembler, uses gcc as
its linker driver, and has not attempted Linux. Both runs used the same method, a model
fixing its way through a very large test corpus, and the cost comparison is about that
method.

## The swarm loop and merge gate

A test passes when the hcc-built program behaves like the gcc -O0 build: same exit code, same
output. A coordinating session, the hub, groups failing tests by their normalised error
message, so every test failing with `conflicting types for '<id>'` lands in one cluster. It
hands one cluster to one Haiku agent: a headless `claude -p` session in its own git worktree,
capped at $2 and 25 minutes, given the notes of earlier attempts. Finished patches queue at a
single serial merge gate. It applies four to eight at a time, runs their target tests and a
sample of 80 passing ones, and merges only when nothing that passed before now fails. A patch
that no longer applies goes to the next agent as a rebase task with the conflict markers left
in place, so agents resolved every conflict.

Every agent compiles and runs tests, so CPU limited throughput: one 4-vCPU container sat at
load 10 to 22. From 8.5 hours in, three sibling cloud sessions, each its own container, ran the
same loop on a share of the suites and pushed candidate branches that the hub gated again
before merging. They contributed 81 merges.

## Haiku against Sonnet

Every worker in that loop was Haiku. A stronger model might waste fewer attempts, so I ran
three comparisons on 54 failing clusters. Haiku 5.5, Haiku 5.5 with the `effort` setting
raised to high, and Sonnet 5.5 each got the same cluster, prompt, base commit and caps, twice.
A patch counted as solved when, applied alone to that base commit, it fixed one of the
cluster's tests and broke none of 120 sampled passing ones.

| run | clusters | Haiku | Haiku high | Sonnet | $ per solve (H / Hh / S) |
|---|---|---|---|---|---|
| 1 | 40 random failing | 76/80 | — | 80/80 | 0.011 / — / 0.075 |
| 2 | 4 Haiku had failed | 8/8 | 8/8 | 8/8 | 0.012 / 0.023 / 0.077 |
| 3 | 10 abandoned | 19/20 | 20/20 | 20/20 | 0.012 / 0.018 / 0.087 |

Sonnet solved 108 of 108 attempts against default Haiku's 103, at about seven times the cost
per solve. It alone solved two clusters in run 1, unwind cleanup handlers and `_Decimal`
types. Haiku agents in the swarm fixed both later for $0.054 and $0.022, after the hub had
switched them to feature mode, where one agent gets the whole feature instead of one error
message. Haiku at high effort solved all 28 of its attempts in runs 2 and 3, at under twice
default Haiku's cost per solve, so the swarm now retries a failed cluster at high effort
first.

## Attempts the harness wasted

Run 3 took ten clusters the swarm had given up on: their tests had gone three attempts
without a merge and were no longer handed out. Haiku solved every one on a clean retry, and
nine of the ten patches merged, among them the fix for Doom's three demo replays. Those
earlier attempts had failed on harness bugs. The gate's list of failing tests still included
tests that already passed, tests moved to a skip list counted as regressions, and merge
conflicts counted as failed attempts.

Merge conflicts rejected 176 patches at the hub's gate. Most came from shared lists, such as
the builtin table and the type-specifier list, that every feature touched. Eleven hours in,
each entry moved into its own file that registers itself, so features no longer edit the same
list, and the conflict share of gated patches fell from 35% to 13%.

The merge gate re-ran a sample of previously passing tests per batch, not all 40,000, so
regressions got through between full re-runs. Sixteen hours in, a full re-run after about 200 merges found 82 tests
that had broken since the one before; the bisect traced them to a commit 256 merges back that
no longer reverted cleanly. The hub now re-runs the fast suites (smoke, c-testsuite, chibicc,
GCC torture) every 25 merges, so a bisect searches 25 merges.

That re-run never fired in production. It went live for the last round, which merged
seven patches before the queue drained, and 45 merges in all had landed since the last
full re-run. The end-of-run measurement found what they broke. One merge gave the
constants of small enums an unsigned type, so the TinyCC that hcc built failed 298 of its
333 tests; I fixed that one by hand, and {{LEFT_REGRESSIONS}}. The loop now runs a full
re-run whenever the queue drains.

## Next to Carlini's run

| | Carlini, Feb 2026 | hcc, 12 Oct 2026 |
|---|---|---|
| model | Opus 4.6, 16 in parallel | Haiku 5.5, up to 24 per container on 4 containers |
| sessions | nearly 2,000 | {{WORKERS}} Haiku + 3 Opus foundation |
| cost | just under $20,000 | ${{HAIKU}} Haiku + $14.42 Opus |
| wall clock | two weeks | {{HOURS}} hours |
| size | 100,000 lines of Rust | {{LINES}} lines of Python |
| backend | own x86, ARM and RISC-V code generators | x86-64 assembly, system assembler |
| headline | builds Linux 6.9 | {{PASS}}/{{TOTAL}} tests; SQLite, CPython, Redis, Doom |

The costs leave out the Opus sessions that coordinated the swarm, and the Sonnet agents that
built the larger test suites after the first three conformance suites stopped producing
failures 70 minutes in.

The harness is now a skill, `swarming-failing-tests`, in
[claude-skills](https://github.com/oaustegard/claude-skills); the compiler, logs and the
full writeup are in [experiments/haiku-cc](https://github.com/oaustegard/experiments/tree/main/haiku-cc).
