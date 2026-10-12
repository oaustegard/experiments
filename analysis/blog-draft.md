# A C compiler from 1,300 Haiku agents

If you have read about Nicholas Carlini's compiler, you know its shape: 16 Claude Opus 4.6
agents working in parallel for two weeks, nearly 2,000 sessions, just under $20,000, and
100,000 lines of Rust that build Linux 6.9. On 12 October I ran the same kind of build with
the cheapest model tier. Three Opus 5.5 agents wrote a 7,000-line Python foundation in 27
minutes for $14.42. Then {{WORKERS}} Claude Haiku 5.5 agents, each handed one group of failing
tests, took it to {{LINES}} lines passing {{PASS}} of {{TOTAL}} tests in 22 suites: GCC's
torture tests, gcc.dg, LLVM's single-source tests, and whole programs built by the compiler
and run under their own test suites — SQLite, Lua, CPython 3.12, Redis, Mbed TLS, QuickJS and
Doom's demo playback. The Haiku agents cost ${{HAIKU}} at API list price.

The compiler, hcc, is narrower than Carlini's. It targets x86-64 only, emits assembly for the
system assembler, uses gcc as its linker driver, and has not attempted Linux. It is Python,
so it is slow; inputs over 200 KB go through PyPy. What the two runs share is the method — a
model fixing its way through a very large test corpus — and that is what the cost comparison
measures.

## The swarm loop and merge gate

A test passes when the hcc-built program behaves like the gcc -O0 build: same exit code, same
output. The orchestrator groups failing tests by their normalised error message, so every
test failing with `conflicting types for '<id>'` lands in one cluster, and hands one cluster
to one Haiku agent: a headless `claude -p` session in its own git worktree, capped at $2 and
25 minutes, given the notes of earlier attempts. Finished patches queue at a single serial
merge gate. It applies four to eight at a time, runs their target tests and a sample of
passing ones, and merges only when nothing that passed before now fails. A patch that no
longer applies goes to the next agent as a rebase task, conflict markers left in place.
Nobody resolved a conflict by hand.

## Haiku against Sonnet

The obvious objection is that a stronger model would waste fewer attempts. I tested it in
three runs on 54 failing clusters: Haiku 5.5, Haiku 5.5 at high effort and Sonnet 5.5 each got
the same cluster, prompt, base commit and caps, twice, graded in isolation.

| run | clusters | Haiku | Haiku high | Sonnet | $ per solve (H / Hh / S) |
|---|---|---|---|---|---|
| 1 | 40 random failing | 76/80 | — | 80/80 | 0.011 / — / 0.075 |
| 2 | 4 Haiku had failed | 8/8 | 8/8 | 8/8 | 0.012 / 0.023 / 0.077 |
| 3 | 10 abandoned | 19/20 | 20/20 | 20/20 | 0.012 / 0.018 / 0.087 |

Sonnet solved slightly more, at about seven times the cost per solve. It alone solved two
clusters in run 1, unwind cleanup handlers and `_Decimal` types; Haiku agents in the swarm
fixed both later, with a whole-feature brief, for $0.054 and $0.022. Haiku at high effort
missed nothing, at under twice default Haiku's cost, so the swarm now retries a failed
cluster at high effort before anything else.

## Where the losses came from

Run 3's ten clusters were the swarm's abandoned tail: tests parked after three attempts
without a merge. Haiku solved every one of them on a clean retry, and nine of the ten patches
merged, among them the fix for Doom's three demo replays. The attempts that had parked them
were wasted by the harness: a gate state that still listed fixed tests as failing, tests
moved to a skip list counted as regressions, and merge conflicts counted as failed attempts.

Merge conflicts were the largest single loss: 176 rejected patches on the hub. Most came from
shared lists — the builtin table, the type-specifier list — that every feature touched. Moving
each entry into its own self-registering file at 13:50Z cut the conflict share of gated
patches from 35% to 13%.

The ceiling was CPU, not the model. Every agent compiles and runs tests, so one 4-vCPU
container sat at load 10 to 22 however the agents were launched. From 11:27Z three sibling
cloud sessions, each its own container, ran the same loop on a partition of the suites and
pushed candidate branches that the hub re-gated; they contributed 81 merges.

The gate's 80-test sample also let regressions through. At 19:19Z a full re-run after about
200 merges found 82 tests that had broken since the last one; the bisected culprit sat 256
commits back and no longer reverted cleanly. The fix is a quick re-run of the fast suites
every 25 merges; a bisect then searches only those 25. {{REGRESSION_OUTCOME}}

## Next to Carlini's run

| | Carlini, Feb 2026 | hcc, 12 Oct 2026 |
|---|---|---|
| model | Opus 4.6, 16 in parallel | Haiku 5.5, up to 24 per container on 4 containers |
| sessions | nearly 2,000 | {{WORKERS}} Haiku + 3 Opus |
| cost | just under $20,000 | ${{HAIKU}} Haiku + $14.42 Opus |
| wall clock | two weeks | {{HOURS}} hours |
| size | 100,000 lines of Rust | {{LINES}} lines of Python |
| backend | own x86, ARM and RISC-V code generators | x86-64 assembly, system assembler |
| headline | builds Linux 6.9 | {{PASS}}/{{TOTAL}} tests; SQLite, CPython, Redis, Doom |

Not metered: the Opus sessions that coordinated the swarm, and the Sonnet agents that built
the larger test suites when the first conformance suites ran out after 70 minutes.

The harness is now a skill, `swarming-failing-tests`, in
[claude-skills](https://github.com/oaustegard/claude-skills); the compiler, logs and the
full writeup are in [experiments/haiku-cc](https://github.com/oaustegard/experiments/tree/main/haiku-cc).
