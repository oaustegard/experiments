# harness-bench — three harnesses over Aider Polyglot at a fixed model

**Date**: 2026-09-06. **Model**: `claude-haiku-4-5`, identical in all three arms.
**Harness**: Claude Code on the Web; each arm is a batch of subagents, one per
language. **Pre-registration**: [`PLAN.md`](PLAN.md), written before any agent ran.

Prompted by [dollspace.gay's 2026-09-06 post](https://bsky.app/profile/dollspace.gay/post/3muumpa6wlc2d)
that a self-built harness (Peritus) beat the leader on a harness benchmark. The
question here is what our own stack scores under the same kind of measurement.

## Result

Twelve exercises, three harnesses, one model. Nothing changes between arms except
what the harness does with a failed attempt.

| task | oneshot | +1 retry | tool loop |
|---|---|---|---|
| python/robot-name | fail | **PASS** | PASS |
| python/pov | fail | fail | **PASS** |
| python/two-bucket | PASS | PASS | PASS |
| python/poker | fail | fail | **PASS** |
| go/zebra-puzzle | fail | fail | fail |
| go/wordy | fail | **PASS** | PASS |
| go/alphametics | fail | **PASS** | PASS |
| go/bottle-song | fail | **PASS** | PASS |
| rust/book-store | fail | **PASS** | PASS |
| rust/acronym | fail | fail | **PASS** |
| rust/doubly-linked-list | PASS | PASS | PASS |
| rust/variable-length-quantity | PASS | PASS | PASS |
| | **3/12** | **8/12** | **11/12** |

| comparison | recovered | regressed | exact McNemar, two-sided |
|---|---|---|---|
| oneshot → +1 retry | 5 | 0 | p = 0.0625 |
| +1 retry → tool loop | 3 | 0 | p = 0.2500 |
| oneshot → tool loop | 8 | 0 | p = 0.0078 |

No task regressed under a richer harness, in any comparison. No agent in any arm
wrote a file outside the solution set the exercise declares.

## Which failures the loop recovered

Four of the five retry recoveries are not reasoning failures:

- `go/alphametics` — `"regexp" imported and not used`. The implementation was
  already correct and the package would not compile.
- `go/bottle-song` — number words capitalised at line starts and lowercase
  mid-sentence.
- `go/wordy` — the parser read `"multiplied by"` before the sign of a negative
  operand, so every negative result came back `ok=false`.
- `python/robot-name` — `reset()` did not force a different name when the RNG
  was re-seeded to the same value.

Each is reported verbatim by one build or one test run. The gap between 3/12 and
8/12 is mostly the cost of not looking at the output, not a difference in what
the model knows.

The three the tool loop adds on top — `pov`, `poker`, `acronym` — needed more
than one round: 3, 1 and 2 checks respectively, against algorithms that were
wrong rather than mistyped (tree reorientation, a 5-high straight flush ranked
above a 6-high straight, camelCase word boundaries).

`go/zebra-puzzle` survives every arm. The oneshot attempt was a 5!^5 brute force
that ran past the 420 s timeout; the retry backtracked and got under 3 s with a
wrong answer; the tool loop spent all 6 checks and still returns
`DrinksWater: "Japanese"`. Six runs of a suite that says only which two fields
are wrong is not enough signal to find a bad constraint.

## The same benchmark at Opus 5

The Haiku arms above measure the loop. They say nothing about how the stack
scores, which was the original question. A second task set answers it: 30
exercises, 10 each python/go/rust, certified the same way, run at Opus 5.

| arm | protocol | score | Wilson 95% CI | P(X ≥ k \| p = 0.880) |
|---|---|---|---|---|
| `opus-oneshot` | write the file, no test access, no iteration | **29/30** = 0.967 | [0.833, 0.994] | 0.110 |
| `opus-toolloop` | agent runs the hidden suite, ≤ 6 checks | **30/30** = 1.000 | [0.886, 1.000] | 0.022 |

The reference is aider 0.86.0 + gpt-5.2 at 0.880 over all 225 tasks.

The single one-shot failure was `python/dot-dsl`, on the exact text of a raised
error: `Unknown item 99` where the suite asserts `Unknown item`. The tool loop
recovered it on its second check. Twenty-four of the 30 tool-loop exercises
passed on the first check run; the arm spent 31 checks against a budget of 180.

**30/30 exhausts the task set.** The ceiling stop that did not fire at Haiku
fires here in the other direction: a set the harness sweeps cannot measure how
much better than 0.880 the stack is, only that it is not worse. The interval is
the honest statement — at least 0.886 with 95% confidence, and no upper
resolution. Reading 1.000 as a leaderboard score would repeat the mistake this
file's Limits section already names, with 30 tasks against 225 and three
languages against six.

Where the remaining headroom sits, for anyone extending this: run the full 225,
or move to a benchmark whose tasks span files. Aider Polyglot at Opus is done.

## Grader

Every task is admitted only if the grader goes both ways on it: reference
solution PASS, untouched stub FAIL (`harness/certify.py`). Grading rebuilds a
pristine tree from the upstream exercise and overlays only the files
`.meta/config.json` names as `files.solution`, then reports anything the agent
touched outside that set. Arms A and B never see a test file; arm C runs the
suite through `harness/runtests.py`, which grades from the same pristine tree in
a scratch directory and rewrites its path out of the output, so the loop gets
failure evidence without the assertions entering its context.

Two defects the certification caught before any agent ran:

- A shared `CARGO_TARGET_DIR` let a `todo!()` stub reuse the gold arm's compiled
  test binary and report **23 passed**. Now per task.
- `cargo test` without `-- --include-ignored` scores an Exercism Rust stub green
  on 1 passed / 11 ignored.

Two tasks were rejected: `go/markdown`, a refactoring exercise whose stub already
passes, and `rust/robot-name`, whose reference solution needs a crate the
exercise's `Cargo.toml` does not declare.

## Limits

**Not comparable to the leaderboard.** The entry to sit next to on
[TheLime1/harness-bench](https://github.com/TheLime1/harness-bench) is aider
0.86.0 + gpt-5.2 at 0.880 over all 225 tasks. This is Haiku 4.5 over 12,
scored pass-at-attempt rather than aider's protocol. 11/12 is not 0.917 on the
same axis and should not be quoted as one.

**n = 12.** The oneshot → tool loop comparison clears p = 0.01; neither adjacent
step does on its own. Twelve paired tasks can show a direction and cannot size
the effect.

**One replicate, no temperature control.** Each arm ran once.

**Arm C got a bigger budget than arm B.** The tool loop had up to 6 suite runs
per exercise where the retry arm had exactly one round. Some of the 8 → 11 comes
from the extra attempts rather than from better feedback, and this design cannot
tell those apart.

## The ceiling stop

The pre-registered ceiling stop did not fire. This is the first arm comparison
in the repo where the weak arm fails often enough to leave something to measure —
`orchestrated-coding-pareto` (haiku-solo 14/14 = opus-solo 14/14) and the
[PR #76](https://github.com/oaustegard/experiments/pull/76) stage-1 pilot
(sonnet-low 14/14) both went vacuous. Aider picked these 225 exercises from the
Exercism tracks because the models of the day failed them; that selection is the
difference from a task bank we authored ourselves.

## The full run

203 exercises, all six languages, Opus 5, single-shot: write the file, no test
access, no iteration. Of the three arms, this one is shaped like aider's
protocol, so it is the one to put next to the leaderboard.

**182/203 = 0.897**, Wilson 95% CI [0.847, 0.931].

Against aider 0.86.0 + gpt-5.2 at 0.880 over all 225 tasks:
P(X ≥ 182 | p = 0.880, n = 203) = 0.274. The two are not separable at this
sample size. At par is the claim the data supports.

| language | score | | language | score |
|---|---|---|---|---|
| javascript | 46/48 = 0.958 | | java | 35/39 = 0.897 |
| python | 32/34 = 0.941 | | rust | 20/24 = 0.833 |
| cpp | 20/22 = 0.909 | | go | 29/36 = 0.806 |

No files touched outside the declared solution set in any of the 203.

### The 22 excluded exercises

Certification admitted 203 of 225. Six exercises are refactoring tasks whose
stub already passes (`ledger` in go, javascript and java, plus `go/markdown`,
`go/counter`, `java/tree-building`) and carry no signal in either direction.
Sixteen have reference solutions that will not build here — mostly rust reaching
for crates the exercise `Cargo.toml` does not declare, plus 4 cpp and 6 java.

The excluded 22 are not a random sample, so 0.897 over 203 is not
interchangeable with a score over 225. Which way the exclusions push is
unmeasured: dropping the six stub-passes removes free points, and dropping the
sixteen gold-fails removes exercises of unknown difficulty.

### Stub defects scored as failures

At least two of go's seven failures are not solve failures. `go/hexadecimal` and
`go/trinary` ship stubs like
`func ParseHex(in string, out int64, errCase string)` — the exercise's
test-case struct fields leaked into the signature, and the function returns
nothing, so no test can call it. Agents wrote a callable signature and the
compile failed against the real suite either way. They are scored as failures
here because that is what the grader saw.

### Three harness defects this run exposed

All three were mine, all three were found by agents' own reports or by
re-checking, and all three are fixed in the committed code:

1. `cmd_prompts` rendered `solution_files[0]` only. cpp declares a `.cpp` and a
   `.h`, and the hidden test includes the header, not the source — so a
   `.cpp`-only answer could never link, whatever it contained. The cpp arm was
   re-run from pristine stubs with both files shown. It scores 0.909; before the
   fix it would have scored near zero for a reason unrelated to coding.
2. Regenerating prompts **before** resetting the work tree put the previous
   run's solutions into the prompts as "current contents", turning a re-run into
   a retry with prior code visible. Caught when an agent reported its stub
   "already carried a complete implementation". Order is now reset → regenerate
   → dispatch.
3. Resetting a work tree while an agent from the previous wave was still writing
   to it left last-writer-wins contamination on five exercises. Those five were
   reset and re-run with nothing in flight.

Two cpp passes were discarded because defects 1 and 2 invalidated them. Neither
was ever graded.

## Cost of the full run

Measured on this container, 2026-09-06, before committing to the sweep.

Per-task model cost, from the six Opus batches: the one-shot arm spent 416,486
tokens over 30 tasks and the tool loop 408,308, so roughly **13.8k tokens per
task per arm**, or **~3.1M tokens per arm** over 225. Subagent batches of five
took 77–355 s, six at a time.

The three unrun languages all work here, with caveats measured rather than
assumed:

| language | n | runner | measured |
|---|---|---|---|
| cpp | 26 | cmake + vendored catch.hpp | ~1–3 s per build; offline |
| javascript | 49 | npm install + jest | 28 s cold install, 11.5 s warm, **127 MB of node_modules each** |
| java | 47 | `./gradlew test --no-daemon` | 21 s per run; the 8.7 distribution and Maven deps resolve through the proxy and cache to `~/.gradle` |

Two engineering gaps rather than unknowns. `certify.py` copies only the first
`example*` file, and cpp exercises declare two solution files (`.cpp` and `.h`)
and two example files — it needs to copy the whole set. And 49 × 127 MB of
node_modules is 6.2 GB against 26 GB free, so the js runner has to install and
clean per task rather than leaving them in place.

Rough totals: certification of all 225 in both directions ≈ 1.5 h, grading ≈
30–40 min per arm (cold Rust compiles and gradle dominate), dispatch ≈ 30–40 min
per arm. Call it **4–6 hours and ~6M tokens for both arms**, or half that for
the one-shot arm alone.

What it buys is a narrower interval, ±0.03 rather than ±0.09, and nothing else.
It does not make the number a leaderboard entry: aider's protocol is a specific
edit format at two attempts with no test execution, so the one-shot arm is the
only one of ours that is even close in shape.

## Reproducing

```bash
git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark
ln -s $PWD/polyglot-benchmark harness-bench/polyglot-benchmark
python3 harness/certify.py 9                              # admit tasks
python3 harness/bench.py prepare --tasks results/tasks-pilot.json --arm oneshot
python3 harness/bench.py prompts --tasks results/tasks-pilot.json --arm oneshot
python3 harness/batch.py oneshot                          # one brief per language
# dispatch the briefs to subagents, then:
python3 harness/bench.py grade --tasks results/tasks-pilot.json --arm oneshot \
        --out results/oneshot.json
```
