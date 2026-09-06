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
