# ERRORS — harness-bench

What was wrong, how it was caught, and which way it pushed the conclusion.
Everything here was caught by the grader-certification step, before any agent ran.

## 1. A shared `CARGO_TARGET_DIR` scored a `todo!()` stub as 23 passed

`bench.py` set `CARGO_TARGET_DIR=/tmp/cargo-target` for every Rust task in every
arm. The gold arm ran first and compiled `simple_cipher`; the stub arm then
reused that test binary and reported `test result: ok. 23 passed`. Re-running the
same stub with a private target dir gives `23 failed`.

Caught by the gold/stub validation, which is there precisely because a grader
that only ever runs against plausible code cannot tell whether it can go red.
Direction: would have inflated every arm's Rust score, most on the weakest arm,
which is the arm the whole comparison rests on.

Fixed: `CARGO_TARGET_DIR` is now per graded task directory.

## 2. `cargo test` scores an Exercism Rust stub green

Exercism marks all but the first test in a Rust track `#[ignore]`, so a bare
`cargo test` on the `accumulate` stub reports `1 passed; 11 ignored` and exits 0.
The suite has to run as `cargo test -- --include-ignored`.

Caught the same way. Direction: same as #1, and larger — it applies to every Rust
task rather than to whichever one happened to share a build hash.

## 3. Two tasks rejected by certification

- `go/markdown` — a refactoring exercise. The stub is working code, so the suite
  passes before an agent touches it. No signal in either direction.
- `rust/robot-name` — the reference solution imports `rand` in a form the
  exercise's `Cargo.toml` does not declare, so the gold arm cannot compile. The
  exercise may well be solvable; this environment cannot certify that it is.

Both were replaced from the same seeded pool. `results/certify.json` records the
verdict for every task considered.

## 4. Not an error, but the number most likely to be misread

The 11/12 in the tool-loop arm is not comparable to the 0.880 that
`aider 0.86.0 + gpt-5.2` scores on the same benchmark. Different model, 12 tasks
against 225, and a check budget aider's protocol does not grant. `RESULTS.md`
says so under Limits; repeating the figure without that sentence would be the
error.
