# haiku-cc snapshot

Mirror of the working repo /home/user/haiku-cc (own git history, not included; published as
branch `hcc/main` of this repo) at commit 0ea26d7e, the commit of the final measurement
(2026-10-13T02:52Z), taken with `git archive`. Contents: a C11 compiler in pure Python
(`hcc/`) built out by a Haiku 5.5 swarm (`swarm/`, `federation/`) against `tests/`, and the run
statistics (`analysis/`). Results: `RESULTS.md`; charts and the numbers behind them: `figures/`.

What is mirrored from `tests/suites/`: all of `smoke/`, `sqlite/` (generated SQL scripts),
`csmith/` (reduced repros and fixed programs) and `doom/` (our headless backend); for every
other suite only its metadata (`suite.json`, `manifest.json`, `skip.txt`, `tags.json`,
`SOURCE`, `COMMIT`) and our adapter scripts. Vendored upstream sources (GCC testsuite,
QuickJS and test262, Lua, c-testsuite, chibicc, LLVM test-suite files) are left out;
`tests/scripts` and `tests/drivers` fetch or regenerate them. `tests/tools/` and
`tests/cache/` are never mirrored.

To refresh: `git archive HEAD` of the working repo into an empty directory, apply the
filter above, replace everything here except `SNAPSHOT.md`, `RESULTS.md` and `figures/`,
then copy `analysis/out/*.png`, `metrics.json` and `comparison.md` into `figures/`.
