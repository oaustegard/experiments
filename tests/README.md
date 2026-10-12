# hcc tests

Everything under `tests/` belongs to the test infrastructure. Swarm workers must not edit it.

## Running tests: `tests/run.py`
```
python3 tests/run.py --suite smoke                 # default --cc 'python3 -m hcc', --jobs 4
python3 tests/run.py --suite c-testsuite,chibicc --json out.json -q
python3 tests/run.py --suite gcc-torture --filter 'gcc-torture/pr1*' -v --keep /tmp/k
python3 tests/run.py --suite all --cc gcc          # validate the suites themselves
python3 tests/run.py --suite smoke --baseline before.json   # prints REGRESSION/FIXED lines
```
Each test is one C file, with test id `<suite>/<path relative to suite dir>`. The pipeline:
1. `<cc> <cflags> -S -o t.s test.c` (timeout 60s). Stages: preprocess, parse, sema, codegen.
2. `as --64 -o t.o t.s` (stage assemble). 3. `gcc -no-pie t.o <support> -lm` (stage link).
4. `./t`, with stdin `/dev/null` and a 10s timeout (stage `run` if killed by a signal or abort()).
5. Compare the exit code and stdout with the expectations (stage `compare`).

Output is one line per test, `PASS|FAIL|TIMEOUT|CRASH <id> <reason>`, followed by a `SUMMARY` line that includes a per-stage
breakdown. `--json` writes `{id: {status, reason, stage}}`. `stage` is null for PASS and `timeout` for TIMEOUT.
CRASH means the compiler itself crashed: exit 70, a Python traceback, or a signal. A miscompiled program that segfaults
is a FAIL with stage `run`.
Failing compiles are classified by the `[stage]` tag in hcc's error line (docs/CONTRACT.md). If the tag is missing, the
stage is taken from the innermost `hcc/` frame of the traceback. If that is missing too, the runner reruns with `-E` and
reports preprocess if that fails, otherwise parse. Other flags: `-v` prints the full log, `--keep DIR` keeps `t.s`, `t`
and `log.txt` per test, `--list` lists ids, `--include-skipped` runs skipped tests, `-q` hides PASS lines.
The exit status is 0 only if every test passed.

## Suite layout (`tests/suites/<name>/`)
* `*.c`: the tests. Optional per-test files: `<t>.c.expected` (exact stdout), `<t>.c.exitcode` (default 0), and
  `<t>.c.flags` (extra cflags).
* `suite.json` keys: `glob` (default `["*.c"]`), `cflags` (`{suite}` is replaced by the suite dir), `gcc_cflags` (added
  only when `--cc` is a real gcc/clang), `support` (C files built once by host gcc and linked in), `ldflags` (default
  `["-lm"]`) and `expect_exit`.
* `skip.txt` has one `<relpath>  # reason` per line. Skipped tests are not run and not counted as failures.

| suite | active tests | source / notes |
|---|---|---|
| smoke | 34 | Hand-written and ordered by difficulty. 00-25 need no system headers. Use it for fast gating. |
| c-testsuite | 220 | github.com/c-testsuite/c-testsuite `tests/single-exec`. `.tags` lists the features each test needs. |
| chibicc | 41 | rui314/chibicc `test/`. Uses GNU statement expressions and `support/common.c`. `adapt.py` documents the edits: lines relying on chibicc-only or UB behaviour are commented out as `/* hcc-tests removed (...) */`. |
| gcc-torture | 1632 (+66 skipped) | gcc.c-torture/execute, top level only (no ieee/ or builtins/). Pass = exit 0. `tags.json` holds feature tags (vector, int128, complex, computed-goto, inline-asm, builtin, knr...) for prioritising. |

With `--cc gcc`, all four suites pass 100%. Rebuild the torture suite with
`python3 tests/scripts/import_torture.py <gcc sparse checkout>` (see the script's docstring). It drops ieee/ and
builtins/, unsupported `dg-require-effective-target` features (trampolines, untyped_assembly, expensive, dfp),
GCC-only extensions that clang rejects (nested functions, VLAs in structs...) and tests gcc -O0 itself fails.
hcc must accept the per-test flags it can be given: `-std=gnu89`, `-fgnu89-inline` and `-fwrapv`.

## csmith differential testing: `tests/csmith/`
```
tests/csmith/build.sh                       # one-time: builds csmith into tests/tools/csmith (gitignored)
python3 tests/csmith/diff.py --cc 'python3 -m hcc' --seeds 1:200 --jobs 4 [--json out.json]
```
Each seed is built with `gcc -O0 -w` (the reference) and with `--cc` (`-S`, then `as`, then link). Both programs run with a
10s timeout and their `checksum = ...` lines are compared. Each seed gets one result:
`OK|MISMATCH|CRASH|COMPILE_FAIL|ASM_FAIL|LINK_FAIL|RUN_FAIL|TIMEOUT|SKIP`. SKIP means csmith or gcc itself failed or
timed out. Failing seeds are saved to `tests/csmith/failures/<seed>.c` and `<seed>.json` (kind, reason, csmith args,
both outputs, the cc's stderr and the include dir). The csmith options keep programs small and C11 (`--no-packed-struct`,
4 functions, low expression complexity). Pass extra options with `--csmith-arg=...`.

## Reducing failures: `tests/reduce.py`
```
python3 tests/reduce.py tests/csmith/failures/42.c --mismatch            # wrong output vs gcc -O0
python3 tests/reduce.py bug.c --compile-fail 'KeyError|\[sema\]'        # cc -S fails, stderr matches
python3 tests/reduce.py bug.c --predicate 'sh my_check.sh'              # exit 0 = still interesting
```
The reducer uses `cvise` if it is installed (`apt-get install cvise`). Otherwise, or with `--no-cvise`, it falls back to a
built-in line-level and then token-level delta-debugging reducer. With `--mismatch`, gcc must still build the candidate with
`-Werror=implicit-function-declaration` and similar flags, and the result must run cleanly under ASan and UBSan, so
reductions stay free of undefined behaviour. With `--compile-fail`, gcc must still accept the file. `-I` adds include
dirs; csmith failures pick theirs up from `<seed>.json`. `--preprocess` flattens the input with `gcc -E -P` first.
The output goes to `<file>.reduced.c`, or the path given with `-o`.

## Driver suites (real programs): `lua`, `sqlite`, `quickjs`
A suite whose `suite.json` has `"driver": "<name>"` is run by `tests/drivers/<name>.py` instead of the single-file pipeline.
The driver supplies `list_tests(suite)` (test ids are still `<suite>/<relpath>`) and `run_test(suite, rel, cc_argv, work, R)`
returning a `Result`. `skip.txt` and `--filter` work as usual. Shared machinery is in `tests/drivers/common.py`:
hcc-compiled objects and linked programs are cached on disk per hcc source hash (every `hcc/**/*.py` file), gcc-built
reference programs and their outputs per gcc version and source hash, so a whole program is compiled once per hcc revision,
not once per test. Cache: `$HCC_TEST_CACHE` or `<main checkout>/tests/cache` (shared by swarm worktrees through
`git rev-parse --git-common-dir`); downloaded sources: `$HCC_TEST_TOOLS` or `<main checkout>/tests/tools`. Both are
gitignored. `--cc gcc` runs the same pipeline with gcc as the "compiler under test" and must pass 100%.
Only compilation is done by hcc; `as`, linking and running use the host tools. A compile failure caches its error, so the
same failure is not re-derived by every test.

| suite | tests | what a test is |
|---|---|---|
| lua | 65 (+2 skipped) | Lua 5.4.7 (`src/`, committed). `lua/src/<f>.c`: hcc compiles that one file, links it with gcc -O0 objects of the others, and `smoke.lua` must print what a gcc-built lua prints (luac.c: `luac -l -l`, `-p`, dump + run). `lua/testes/<f>.lua`: an official lua-5.4.7-tests file run by a lua built entirely by hcc with `-e"_U=true; _port=true; _soft=true; _nomsg=true"`; exit code and normalised stdout must equal the gcc-built lua's (timings, random seeds and pointers are normalised in `drivers/lua.py`). If lvm.c's computed-goto build fails, the whole-program build falls back to `-DLUA_USE_JUMPTABLE=0`; the `lua/src/lvm.c` test keeps the default. A whole-program build failure is reported as `lua build: <file>.c: <first error>` for every official test. |
| sqlite | 184 | SQLite 3.49.1 amalgamation (`sqlite3.c` + `shell.c`, downloaded on first use into `tests/tools/sqlite`, sha256 pinned; flags in `suite.json`: FTS5, RTREE, math functions, no threads, no load_extension). `sqlite/sql/<category>_<nn>_<slug>.sql`: 181 small scripts (DDL, DML, SELECT, joins, aggregates, windows, CTEs, JSON/JSONB, FTS5, string/printf/date/math functions, type affinity, transactions, pragmas, blobs, collations, rtree, error messages, shell dot-commands) fed to `sqlite3 -batch :memory:` built by hcc; exit code, stdout and stderr must equal a gcc -O0 build's. `sqlite/build/{sqlite3.c,shell.c}` only compile one file; `build/version` starts the shell. The 9 MB `sqlite3.c` is compiled once per hcc revision (cached, failures too). Scripts are generated by `tests/scripts/gen_sqlite_scripts.py` from `sqlite_scripts_{a,b}.py` and committed. |
