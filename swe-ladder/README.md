# swe-ladder

Does a Haiku 5.5 agent that fails a real SWE-bench Verified task do better
retrying itself with the hidden tests' failure output, or escalating to Sonnet
5.5 with the same output? Follow-up to `temporal-routing-headroom`, whose 14
seeded-bug repos were too easy to separate the tiers (Haiku → Haiku went 14/14).
Results: `RESULTS.md`.

## Design

- **Tasks**: SWE-bench Verified, the two largest pure-Python repos: django (231)
  and sympy (75). A task enters the pool only if, in this harness, its gold patch
  resolves it and the empty patch does not (`data/validation.jsonl`,
  `data/pool.json`).
- **Rung 1**: Haiku 5.5 Agent-tool subagent, issue text only, one checkout per task.
- **Rung 2** on every rung-1 failure, two arms from the identical state (the
  rung-1 diff still applied, plus the failing hidden test names and their
  tracebacks): Haiku 5.5 again, or Sonnet 5.5.
- **Grade**: SWE-bench's procedure without Docker. Fresh checkout, apply the
  diff, reset the test-patch files to base, apply the test patch, run its test
  modules, parse with SWE-bench 4.0.4's log parser; every FAIL_TO_PASS and
  PASS_TO_PASS test must pass.
- **Cost**: input-side dollars per spawn from each subagent's transcript
  (`harness/cost.py`, same method as `temporal-routing-headroom`).

## Isolation

The task repos are code nobody here wrote, so nothing from them runs outside
claude-workspace's `scripts/jail.sh` (no network, uid nobody, empty
environment). Dependencies come from PyPI into one venv per (repo, version); the
repos are never pip-installed, checkouts go on `PYTHONPATH`. Agents run code
only through `swe-test` and `swe-run`, which call the jail. Two hub guards make
that mechanical rather than a prompt request: `guard_jail_route` refuses any
other execution under `/tmp/jail/`, and `guard_subagent_scope` limits subagents
to Bash/Read/Edit/Write/Grep/Glob while `/tmp/claude-subagent-allowlist.json`
exists (claude-workspace c0be8b0).

## Deviations from SWE-bench

- Python 3.5–3.7 have no uv build; those Django versions (2.2–3.2) run on 3.8.
  Validation drops any task this breaks.
- Django at verbosity 2 prints a test's docstring between its name and the
  verdict; the stock parser then keys the test by docstring and misses it.
  `grade.django_docstring_lines` records it under both keys. Before this fix
  8 of the first 11 gold patches "failed".
- Optional Django test extras needing a C toolchain (pylibmc, mysqlclient, ...)
  are skipped; tests needing them are skipped by Django itself.

## Reproduce

```bash
sh setup.sh                                  # venv, interpreters, mirrors, tasks, wrappers
.venv/bin/python harness/envs.py             # 21 dependency venvs
.venv/bin/python harness/validate.py         # gold/base check -> data/pool.json
echo '{"tools":["Bash","Read","Edit","Write","Grep","Glob"],"reason":"swe-ladder"}' \
  > /tmp/claude-subagent-allowlist.json
.venv/bin/python harness/ladder.py stage r1-haiku --rung 1 --model haiku --ids data/pool.json
.venv/bin/python harness/ladder.py next r1-haiku --slots 10   # dispatch each line as an Agent call
.venv/bin/python harness/ladder.py done r1-haiku ID...        # on each completion
.venv/bin/python harness/ladder.py grade r1-haiku
.venv/bin/python harness/ladder.py stage r2-sonnet --rung 2 --model sonnet --from r1-haiku
.venv/bin/python harness/cost.py r1-haiku
```

Workers run as `general-purpose` subagents: a restricted agent type defined
mid-session does not register, so the subagent allowlist does that job.
