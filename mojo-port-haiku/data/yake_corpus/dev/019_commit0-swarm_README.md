# commit0-swarm

Can Haiku 5.5 build a whole Python library from its skeleton? And does a swarm
of Haiku agents working one checkout beat a single one? Follow-up to
`swe-ladder`, where Haiku 5.5 resolved 87% of SWE-bench Verified django+sympy
tasks one bug at a time. Results: `RESULTS.md`.

## Design

- **Tasks**: Commit0-lite (Zhao et al., 2024; data `wentingzhao/commit0_combined`
  on Hugging Face, repos `github.com/commit-0/<lib>`): 16 libraries whose
  function bodies were removed, keeping modules, classes, signatures and
  docstrings. A library is admissible when its reference implementation passes
  most of its own suite in this harness and the stub passes fewer
  (`harness/certify.py`); 15 of 16 are. The graded set is the tests the
  reference passes.
- **Control**: 8 libraries first released on PyPI after 2026-07-01, which
  Haiku 5.5 cannot have trained on, stubbed the same way
  (`harness/control_setup.py`, candidates in `data/control_candidates.json`).
- **Solo arm**: one Haiku 5.5 Agent-tool subagent per library, the skeleton and
  test suite only; a continuation agent where the first left failures.
- **Swarm arm**: up to 8 builders per library in one shared checkout, files
  partitioned by stub count (every source file owned), then fixers that each
  take a contiguous shard of the failing test ids and may edit any source file.
- **Grade**: the agent's `src_dir` diff applied to a fresh stub checkout with
  pristine tests, run in the jail, scored per test from junit XML.
- **Recall measure**: `harness/memorization.py`, share of the agent's
  informative added lines that are verbatim lines of the reference.
- **Cost**: input-side dollars per spawn from transcripts (`harness/cost.py`).
- **Audit**: `harness/audit.py`, every path an agent named outside its own tree.

## Isolation

Library code runs only inside claude-workspace's `scripts/jail.sh` (no network,
uid nobody, empty environment), through `c0-test` and `c0-run`. Dependencies
come from each reference commit's pyproject (PEP 621, extras resolved) and
PyPI, installed outside the jail; libraries are never pip-installed, checkouts
go on `PYTHONPATH`. Agent checkouts carry no git history, because the Commit0
stub commit's parent is the reference.

## Reproduce

```bash
sh setup.sh                                    # harness venv, interpreters, mirrors, wrappers
.venv/bin/python harness/envs.py               # one venv per library
.venv/bin/python harness/fetch_babel_data.py 2.14.0
.venv/bin/python harness/certify.py            # both-direction grader check -> data/certify.json
.venv/bin/python harness/control_setup.py      # post-cutoff control tasks
echo '{"tools":["Bash","Read","Edit","Write","Grep","Glob"],"reason":"commit0-swarm"}' \
  > /tmp/claude-subagent-allowlist.json
.venv/bin/python harness/run.py stage f-solo --arm solo --libs babel jinja ...
.venv/bin/python harness/run.py stage f-swarm --arm swarm --per-worker 12 --libs ...
.venv/bin/python harness/run.py cycle RUN [IDS]    # mark done, print Agent calls up to the 20-agent cap
.venv/bin/python harness/run.py grade RUN
.venv/bin/python harness/run.py stage f-fix --arm fixer --from f-swarm --libs ... --k 8
.venv/bin/python harness/cost.py RUN; .venv/bin/python harness/audit.py RUN
.venv/bin/python harness/memorization.py p-solo f-solo c-solo
.venv/bin/python harness/analyze.py            # tables -> data/summary.json
.venv/bin/python -m pytest tests -q
```

Workers run as `general-purpose` subagents with `model: haiku`; the subagent
allowlist limits their tools.
