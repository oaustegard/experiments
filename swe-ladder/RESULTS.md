# swe-ladder results — 2026-10-09

## Answer

On 295 SWE-bench Verified tasks (django 220, sympy 75), Haiku 5.5 resolved 256
(86.8%) on its first attempt. On the 39 it missed, a second rung that gets the
failing hidden tests' names and tracebacks resolved 30 when the same Haiku
retried and 34 when Sonnet 5.5 took over. Every task the Haiku retry solved,
Sonnet also solved. The 4-task gap is not significant (exact McNemar p = 0.125).
Sonnet's rung cost $13.53 against Haiku's $1.43 (9.4× per spawn). Over the
whole ladder, Haiku → Haiku resolves 286/295 for $8.49 and Haiku → Sonnet
resolves 290/295 for $20.59, so the 4 extra tasks cost $3.02 each.

Most of the rung-2 gain comes from the failure output. On the same 39 tasks,
without it, a fresh Haiku attempt resolved 8/39 and Sonnet with the
issue text alone resolved 17. Neither no-feedback arm solved a task that its
feedback counterpart missed. Sonnet run cold on 50 tasks Haiku had already
solved passed all 50, so the difference between the models on this set sits
in Haiku's 39 failures.

The feedback here comes from the hidden tests SWE-bench grades with. A
deployed ladder only has whatever failing signal it can produce itself, so the
retry numbers are an upper bound for that channel. The Haiku-versus-Sonnet
comparison at rung 2 is fair, because both arms got identical input.

## Findings

1. **Haiku 5.5 discriminates on real tasks where seeded bugs did not.** Rung 1
   resolved 256/295 (95% CI 82–90%): django 195/220, sympy 61/75. The
   `temporal-routing-headroom` seeded set had the weak arm at 14/14. 37 of the
   39 failures kept every PASS_TO_PASS test green and missed a FAIL_TO_PASS
   test; 2 broke a PASS_TO_PASS test; none timed out. (Round 2)
2. **Escalation beats retry by 4 tasks of 39, at 9.4× the rung cost.** Haiku
   retry 30/39, Sonnet escalation 34/39; discordant pairs 0 vs 4 (django-14034,
   django-16502, sympy-20428, sympy-21596). Mean input-side cost per spawn
   $0.0367 (30.6 turns) for Haiku, $0.3469 (16.4 turns) for Sonnet. Five
   tasks failed both arms: django-16263, sympy-16597, sympy-18199, sympy-18763,
   sympy-20438. (Round 3)
3. **The failing tests' output is worth more than the model swap.** Resolved
   counts on the 39 rung-1 failures, by arm: Haiku with no feedback
   8/39, Sonnet with no feedback 17/39, Haiku with feedback 30/39, Sonnet
   with feedback 34/39. Feedback adds 22 tasks for Haiku and 17 for Sonnet;
   the model adds 9 without feedback and 4 with it. (Round 4, Round 5)
4. **Nothing in the no-feedback arms is a subset violation.** Sonnet cold
   solved no task the Haiku retry missed (13 the other way, p < 0.001); the
   Haiku re-roll solved no task the Haiku retry missed. The feedback arms
   dominate their counterparts task by task. (`data/analysis.json`)
5. **Sonnet does not regress on Haiku's successes.** Sonnet cold resolved
   50/50 of a random sample (seed 0) of rung-1 successes. A stratified estimate
   for Sonnet alone on all 295 tasks is 273 resolved (92.5%) for about $67,
   against $7.06 for Haiku's 256. That estimate has no second rung, so it is
   not comparable with the ladder totals. (Round 4)
6. **Concurrent git worktrees of one mirror contaminate each other.** Worktrees
   share `refs/stash`; Haiku agents that ran `git stash` / `git stash pop` in
   parallel popped other tasks' edits (django-15127 received 15037's
   inspectdb changes). 30 affected tasks were redone on independent
   `git clone --shared` checkouts; one grade changed. (Round 2,
   `harness/workspace.py`)
7. **An agent's output file is not its completion signal.** Agents told to
   write the summary file last sometimes keep testing afterwards. Collecting on
   the file deleted the r2-sonnet django-16256 checkout mid-run; its saved diff
   still resolved. `ladder.next_` now only collects unasked after a 600 s
   settle, and completion notices drive collection. (Round 3, `harness/ladder.py`)

## Method

- **Tasks.** SWE-bench Verified from Hugging Face (reachable from this
  container), django and sympy: 306 tasks. A task enters the pool only if, in
  this harness, the gold patch resolves it and the empty patch does not. 11
  django tasks failed the gold check; 295 remain (`data/pool.json`).
- **Environment.** No Docker. One venv per (repo, version) from PyPI; checkouts
  go on `PYTHONPATH`. Python 3.5–3.7 tasks run on 3.8, the oldest `uv` build.
  All repo code runs inside claude-workspace's `scripts/jail.sh` (no network,
  uid nobody, empty environment). Agents reach it only through `swe-test` and
  `swe-run`; hub guards refuse anything else (`guard_jail_route`) and limit
  subagents to Bash/Read/Edit/Write/Grep/Glob (`guard_subagent_scope`).
- **Grading.** SWE-bench's procedure: fresh checkout, apply the diff, reset the
  test-patch files, apply the test patch, run its modules, parse with SWE-bench
  4.0.4's parsers plus a Django docstring-line fix. Every FAIL_TO_PASS and
  PASS_TO_PASS test must pass.
- **Agents.** Claude Code Agent-tool subagents (`general-purpose`), one checkout
  each, model `haiku` or `sonnet`, 15–16 running at once. The prompt is the
  issue text, the checkout path and the tool rules (`harness/prompts.py`).
- **Arms.**

  | run | model | input | tasks |
  |---|---|---|---|
  | `r1-final` | Haiku | issue text | 295 |
  | `r2-haiku` | Haiku | rung-1 diff applied + failing hidden test names and tracebacks | 39 rung-1 failures |
  | `r2-sonnet` | Sonnet | same as `r2-haiku` | same 39 |
  | `r1-sonnet-cold` | Sonnet | issue text | the 39 + 50 rung-1 successes |
  | `r1-haiku-reroll` | Haiku | issue text, fresh checkout | the 39 |

- **Cost.** Input-side dollars per spawn from each transcript (`harness/cost.py`):
  input, cache-write and cache-read tokens at list price. Output tokens in these
  transcripts are streaming-start values, so output cost is left out; it would
  raise both models by the same per-token ratio (20×).
- **Statistics.** Wilson 95% intervals; exact McNemar on paired arms
  (`harness/analyze.py`).
- **Limitations.** One sample per task per arm. Rung-2 feedback is the hidden
  test suite, which a real ladder does not have. Two repos only. Rung-1 cost
  counts the graded attempt; the 30 discarded pre-redo attempts cost $0.56 more.

## Log

### Round 1 — harness and validation

Built Docker-free grading for django and sympy. The stock SWE-bench parser
keyed Django tests by their docstring at verbosity 2, so 8 of the first 11
gold patches "failed" until `grade.django_docstring_lines` recorded both
keys. Validation kept 295 of 306.

### Round 2 — rung 1, Haiku

295 spawns, $7.06 input-side, mean 22.3 turns. Mid-run, django-15127's diff
contained django-15037's edits: worktrees share `refs/stash`. Checkouts moved
to independent clones; `harness/audit.py` found 30 tasks whose agents used
`git stash`, and those were redone (`r1-haiku-redo`, $0.56) and merged into
`r1-final`. One grade changed. The audit also checked for future-fix
leakage: 6 agents ran git commands that reach beyond HEAD, and the two
`git show` calls among them named ancestor commits. Subagents' attempts at
off-list tools (create_session, send_message, Monitor) were refused by the
scope guard.

### Round 3 — rung 2, retry versus escalation

Both arms started from the rung-1 diff applied plus the failing tests' names
and log excerpts. Haiku 30/39 for $1.43; Sonnet 34/39 for $13.53. On
sympy-18199 both arms reported they could not reproduce the hidden
`test_solve_modular` failure, which the rung-1 grade shows as a real
FAIL_TO_PASS miss. An early auto-collect removed r2-sonnet django-16256's
checkout while the agent was still testing; the diff saved at that moment
resolved. The settle guard in `ladder.next_` came from this.

### Round 4 — Sonnet cold

Sonnet with the issue text only, on the 39 rung-1 failures and 50 random
rung-1 successes: 17/39 and 50/50, $20.32 over 89 spawns (mean $0.2283, 11.7
turns). Proposed to separate "a stronger model" from "the failure output" at
rung 2.

### Round 5 — Haiku re-roll

Round 4 left one confound: the Haiku retry differs from Haiku's first attempt
both by having feedback and by being a second sample. A fresh Haiku attempt on
the 39 failures, issue text only, resolved 8/39 for $1.05 (mean $0.0270, 23.9 turns). The sympy-16597 agent was still working at 74 minutes and was stopped; its diff at that point was graded. The
fanout sweep inside `ladder.next_` marks tasks done when their output file
appears without saving their diffs; every diff here was collected
explicitly on its completion notice. The django-13513 agent read the
base commit's earlier fix for the same ticket as the whole fix and returned
an empty diff.
