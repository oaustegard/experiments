# Stage-1 pilot — 2026-09-03

One replicate, two arms, 14 tasks, 28 subagent runs. Workflow run `wf_b3cd3994-4b0`.

**The task set does not discriminate.** The weak arm solved 14/14 and the strong arm
13/14, so the oracle router degenerates to "send everything to the cheap tier" and there
is no routing decision left to condition on. The 3-replicate run does not follow from
this; the bugs need to be harder first.

## Numbers

| arm | model | effort | solved | output tokens | $/task | $/completed |
|---|---|---|---|---|---|---|
| weak | Sonnet 5 | low | 14/14 | 20,108 | $0.0215 | $0.0215 |
| strong | Opus 5 | high | 13/14 | 52,777 | $0.0942 | $0.1015 |

Oracle: $0.0215 per completed task, **0.229×** all-strong — but only because the weak arm
never failed, so the oracle is the all-weak arm under another name.

`|W \ S| = {interval_merge}`: one task the weak arm solved and the strong arm did not
(16/17 hidden tests). At one replicate this is a single event, not a measured effect. SWE-Router's above-all-strong curve needs
this shape; `agent-routing` assumes escalation always helps, which requires the set to be
empty. Worth re-checking on a harder set before reading anything into it.

`budget.spent()` is turn-cumulative, so the marks give per-arm totals. Per-task figures
are the arm mean; `data/marks_stage1.json` records this rather than leaving it implied.

## Bug difficulty

Every bug is a single verified find/replace in a module under 110 lines, and both tiers
named the actual cause on every task. The three `contract` tasks were built so a locally-plausible
patch would pass the visible suite and leave hidden tests red. No run took that bait; all
six found the real cause.

`orchestrated-coding-pareto` hit this same wall in its round 0, where the orchestration
arms went vacuous because the cheap tier had zero failures to seed them. Converting those
tasks into repos changed what the agent has to do to find the bug; it did not change how
hard the bug is once found.

For Stage 2 the set needs failures at the cheap tier. Candidates, in rough order of cost:
bugs spanning two call sites that must change together; a repo large enough that
localisation is the work; real SWE-bench Verified instances, which is what the paper used
and what removes the authored-difficulty objection along with it.

## Two harness defects the pilot found

Neither would have surfaced without running agents against the set.

**The visible suite for `roman_strict` and `toposort_lex` died on NameError.**
`slice_tests` kept top-level constants and classes but dropped non-test helper functions
that the hidden suites define for their own use, namely `_to_roman` and `_check_cycle`. The
build-time invariant asserted only that the visible suite was red under the bug, and a
suite that cannot resolve a name is also red. Three of the four runs on those tasks
answered by injecting the missing name through the repo's root `conftest.py`, so their
trajectories are about a harness defect, not the seeded bug. The hidden-suite grades
survive — grading rebuilds from pristine plus the package — but the token counts for
those runs include the detour.

Fixed: the slicer keeps every top-level statement except tests it is not exposing. Two
new invariants cover it. The visible slice must be **green on the unbugged reference**,
and it must fail rather than error under the bug.

**The grader's scope was too narrow.** It restored `tests/` and compared against it,
which caught nothing when a run wrote to `conftest.py` instead. It now rebuilds from the
pristine repo and overlays only the package directory, and reports every file changed
outside it. The tamper check also mis-fired on `__pycache__` left by a local pytest run,
marking all 28 runs tampered on the first grading pass.

## Reproducing

```bash
python3 harness/build_tasks.py --check     # tasks/ matches seeds/
python3 -m pytest tests/ -q                # 107 invariants
python3 harness/oracle.py                  # rebuild analysis_stage1.json from data/
```

`data/results_stage1_r1.json`, `data/tokens_stage1_r1.json` and `data/marks_stage1.json`
are the pilot as graded, kept for the record. They were produced before the slicer fix,
so `roman_strict` and `toposort_lex` in them carry the defect described above.
