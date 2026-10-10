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

## Follow-up — six `paired` tasks (unrun)

`expr_eval`, `cron_next`, `wrap_text`, `lru_ttl`, `stack_vm`, and `text_table` were
re-seeded so each carries one wrong assumption written into two call sites. Fixing either
site alone leaves the hidden suite red, and fixing the first turns the visible suite
green. The remaining eight stay as they are: the pilot measured them at 14/14, so they are a
control anchor rather than a guess about what "easy" means.

Three of the six first pairings did not couple, and the build rejected each:

- `cron_next`: no hidden test exercises a wildcard field at its maximum value, so
  seeding the `*` branch's inclusive bound changed no test outcome. Re-seeded onto the
  restricted-flag contract: util decides which fields are restricted, core decides what
  to do when both day-of-month and day-of-week are.
- `expr_eval`: the tokenizer/parser pair interlocked completely. The tokenizer emitted a
  token no parser branch matched, so repairing either half alone changed nothing an agent
  could observe — harder in one sense, but it produces no stop-early trap. Re-seeded onto a separable
  pair inside the same `**` handling.
- `wrap_text`: at `width=1` the chunker pairing made no progress and hung pytest instead
  of failing it. `run_pytest` now reports a timeout as a build error naming the repo. Re-seeded onto the paragraph
  splitter and its joiner.

Verified per task, live: after repairing one site, the visible suite is green and the
hidden suite is still red on between 1 and 8 tests, per the table in the README.

Nobody has run an arm against these six yet, so whether they make the weak arm fail is
an open question. The pilot's lesson was that a set built to be hard is not the same as
a set measured to be hard.

## Stage-1 pilot on the paired task set (2026-09-03)

Second replicate on the re-seeded set. Workflow run `wf_e8f72a2c-ec2`, 28 agents, 577 s.
Not pooled with the run above: that one measured a different set of tasks.

| arm | model | effort | solved | output tokens | $/task | $/completed |
|---|---|---|---|---|---|---|
| weak | Sonnet 5 | low | 9/14 | 16,779 | $0.0180 | $0.0280 |
| strong | Opus 5 | high | 10/14 | 55,674 | $0.0994 | $0.1392 |

Oracle: **$0.0659** per completed task, **0.473×** all-strong. The ceiling is gone and the
headroom is real.

| set | tasks |
|---|---|
| both solve | csv_line, expr_eval, glob_match, parse_range, roman_strict, semver_cmp, template_render, text_table, toposort_lex |
| strong only | stack_vm |
| weak only | — |
| neither | cron_next, interval_merge, lru_ttl, wrap_text |

## Which failures were the trap

Seven of the nine failures are **exactly** the residue the build recorded — the run
repaired one site, watched the visible suite go green, and stopped.

| task | arm | hidden tests left red |
|---|---|---|
| cron_next | weak, strong | test_dom_and_dow_or_rule, test_dow_step_counts_as_restricted |
| lru_ttl | weak, strong | test_len_counts_only_live |
| wrap_text | weak, strong | test_paragraph_with_whitespace_only_line |
| stack_vm | weak | test_stack_manip |

Opus at effort high falls for the same trap as Sonnet at effort low on three of the four.
`expr_eval` is the one both arms escaped: the weak run found and fixed both sites, naming
the parser's right-associativity as a second cause after the tokenizer.

The two `interval_merge` failures are not the trap. It is a shallow control task, and each
arm broke a different edge case while rewriting the merge loop — `test_duplicate_points_collapse`
for weak, `test_point_on_end_touching` for strong. Weak solved it in the earlier pilot. A control task
flipping between replicates is the run-to-run variance the protocol warns about.

## Headroom versus discrimination

The 0.473× headroom comes almost entirely from the nine tasks **both** arms solve: route
those cheap and the saving follows. Escalating pays on exactly one task
(`stack_vm`); on four more no tier succeeds, so escalating there buys nothing.

Discrimination — tasks where the right tier differs — is therefore 1 of 14, and one of
those flips between replicates. A cascade that runs cheap and escalates on a **verified**
failure captures most of this headroom without conditioning on anything. That is
`agent-routing`'s existing design, and this set can measure it.

SWE-Router's claim is about discrimination: predicting, from a partial trajectory, which
tasks need the expensive tier. Measuring that needs a set where the tiers disagree far
more often than once in fourteen. Whether a trap can be tuned to catch the weak arm while
the strong arm escapes is a question about task design, and nobody has answered it.

`|W \ S|` is empty this replicate, so nothing here reproduces SWE-Router's above-all-strong
synergy. The earlier pilot's single instance did not survive the change of task set.

## Sonnet-to-Opus ladder plus a capability probe (2026-09-03)

Workflow `wf_6b7c41cd-2bd`, 11 agents. Rung 1 is the already-measured r2 weak arm, so only
the escalation step is new: Opus at effort high on the five tasks Sonnet failed, carrying
the worker's patch and the held-out failure output.

| arm | solved | total $ | $/completed | vs all-strong |
|---|---|---|---|---|
| all-weak, Sonnet 5 @ low | 9/14 | 0.2517 | 0.0280 | 0.18× |
| all-strong, Opus 5 @ high | 10/14 | 1.3919 | 0.1392 | 1.00× |
| oracle over the two solo arms | 10/14 | 0.6590 | 0.0659 | 0.47× |
| **ladder, Sonnet → Opus** | **13/14** | **1.0643** | **0.0819** | **0.76×** |

Opus ran on 5 of 14 tasks and rescued 4: `cron_next`, `lru_ttl`, `stack_vm`, `wrap_text`.

The ladder solves three more tasks than always-Opus at three quarters of the cost. It also
beats the oracle, which means the oracle was misnamed: it bounds *routing between solo
arms*, not pipelines. A cascade can exceed it because rung 2 is not the same as a cold
Opus run.

That gap is the whole result. Opus starting from the issue text fell into the paired trap
on `cron_next`, `lru_ttl` and `wrap_text` — the same three that caught Sonnet. Opus
starting from Sonnet's patch plus the failing assertions fixed all three. The failure
signal was worth more than the tier.

`interval_merge` failed at every tier: weak, strong, and the escalation. It is a shallow control whose
reference carries a baroque touching-and-containment condition, and every run that
rewrote the merge loop broke a different case. That reads as a problem with the task rather than
evidence about the tiers.

## The escalation decision

The r2 arms reported success on 28 of 28 runs while passing 19. Every failing run said it
was done. "Try it, and ask for help if you fail" cannot work on that signal, because the
worker's verifier is the visible suite and the visible suite is satisfiable while the task
is unfinished.

The ladder works because the escalation decision is made by whoever holds the held-out
suite. Same cascade, same rungs; the difference is who decides.

## Capability probe results

Three shapes, Sonnet 5 @ low against Opus 5 @ high, one run each.

| shape | built from | what it targets | sonnet | opus |
|---|---|---|---|---|
| three_sites | text_table, 3 coupled sites | thoroughness past two sites | pass 15/15 | pass 15/15 |
| ambiguous | interval_merge, flag branch removed | inferring intent the tests underdetermine | pass 17/17 | pass 17/17 |
| no_tests | cron_next, no visible suite at all | correctness with no test feedback | pass 31/31 | pass 31/31 |

Six for six. Sonnet at effort low solved a two-site coupled bug with **no test suite in the
repository**. It had a symptom report and the source, and nothing else.

That kills the mechanism I had assumed. The paired trap does not catch runs because the
visible suite is a crutch; remove the crutch entirely and Sonnet still reasons to a correct
fix. What the trap catches is a run that has a green signal and takes it.

At one run per cell this rules nothing out, but three shapes aimed at three different
hypothesised gaps all missed. Combined with Opus falling for the same traps as Sonnet in
r2, the evidence says seeded-bug repair in a small module does not separate these tiers at
all, and that a set built to discriminate them has to leave this task family.

## The cascade agent-routing prescribes (2026-09-03)

Workflow `wf_335d96cd-611`, 19 agents. The skill's rung 2 is the **same model at higher
effort**, not a tier jump, and both rungs carry the concision lever. Two phases: rung 2
run from the identical starting state the Opus rung received, so only the model and effort
differ, and a concision-enabled rung 1 to price the lever.

### Rung 2 from an identical starting state

| rung 2 | output tokens | cost | rescued |
|---|---|---|---|
| Opus 5 @ high | 32,504 | $0.8126 | 4/5 — cron_next, lru_ttl, stack_vm, wrap_text |
| Sonnet 5 @ medium + concision | 11,691 | $0.1754 | 4/5 — the same four |

Identical outcome, down to which task each rescued and which one (`interval_merge`)
neither could. Sonnet's rung is **4.6× cheaper**.

### Full ladders over the same rung 1

| arm | solved | total $ | $/completed | vs all-strong |
|---|---|---|---|---|
| all-weak, rung 1 only | 9/14 | 0.2517 | 0.0280 | 0.18× |
| all-strong, Opus 5 @ high | 10/14 | 1.3919 | 0.1392 | 1.00× |
| ladder → Opus 5 @ high | 13/14 | 1.0643 | 0.0819 | 0.76× |
| **agent-routing → Sonnet 5 @ medium** | **13/14** | **0.4270** | **0.0328** | **0.31×** |

The skill's own prescription beats the tier jump I reached for by 2.5×, at identical
quality. Against always-Opus it solves three more tasks for 31% of the money.

Two things the skill said and this arm confirms. "Rungs can be the same model at different
effort — often better than a tier jump, because it keeps rung 1 genuinely cheap." And the
escalation's value is the failure evidence, not the tier: the same assertions that let Opus
find the second site let Sonnet find it too, at one effort step up.

The gap is likely wider than 2.5× on a real bill. These are output-token costs; caches are
model-scoped with no escape hatch, so the Opus rung discards the Sonnet prefix entirely
while the same-model rung keeps at least the tools and system tiers. An `effort` change
does invalidate the messages cache on every model, and Sonnet 5 has no per-message effort
escape hatch, so the same-model rung does not keep everything.

### The concision lever did not transfer

| rung 1 | output tokens | solved |
|---|---|---|
| no concision | 16,779 | 9/14 |
| + concision | 16,294 | 9/14 |

**2.9%**, against the 37% the skill measured on Sonnet. No quality change either way.

The skill scopes the lever to "every long-output generation spawn", and it measured that
37% on spec-dense module generation. A bug fix emits a small patch and a paragraph, so
there is little deliberation to cut. The lever is not wrong; it does not reach agentic
repair work, and the skill's wording invites applying it there.

The two rung-1 runs also make a variance point. Same model, same effort, one instruction
different: both scored 9/14 and their failure sets differ by one task each way —
`interval_merge` passed here and failed before, `expr_eval` the reverse. Four tasks failed
in both. That is the disjoint-failure-set effect the protocol warns about, showing up
inside a single-variable comparison.

## Handing rung 2 the first rung's reasoning (2026-09-03)

Workflow `wf_40b0f233-bd7`, 30 agents. Two arms, three replicates, five tasks. Both arms
are `sonnet@medium` + concision from the identical starting state; the only difference is
one inserted block carrying rung 1's stated cause, framed as a claim to check rather than
fact. Both arms were re-run rather than comparing against the earlier n=1 figure.

This is the variable `agent-routing` and SWE-Router disagree about. The skill says carry
the prior attempt and the raw failure output. SWE-Router restarts the strong model from
the task description, on an uncited claim that conditioning it on the weak model's
reasoning biases it toward the weak model's mistakes.

| task | with reasoning (a/b/c) | without (a/b/c) |
|---|---|---|
| cron_next | pass pass pass | pass pass pass |
| lru_ttl | pass pass pass | pass pass pass |
| stack_vm | pass pass pass | pass pass pass |
| wrap_text | pass pass pass | pass pass pass |
| interval_merge | fail **pass** fail | fail fail fail |

**13/15 with, 12/15 without.** Output tokens 24,252 against 24,504, a 1.0% difference in
the with-reasoning arm's favour — noise in both directions.

The whole difference is one `interval_merge` replicate, on the one task that has flipped
in every comparison this experiment has run. Four of five tasks pass in all six cells.

So neither claim survives. The reasoning does not help, and it does not anchor. Whatever
the informed retry is worth, the patch and the failing assertions already carry it; the
narrative on top adds nothing measurable. `agent-routing`'s wording is right as written: it asks
for the prior attempt and the raw failure *output*. The model's account of itself was
never part of the prescription, and adding it changes nothing.

Two smaller readings. Every with-reasoning run classified the prior diagnosis as
`correct-but-incomplete`, which is accurate: rung 1 had fixed its site correctly and
stopped. Being told a plausible, correct-as-far-as-it-goes diagnosis did not stop any run
from finding the second site.

And the self-report failure repeats. All 30 runs reported `confident_complete: true`; five
of them were still failing. Every one is `interval_merge`.

`interval_merge` has now been attempted 8 times across models, efforts and contexts, and
passed twice. Its reference carries a touching-and-containment condition that every run
rewrites and most get wrong. Treat its result as a property of the fixture.

## Sonnet 5.5 and Opus 5.5 on the paired set (2026-09-29)

Two replicates, four arms, 14 tasks, 112 subagent runs. Workflow runs `wf_7efa81de-e85`
(r1, arms concurrent) and `wf_e073fa41-52d` (r2, arms in sequence so `budget.spent()`
could be read per arm). Prompts are byte-identical to `emit_prompts.py` apart from the
run root. Data: `data/results_s55_r1.json`, `data/results_s55_r2.json`,
`data/tokens_s55.json`. No run edited its tests or wrote outside its package.

| arm | model | effort | r1 | r2 | r2 output tokens | $/task | $/completed (r2) |
|---|---|---|---|---|---|---|---|
| s55low | Sonnet 5.5 | low | 10/14 | 11/14 | 11,466 | $0.0082 | $0.0104 |
| s55med | Sonnet 5.5 | medium | 11/14 | 12/14 | 10,853 | $0.0078 | $0.0090 |
| s55high | Sonnet 5.5 | high | 12/14 | 12/14 | 14,488 | $0.0103 | $0.0121 |
| o55high | Opus 5.5 | high | 11/14 | 13/14 | 18,464 | $0.0264 | $0.0284 |

Prices $2/$10 (Sonnet 5.5) and $4/$20 (Opus 5.5) per MTok, output only. Output tokens are
from r2 alone: r1 ran its arms concurrently, and the workflow transcripts record only the
streaming-start usage of each message, so r1's per-arm output is not recoverable.

For comparison on the same 14 tasks (r2 of the earlier pilot): Sonnet 5 at `low` solved
9/14 with 16,779 output tokens, Opus 5 at `high` 10/14 with 55,674.

Tasks that any arm missed, solves out of two replicates:

| task | s55low | s55med | s55high | o55high |
|---|---|---|---|---|
| cron_next | 0 | 0 | 0 | 1 |
| interval_merge | 2 | 1 | 1 | 1 |
| lru_ttl | 0 | 1 | 2 | 1 |
| stack_vm | 1 | 2 | 2 | 2 |
| wrap_text | 0 | 1 | 1 | 1 |

**`cron_next` measures a convention.** Seven of eight runs repaired both seeded sites and
then failed the same hidden test, `test_dow_step_counts_as_restricted`. Each of those runs
wrote in its summary that a `*/n` field counts as unrestricted, citing Vixie cron. The
hidden suite expects the opposite. The failure records which convention the reference
chose, not whether the run found the bug. Without it the arms solve 21, 23, 24 and 23 of 26.

What this changes for `agent-routing`:

- **`low` does not switch Sonnet 5.5's thinking off.** Sonnet 5 at `low` solved 9/14.
  Sonnet 5.5 at `low` solved 10/14 and 11/14 and missed only paired-trap tasks. `medium`
  used slightly fewer output tokens than `low` in r2 (10,853 against 11,466), a gap
  inside the 23% run-to-run variance this repo has measured. On this task family the two
  levels cost the same.
- **Sonnet 5.5 at `high` matches Opus 5.5 at `high`.** Each solved 24 of 28 across the
  two replicates. Sonnet cost $0.0121 per completed task against Opus's $0.0284, 0.43×.
- **Opus 5.5 at `high` emits a third of Opus 5's output on the same tasks** (18,464
  against 55,674), and costs $0.0264 a task against $0.0994.
- **Effort buys the paired traps on Sonnet 5.5.** `lru_ttl` and `wrap_text` went from 0
  of 2 at `low` to 2 of 2 and 1 of 2 at `high`. The earlier finding that Opus 5 at `high`
  fell for the same traps as Sonnet 5 at `low` holds on 5.5 at the tier level: Opus 5.5
  missed `lru_ttl` and `wrap_text` once each.

Limits: two replicates at n=14; a one-task pass difference between arms is noise. Seeded
repair in small modules does not separate tiers, as the 2026-09-03 probe already showed,
so the Sonnet/Opus result here says the two are close on this family and nothing about
harder work. Thinking share could not be measured: Claude Code omits thinking text and
the transcripts carry no final usage.

## Haiku 5.5 as rung 1, and the same-model informed retry (2026-10-07)

One replicate, 14 tasks, then rung 2 on the misses two ways. 20 Agent-tool subagents
dispatched by session 7c199534 (CCotw); prompts byte-identical to `emit_prompts.py`
apart from the run root and a trailing `[no-context]`, which keeps the
delegating-with-context hook from appending the parent's transcript. The Agent tool
cannot set effort, so every run used the session's level. Every transcript names
`claude-haiku-5-5` or `claude-sonnet-5-5`. No run edited its tests. Data:
`data/results_h55_r1.json`, `data/results_h55_r1_rung2_h55.json`,
`data/results_h55_r1_rung2_s55.json`, `data/cost_h55_ladder.json`.

| arm | solved | input-side $ | $ per spawn |
|---|---|---|---|
| Haiku 5.5, rung 1 | 11/14 | 0.1082 | 0.0077 |
| rung 2: Haiku 5.5 + patch + hidden failures | 3/3 | 0.0145 | 0.0048 |
| rung 2: Sonnet 5.5 + patch + hidden failures | 3/3 | 0.5111 | 0.1704 |
| **ladder Haiku → Haiku** | **14/14** | **0.1227** | — |
| ladder Haiku → Sonnet | 14/14 | 0.6193 | — |

Rung 1 missed `cron_next`, `lru_ttl` and `wrap_text`: three of the six paired traps, the
set Sonnet 5.5 at `low` also misses (it solved 10 and 11 of 14 on 2026-09-29). All three
Haiku runs named the second defect in their own summaries ("`day_ok` requires both to
match... POSIX uses OR", "`__len__` still counts expired entries", "a whitespace-only line
does not end a paragraph") and left it, because it was "outside the reported bug". They
found the trap's second site and declined to fix it. Handed the
patch and the hidden suite's failure output, Haiku fixed all three, as Sonnet did. On
`cron_next` both chose the convention the reference uses (`*/7` counts as restricted),
which seven of eight Sonnet/Opus runs on 2026-09-29 got wrong from the issue text alone.

**The cost is the prefix, not the output.** Every earlier table in this file prices
output tokens only. An Agent-tool subagent starts from a ~55K-token Claude Code prefix
(system prompt, tools, skills list) that it writes to the 5-minute cache on its first
request, and these runs show that prefix dominating the bill: mean cache write 45K tokens
per Haiku rung-1 run and 59K per Sonnet rung-2 run (parallel launches share nothing).
Priced at the 5-minute write rate (1.25× input) and the read rate (0.1×):

- one Haiku 5.5 spawn: $0.005–0.011
- one Sonnet 5.5 spawn: $0.17, which is 15× the $0.0121 output-only figure the 2026-09-29
  `s55high` arm reported per completed task

By the same input-side measure, the Haiku → Haiku ladder solved 14/14 for less than one
Sonnet 5.5 spawn. Output does not change the order. These transcripts carry
streaming-start usage only (`out_floor` in the data is a floor), but at $0.50/MTok a Haiku
run would need ~340K output tokens to cost what a Sonnet spawn's prefix does.

What this changes for `agent-routing`:

- **The cascade precondition now holds for Haiku.** At Haiku 4.5's $1/$5, its verbosity
  made `haiku → sonnet` worse than Sonnet alone. Haiku 5.5 is $0.10/$0.50, 20× under
  Sonnet 5.5 on every token class, and on this battery it matched Sonnet 5.5 `low`'s pass
  set.
- **Rung 2 can stay on Haiku.** The informed retry, not the tier, carried the rescue, as
  the 2026-09-03 Sonnet ladder found. Escalating to Sonnet bought nothing here and cost
  35× per spawn.
- **Price subagents per spawn.** In Claude Code the fixed prefix outweighs the work on
  short repair tasks, so a tier's per-spawn input cost is the first number to compare.

Limits: one replicate, n=14; three-task rung 2. The pass sets agree with the earlier
Sonnet runs, but a one-task difference is noise here. Effort was not controlled. Haiku
5.5 accepts `effort` on the API (default `medium`), and whether the Workflow tool's
`agent({effort})` reaches it is unmeasured. The input-side cost assumes cold caches.
Sequential spawns of one model share a cached prefix (32 of 36 probes did on 2026-09-28),
which cuts a Sonnet spawn to ~$0.011 and a Haiku spawn to ~$0.0006, and leaves the ratio
unchanged.
