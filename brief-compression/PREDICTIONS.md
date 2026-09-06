# Pre-registered predictions — brief-compression

Written 2026-09-06 before any subagent was dispatched.

Design: one task (count ERROR-not-DEBUG lines, write a 3-key JSON, append one
log line, touch nothing else, reply DONE), 20 replicate inputs, three brief
styles carrying identical information, two receiver models (Haiku 4.5 and
Sonnet 5 as Claude Code `general-purpose` subagents). 120 runs. Nine binary
checks per run; success = 9/9.

Brief cost (o200k_base as proxy; Claude's tokenizer is not available in-session):
prose 184, structured 119, telegraphic 99 tokens. Stripping spaces was shown
earlier today to save nothing on o200k (53 vs 53 tokens on the screenshot text),
so the telegraphic arm keeps spaces between words and fuses only where the
screenshots did (e.g. "exactly3keys", "Append1line", "Inputreadonly").

P1. Sonnet: no style difference detectable at n=20 (all three cells >= 18/20).
P2. Haiku: telegraphic loses to prose by >= 3/20 on success, and the failures
    concentrate in `reply_done` and `keys_exact` (the format constraints that
    telegraphese compresses hardest), not in `count_ok`.
P3. Haiku: structured ties prose (within 2/20) at ~35% fewer brief tokens, i.e.
    the token saving comes from structure, not from telegraphese.
P4. No arm modifies the input file (input_unmodified 20/20 in every cell).

Kill criterion for the recommendation "write terse structured briefs, not
telegraphic ones": P3 fails (structured loses to prose by >= 3/20 on Haiku).

## Amendment 1 — Task B added after Task A hit ceiling (2026-09-06, after 22 Task A runs)

Task A scored 22/22 at 9/9 before any cell had 10 runs, so it cannot separate
the styles. Task A is capped at 6 runs per cell (28 total; the Task A
predictions P1–P4 stand as written and are reported against that n). Task B
is a denser brief: a CSV filter with three conjunctive conditions (one of them
a case-sensitive prefix with a decoy `Svc_ops` user and a decoy lowercase
`get` action), a per-user aggregate with a tie-break rule, a TSV with an exact
header, a 3-key JSON, a log line carrying a computed number, and a reply that
must carry the same number. 13 checks. n = 10 per cell, 60 runs.

Brief cost, o200k proxy: prose 287, structured 218, telegraphic 178.

P5. Sonnet: no style difference detectable on Task B either (every cell >= 8/10).
P6. Haiku: telegraphic loses to prose by >= 3/10 on Task B success; failures
    concentrate in the rules telegraphese abbreviated hardest — `report_rows_ok`
    (the `Svc_ops` / lowercase-`get` decoys under "casesens", "lc") and
    `reply_ok` ("Reply exactly \"OK N\" only").
P7. Haiku: structured within 1/10 of prose on Task B.
P8. Telegraphic runs use more tool calls per run than prose in the same cell
    (re-reading the brief or the data to resolve abbreviations), on both models.
    This was suggested by Task A (haiku telegraphic mean tool_uses > haiku prose)
    and is written down here before Task B is dispatched.
