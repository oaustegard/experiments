# bulk-worker: with vs without the examples (2026-11-26)

Question: does the `<examples>` block in `.claude/agents/bulk-worker.md` (claude-workspace 9030932) change what a Haiku worker produces?

## Setup

- Fixture: `fixture/scripts/fx.py`, six functions. `median` has a bug (even count returns the upper middle value). `days_in_month` leaps on `year % 4` only (1900 comes out as 29). `parse_kv` has no stated behaviour for input without `=`.
- Task: write tests/test_fx.py for all six, check with `pytest -k <function>`, report in the agent's line format.
- Arms: A = full instructions (7 examples), B = same text with the `<examples>` block removed. A1/A2/B1/B2, each a fresh `general-purpose` subagent with `model: haiku`, instructions read from INSTRUCTIONS.md in its own directory, effort left at the default.
- Judge (`judge.py`, deterministic): implementation unchanged and nothing written outside tests/; per-function pass/fail against the buggy code and against `ref_fx.py` (bug fixed); mutation kill rate over 19 hand-written mutants of the reference; claimed counts against a rerun.
- Not covered: the agent file was not loaded as a real agent type (definitions load at session start), n = 2 per arm, and the six functions resemble the examples closely.

## Deterministic results (same in all four arms unless noted)

| | A1 | A2 | B1 | B2 |
|---|---|---|---|---|
| Implementation untouched, no extra files | yes | yes | yes | yes |
| Tests passing against the fixed reference | all | all | all | all |
| Tests with `is not None` or mirroring the code | 0 | 0 | 0 | 0 |
| Test of `parse_kv` without `=` (unspecified) | none | none | none | none |
| Mutants killed | 18/19 | 18/19 | 18/19 | 18/19 |
| Survivor | `month in (4,6,9)` (tested only months 4, 9) | century-rule mutant | century-rule mutant | century-rule mutant |
| `median` flagged BUG-SUSPECT with path:line | yes (20) | yes (20) | yes (20) | yes (20) |
| `days_in_month` 1900 bug found | yes | no | no | no |
| Per-function counts match the rerun | yes | yes | yes | yes |
| Test total in the trailer line | 19 (actual 18) | 19 (actual 21) | "19 passing" (actual 21) | 22 (correct) |
| Subagent tokens / seconds | 75,724 / 23 | 76,589 / 28 | 75,409 / 33 | 74,011 / 27 |

## Format fidelity (read by hand)

- A1 added a markdown table header and separator row. A2 matched the specified line format exactly.
- B1 matched the format but put its UNSPECIFIED notes in a trailing sentence, not in the per-item notes column. B2 shortened item names to the bare function name.
- All four appended a trailing sentence after TOTAL although the brief said to reply with the report only.

## Reading

- On this batch the examples did not change test quality: same kill rate, no mirrored assertions, same handling of the underspecified `parse_kv` case, same `median` bug report.
- They changed report habits. The A arms put UNSPECIFIED notes in the item line as designed (including extras: multiple spaces, unhashable items, invalid month). The B arms mostly moved them to the trailer.
- The one real difference in findings: A1 asserted 1900 is not a leap year, flagged the code as BUG-SUSPECT, and said its expectation rested on reading "accounting for leap years" as the Gregorian rule. The other three treated the century rule as UNSPECIFIED under rule 5 and shipped tests that pass against the buggy code. A2 had the examples and still declined, so this is variance, not an example effect.
- Rule 10 (report only counts CHECK printed) was broken in the trailer by three of four runs, with and without examples.
- Cost of the examples: about 1.5k tokens of input, about 2% of a run that already carries about 75k tokens.
