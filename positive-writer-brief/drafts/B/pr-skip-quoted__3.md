Adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. With the flag set, the linter skips Markdown table rows and still scans everything else.

## Why

On 2026-11-07 I ran the linter over 41 published posts. 63 of its 300 hits were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because table cells hold numbers and labels rather than sentences.

## Behaviour

- With `--skip-tables`, any line that starts with `|` is blanked before scanning. The line itself stays in place, so reported line numbers are unchanged.
- `--skip-quoted` already blanks table cells, along with blockquotes and italics. `--skip-tables` skips tables only. It is meant for drafts that quote bad prose in italics on purpose, where the italics should still be linted and the tables should not.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Housekeeping

- Version bump from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
