Adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. With the flag set, the linter skips Markdown table rows and still scans everything else.

## Table hits in published posts

A scan of 41 published posts on 2026-11-07 produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose: numbers and labels, not sentences.

## Behaviour

- With `--skip-tables`, any line that starts with `|` is blanked before scanning. The line stays in place, so reported line numbers are unchanged.
- `--skip-quoted` already blanks table cells, along with blockquotes and italics. `--skip-tables` skips tables only. It is for drafts that quote bad prose in italics on purpose, where the italics should still be linted and the tables should not.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Housekeeping

- Version bump from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
