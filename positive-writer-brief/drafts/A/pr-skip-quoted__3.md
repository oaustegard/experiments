This PR adds a `--skip-tables` flag to `declaude_lint.py`. It blanks every line that starts with `|` before the scan runs, so the linter no longer reports Markdown table cells.

## Why

On 2026-11-07 I ran the linter over 41 published posts. Of the 300 hits, 63 were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose. Most table cells hold numbers and labels rather than sentences, so register rules written for sentences misfire on them.

## Behaviour

- With `--skip-tables`, a line starting with `|` is replaced by an empty line instead of being removed. Line numbers in the report stay the same as in the source file.
- `--skip-quoted` already blanked table cells, along with blockquotes and italics. The new flag skips tables and nothing else. It is meant for drafts that quote bad prose in italics on purpose, where the italics should still be scanned and the tables should not.
- Without the flag, the linter behaves as before.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Housekeeping

- `metadata.version` bumped from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
