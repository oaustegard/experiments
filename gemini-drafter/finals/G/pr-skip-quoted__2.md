This PR adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill, so callers can skip Markdown tables without also skipping blockquotes and italics.

## Reason

On 2026-11-07, a scan of 41 published posts produced 300 lint hits. Sixty-three of them were inside Markdown table cells, which hold numbers and labels rather than sentences. The two judges labelled 58 of those 63 as legitimate prose.

`--skip-quoted` already blanks table cells, but it also blanks blockquotes and italics. `--skip-tables` blanks table cells only. That suits drafts that quote bad prose in italics on purpose, where the italics should still be scanned and the tables should not.

## Changes

With `--skip-tables`, `declaude_lint.py` blanks every line that starts with `|` before scanning. Because the line is blanked rather than removed, line numbers for hits elsewhere in the file stay the same.

The skill version goes from 0.13.2 to 0.13.3, and there is a CHANGELOG entry.

## Tests

There are two new cases for `--skip-tables` in `tests/test_lint.py`. The full suite of 212 tests passes.
