This PR adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. With the flag set, the linter does not scan Markdown table rows.

## Table hits in the 2026-11-07 scan

A scan of 41 published posts on 2026-11-07 produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because the cells held numbers and labels rather than sentences.

## Behaviour

- With `--skip-tables`, the linter blanks every line that starts with `|` before scanning. Blanking keeps the line count the same, so reported line numbers still match the source file.
- `--skip-quoted` already blanks table cells, and it also blanks blockquotes and italics. `--skip-tables` blanks tables and nothing else. Use it on drafts that quote bad prose in italics on purpose, where the italics should still be linted but the tables should not.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Release

- Version bumped from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
