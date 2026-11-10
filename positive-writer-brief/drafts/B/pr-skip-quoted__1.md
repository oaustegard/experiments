This PR adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. With the flag set, the linter does not scan Markdown table rows.

## Why

On 2026-11-07 we scanned 41 published posts and got 300 lint hits. Of those, 63 were inside Markdown table cells. The two judges labelled 58 of the 63 as legitimate prose: the cells held numbers and labels, not sentences.

## Behaviour

- With `--skip-tables`, the linter blanks every line that starts with `|` before scanning. Blanking keeps the line count the same, so reported line numbers still match the source file.
- `--skip-quoted` already blanks table cells, and it also blanks blockquotes and italics. `--skip-tables` blanks tables and nothing else. Use it on drafts that quote bad prose in italics on purpose, where the italics should still be linted but the tables should not.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Release

- Version bumped from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
