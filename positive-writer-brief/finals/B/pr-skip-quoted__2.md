Adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. With the flag set, the linter leaves Markdown tables out of the scan.

## Table hits in the 2026-11-07 scan

A lint run over 41 published posts on 2026-11-07 produced 300 hits. Of those, 63 were inside Markdown table cells. The two judges labelled 58 of the 63 as legitimate prose: the cells held numbers and labels rather than sentences.

## Behaviour

- With `--skip-tables`, the linter blanks every line that starts with `|` before it scans. A blanked line stays in place, so the line numbers it reports match the source file.
- `--skip-quoted` already blanks table cells, along with blockquotes and italics. `--skip-tables` blanks tables and nothing else. It is meant for drafts that quote bad prose in italics on purpose, where the italics should still be linted but the tables should not.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Version and changelog

- Version bumped from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
