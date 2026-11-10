This PR adds a `--skip-tables` flag to `declaude_lint.py`. With the flag set, the linter blanks every line that starts with `|` before the scan runs, so it reports no hits from Markdown table cells.

## Table-cell hits in 41 posts

A scan of 41 published posts on 2026-11-07 produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because the cells held numbers and labels rather than sentences.

## Behaviour

- With `--skip-tables`, a line starting with `|` is replaced by an empty line instead of being removed, so line numbers in the report match the source file.
- `--skip-quoted` already blanked table cells, along with blockquotes and italics. The new flag skips tables and nothing else. It is meant for drafts that quote bad prose in italics on purpose, where the italics should still be scanned and the tables should not.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Version and changelog

- `metadata.version` bumped from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
