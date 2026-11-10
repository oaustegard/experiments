Adds a `--skip-tables` flag to `declaude_lint.py`. With the flag set, the linter blanks every line that starts with `|` before it scans, so the line numbers it reports stay the same.

## Table-cell hits

A scan of 41 published posts on 2026-11-07 produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because the cells held numbers and labels rather than sentences.

## Relation to `--skip-quoted`

`--skip-quoted` already blanks table cells, but it also blanks blockquotes and italics. Some drafts quote bad prose in italics on purpose and should still have those italics scanned, while their tables should be skipped. `--skip-tables` skips the tables and nothing else.

## Changes

- `scripts/declaude_lint.py`: new `--skip-tables` flag
- `tests/test_lint.py`: two new cases. The full suite passes (212 tests).
- `metadata.version` bumped from 0.13.2 to 0.13.3
- CHANGELOG entry
