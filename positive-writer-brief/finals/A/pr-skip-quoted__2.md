Adds a `--skip-tables` flag to `declaude_lint.py`. With the flag set, the linter blanks every line that starts with `|` before scanning. The lines are blanked rather than removed, so reported line numbers match the source file.

## Table-cell hits in the 2026-11-07 scan

A scan of 41 published posts on 2026-11-07 produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because the cells held numbers and labels rather than sentences.

## Relation to `--skip-quoted`

`--skip-quoted` already blanks table cells, but it blanks blockquotes and italics along with them. Some drafts quote bad prose in italics on purpose and still need their tables skipped. `--skip-tables` skips tables and nothing else, so the italic specimens are still scanned.

## Changes

- `scripts/declaude_lint.py`: new `--skip-tables` flag
- `tests/test_lint.py`: two new cases for the flag
- `SKILL.md`: version 0.13.2 → 0.13.3
- `CHANGELOG.md`: entry for 0.13.3

## Testing

The full suite passes (212 tests).
