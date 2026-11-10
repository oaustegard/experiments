Adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill.

## Why

On 2026-11-07 I scanned 41 published posts with the linter. It reported 300 hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because the cells held numbers and labels rather than sentences.

## Behaviour

With `--skip-tables`, the linter blanks any line that starts with `|` before scanning. The blanked lines stay in place, so the reported line numbers match the source file.

`--skip-quoted` already blanked table cells, but it also blanked blockquotes and italics. The new flag skips tables and nothing else. It is meant for drafts that quote bad prose in italics on purpose, where those italics should still be scanned and the tables should not.

## Changes

- `scripts/declaude_lint.py`: new `--skip-tables` flag
- `tests/test_lint.py`: two new cases for the flag
- `metadata.version` bumped from 0.13.2 to 0.13.3
- CHANGELOG entry for 0.13.3

## Testing

The full suite passes, all 212 tests.
