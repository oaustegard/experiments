`declaude_lint.py` in the `declauding` skill gets a `--skip-tables` flag. It blanks Markdown table rows before scanning, so reported line numbers stay the same.

## False positives in table cells

On 2026-11-07, a scan of 41 published posts produced 300 lint hits. Sixty-three of them were inside Markdown table cells, and the two judges labelled 58 of those 63 as legitimate prose: numbers and labels, not sentences.

`--skip-quoted` already blanks table cells, but it also blanks blockquotes and italics. `--skip-tables` skips only tables. It is for drafts that quote bad prose in italics on purpose, where the italics should still be scanned and the tables should not.

## Changes

- `declaude_lint.py`: with `--skip-tables`, any line that starts with `|` is blanked before scanning. The line is blanked rather than removed, so line numbers in the output do not change.
- `tests/test_lint.py`: two new test cases. The full suite passes (212 tests).
- The `declauding` version goes from 0.13.2 to 0.13.3, with a CHANGELOG entry.
