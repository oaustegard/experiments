`declaude_lint.py` in the `declauding` skill gains a `--skip-tables` flag that blanks Markdown table rows before scanning while keeping reported line numbers intact.

## Why

On 2026-11-07, a scan of 41 published posts produced 300 lint hits. Sixty-three of those hits fell inside Markdown table cells, and the two judges labelled 58 of the 63 as legitimate prose consisting of numbers and labels rather than sentences.

`--skip-quoted` already blanks table cells alongside blockquotes and italics. `--skip-tables` skips tables on their own, for drafts where tables should be ignored while italicized quotes of bad prose still get scanned.

## Changes

- `declaude_lint.py`: when `--skip-tables` is passed, any line that starts with `|` is blanked before scanning, preserving line numbers in the output.
- `tests/test_lint.py`: adds 2 test cases; the full suite of 212 tests passes.
- Bumps the `declauding` skill version from 0.13.2 to 0.13.3 and adds a CHANGELOG entry.
