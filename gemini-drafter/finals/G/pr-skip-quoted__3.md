Adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. The flag blanks Markdown table rows before the scan runs, so reported line numbers do not change.

## Reason

On 2026-11-07 a scan of 41 published posts produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose: the cells held numbers and labels, not sentences.

`--skip-quoted` already blanks table cells, but it blanks blockquotes and italics along with them. `--skip-tables` skips only tables. It is meant for drafts that quote bad prose in italics on purpose but whose tables should be skipped.

## Behaviour

With `--skip-tables`, `declaude_lint.py` blanks every line that starts with `|` before scanning. Each line is blanked where it stands, so the line numbers in the report match the source file.

## Changes and tests

- Added `--skip-tables` to `declaude_lint.py`.
- Added two test cases to `tests/test_lint.py`. The full suite (212 tests) passes.
- Bumped the skill version from `0.13.2` to `0.13.3` and added a `CHANGELOG` entry.
