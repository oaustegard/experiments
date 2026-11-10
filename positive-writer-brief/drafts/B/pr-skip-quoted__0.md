Adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. With the flag set, the linter skips Markdown table rows and still scans the rest of the draft.

## Why

On 2026-11-07, a scan of 41 published posts produced 300 lint hits. 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose, because table cells hold numbers and labels and not sentences.

## Behaviour

- With `--skip-tables`, every line that starts with `|` is blanked before scanning. Lines are blanked and not removed, so the reported line numbers match the source file.
- `--skip-quoted` already blanks table cells, but it also blanks blockquotes and italics. `--skip-tables` skips tables only. It is meant for drafts that quote bad prose in italics on purpose and still need their tables skipped.

## Tests

- Two new cases in `tests/test_lint.py`.
- The full suite passes (212 tests).

## Housekeeping

- Version bumped from 0.13.2 to 0.13.3.
- CHANGELOG entry added.
