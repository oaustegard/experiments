This PR adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill. The flag skips Markdown tables and keeps scanning blockquotes and italics.

## Reason

`--skip-quoted` already blanks table cells, but it blanks blockquotes and italics along with them. Some drafts quote bad prose in italics on purpose, so they need `--skip-quoted` off, and their tables still need skipping. `--skip-tables` skips the tables and nothing else.

On 2026-11-07, a scan of 41 published posts produced 300 lint hits, and 63 of them were inside Markdown table cells. The two judges labelled 58 of those 63 as legitimate prose: numbers and labels, not sentences.

## Behaviour

With `--skip-tables`, the linter blanks every line that starts with `|` before scanning. The lines are blanked in place, so line numbers in the report stay the same.

## Tests and release

- Two new cases in `tests/test_lint.py` cover `--skip-tables`. The full suite passes (212 tests).
- The version goes from 0.13.2 to 0.13.3, and there is a new `CHANGELOG` entry.
