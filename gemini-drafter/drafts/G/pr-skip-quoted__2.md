This PR adds a `--skip-tables` flag to `declaude_lint.py` in the `declauding` skill so callers can ignore Markdown tables without ignoring blockquotes or italics.

## Reason

On 2026-11-07, a scan of 41 published posts produced 300 lint hits. Sixty-three of those 300 hits fell inside Markdown table cells, where numbers and labels sit instead of full sentences. Two judges labelled 58 of those 63 table hits as legitimate prose.

`--skip-quoted` already blanks table cells alongside blockquotes and italics. `--skip-tables` blanks table cells alone, which covers drafts where tables should be skipped while quoted bad prose in italics is scanned.

## Changes

When `--skip-tables` is passed, `declaude_lint.py` blanks any line that starts with `|` before scanning. Blanking the line instead of removing it preserves the reported line numbers for hits elsewhere in the file.

The `declauding` skill bumps from 0.13.2 to 0.13.3 with a CHANGELOG entry.

## Tests

`tests/test_lint.py` adds two test cases for `--skip-tables`. The full suite passes at 212 tests.
