`declaude_lint.py` in the `declauding` skill adds a `--skip-tables` flag that excludes Markdown tables from lint scans without suppressing blockquotes or italics.

## Reason

Today, `--skip-quoted` blanks table cells alongside blockquotes and italics. On 2026-11-07, a scan of 41 published posts produced 300 lint hits, and 63 of those hits fell inside Markdown table cells. Two judges labelled 58 of those 63 table hits as legitimate numbers and labels rather than prose sentences. Drafts that intentionally quote bad prose in italics still need italic text checked while skipping tables.

## Behaviour

When the caller passes `--skip-tables`, `declaude_lint.py` blanks any line that starts with `|` before scanning. Blanking the line in place preserves the reported line numbers for the rest of the file.

## Tests and release

`tests/test_lint.py` adds two test cases for `--skip-tables`, and the full suite of 212 tests passes. The `declauding` version bumps from 0.13.2 to 0.13.3 with an entry in `CHANGELOG`.
