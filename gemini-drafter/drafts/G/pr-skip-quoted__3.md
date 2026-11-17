`declaude_lint.py` in the `declauding` skill gains a `--skip-tables` flag that blanks Markdown table rows before scanning while keeping line numbers intact.

## Reason

On 2026-11-07, a scan of 41 published posts produced 300 lint hits. Sixty-three of those 300 hits fell inside Markdown table cells, where the cells held numbers and labels rather than sentences. Two judges labelled 58 of those 63 table hits as legitimate prose.

`--skip-quoted` already blanked table cells alongside blockquotes and italics. `--skip-tables` skips tables alone, for drafts where tables should be ignored while bad prose quoted in italics should still be checked.

## Behaviour

When `--skip-tables` is passed, `declaude_lint.py` blanks any line that starts with `|` before running the checks. Blanking lines in place keeps the reported line numbers unchanged.

## Changes and tests

- Added `--skip-tables` to `declaude_lint.py`.
- Added two test cases in `tests/test_lint.py`; the full suite of 212 tests passes.
- Bumped the skill version from `0.13.2` to `0.13.3` and added a `CHANGELOG` entry.
