# Task: wrap_text

Write a Python module defining exactly one public function:

```python
def wrap_text(text: str, width: int) -> str
```

Greedy paragraph-aware text wrapping with exact output rules. Do **not** use the
`textwrap` module.

Rules:
- `width < 1` raises `ValueError`.
- Paragraphs: the input is split into paragraphs by **blank lines** (lines that are
  empty or whitespace-only). One or more consecutive blank lines is a single
  paragraph break. In the output, paragraphs are separated by exactly one empty
  line (i.e. `"\n\n"` between the last line of one paragraph and the first of the
  next).
- Within a paragraph, all whitespace (spaces, tabs, single newlines) collapses:
  the paragraph is a sequence of words.
- Greedy wrapping: a line accumulates words separated by single spaces; the next
  word joins the current line if `len(line) + 1 + len(word) <= width` (or the line
  is empty and `len(word) <= width`).
- **Overlong words** (longer than `width`): finish the current line (if non-empty),
  then hard-break the word into `width`-sized chunks; each full chunk is its own
  line, and the final chunk (which may be up to `width` long) starts a new current
  line that later words may join (subject to the width rule).
- Output has no trailing spaces, no trailing newline, and no leading/trailing blank
  lines. Input that is empty or entirely whitespace returns `""`.
- Words of length exactly `width` are ordinary words (their own line if they don't
  fit the current one).

Examples (`width=10`):
- `wrap_text("the quick brown fox", 10)` -> `"the quick\nbrown fox"`
- `wrap_text("hello extraordinary world", 10)` ->
  `"hello\nextraordin\nary world"`
- `wrap_text("a\n\n\nb", 10)` -> `"a\n\nb"`
- `wrap_text("abcdefghijklmno", 5)` -> `"abcde\nfghij\nklmno"`

No I/O. Standard library only (but not `textwrap`). Only `wrap_text` is tested.
