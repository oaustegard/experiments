# Task: text_table

Write a Python module defining exactly one public function:

```python
def format_table(rows: list[list], headers: list[str], *,
                 aligns: list[str] | None = None,
                 max_col_width: int = 20) -> str
```

Render an ASCII table with exact output rules. Every rule below is tested literally —
match them to the character.

Layout:
- Border and separator style:
  - top border, header row, header separator, data rows, bottom border
  - corners and intersections are `+`, horizontal lines are `-`, verticals are `|`
- Every cell is rendered as `" "` + content padded to the column width + `" "`
  (one space padding on each side, always).
- Column width = the larger of the header's display width and the widest cell line
  in that column, but never more than `max_col_width`.

Content rules:
- Cell values are converted with `str()`. `None` renders as the empty string.
- A cell whose text exceeds the column width is **wrapped** by hard-breaking into
  chunks of exactly the column width (no word awareness); the row then occupies as
  many physical lines as its tallest cell. Shorter cells pad with blank lines
  **below** (top-aligned).
- Alignment (`aligns`, one of `"l"`, `"r"`, `"c"` per column; default all `"l"`):
  - `"l"`: content left-justified in the column width
  - `"r"`: right-justified
  - `"c"`: centered — when the leftover space is odd, the extra space goes on the
    **right**
  - Headers use the same alignment as their column's data.
  - Wrapped chunks are aligned individually.
- Newlines already present in a cell (`"a\nb"`) split it into lines first; each
  line is then independently hard-wrapped if over-width.
- Headers wrap exactly like cells when they exceed the column width (the header
  "row" then spans several physical lines, followed by the header separator).
- With no data rows, the output is: top border, header line(s), header separator,
  bottom border (nothing between the last two).

Validation (raise `ValueError`):
- `headers` empty
- any row whose length differs from `len(headers)`
- `aligns` given but wrong length, or containing anything other than `"l"/"r"/"c"`
- `max_col_width < 1`

Exact example — `format_table([[1, "hi"], [22, "world!"]], ["n", "word"])` with
`max_col_width=5` returns exactly (no trailing spaces trimmed — pad every cell to
full width; no trailing newline):

```
+----+-------+
| n  | word  |
+----+-------+
| 1  | hi    |
| 22 | world |
|    | !     |
+----+-------+
```

(Column "word": widest value `"world!"` is 6 > 5, so width is 5 -> wraps to
`world` / `!`; header separator dashes span width+2 for the padding spaces.)

No I/O. Standard library only. Only `format_table` is tested.
