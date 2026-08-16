# Task: csv_line

Write a Python module defining exactly one public function:

```python
def parse_csv_line(line: str) -> list[str]
```

Parse a single CSV record (RFC 4180 style) into its fields. Do **not** use the `csv`
module — implement the state machine yourself.

Rules:
- Fields are separated by commas.
- A field may be quoted: it then starts with `"` as its **first** character and ends with
  a matching `"`. Inside a quoted field, commas are literal, and an escaped quote is
  written as two double quotes (`""` -> one literal `"`).
- Quoted fields may contain any characters, including newlines if present in the string.
- The empty string parses to `[""]` (one empty field). `","` parses to `["", ""]`.
- Whitespace is not special: ` a ` stays `" a "`, and a space before a quote makes the
  quote part of an unquoted field — which is an error (see next rule).
- Errors (raise `ValueError` in every case):
  - a `"` appearing anywhere inside an **unquoted** field (`a"b`, ` "a"`)
  - any character other than a comma (or end of string) immediately after the closing
    quote of a quoted field (`"a"b`, `"a" ,b`)
  - an unterminated quoted field (`"abc`, `"a""`)
- Valid examples:
  - `a,b,c` -> `["a", "b", "c"]`
  - `"a,b",c` -> `["a,b", "c"]`
  - `"he said ""hi""",x` -> `['he said "hi"', "x"]`
  - `""` -> `[""]`, and `"",` -> `["", ""]`

No I/O. Standard library only (but not `csv`). Only `parse_csv_line` is tested.
