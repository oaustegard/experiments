# Task: parse_range

Write a Python module defining exactly one public function:

```python
def parse_range(s: str) -> list[int]
```

Parse a comma-separated range expression into a sorted, deduplicated list of integers.

Rules:
- Tokens are separated by commas. Each token is either a single integer or a range `A-B`
  (inclusive on both ends), where `A` and `B` are integers.
- Integers may be negative: `-5` is a valid single token, and `-3--1` is the range from
  -3 to -1 (i.e. `[-3, -2, -1]`).
- Whitespace is allowed around tokens and around the range hyphen: `" 1 - 3 , 5 "` is valid.
- The result is sorted ascending with duplicates removed (`"3,1-3"` -> `[1, 2, 3]`).
- The empty string (or a string that is only whitespace) returns `[]`.
- Errors (raise `ValueError` in every case):
  - a range where A > B (`"5-3"`)
  - an empty token caused by leading/trailing/double commas (`"1,,3"`, `",1"`, `"1,"`)
  - any token that is not a valid integer or integer range (`"a"`, `"1-2-3"`, `"1.5"`, `"--3"`)

No I/O, no imports beyond the standard library. Only `parse_range` is tested.
