# Task: glob_match

Write a Python module defining exactly one public function:

```python
def glob_match(pattern: str, path: str) -> bool
```

Match a slash-separated path against a glob pattern. Do **not** use `fnmatch`,
`glob`, or `pathlib` — implement the matching yourself. (Building your own regex is
allowed, but the `**` segment semantics below must be implemented correctly.)

Semantics (git/.gitignore-style):
- The path is a `/`-separated string like `"src/lib/util.py"`, with no leading or
  trailing slash. The empty path `""` has zero segments; any other path's segments
  are non-empty.
- `?` matches exactly one character **within a segment** (never `/`).
- `*` matches zero or more characters **within a segment** (never `/`).
- `**` as a **whole pattern segment** matches **zero or more whole path segments**:
  - `"a/**/b"` matches `"a/b"` (zero segments), `"a/x/b"`, `"a/x/y/b"`.
  - `"**/b"` matches `"b"`, `"x/b"`, `"x/y/b"`.
  - `"a/**"` matches `"a/x"`, `"a/x/y"` — but **not** `"a"` (matching zero segments
    here would leave a dangling slash; a trailing `/**` requires at least one
    following segment).
  - The pattern `"**"` on its own matches every path, including the empty path.
- A `**` appearing **inside** a segment (e.g. `"a**b"`, `"f**"`) is not special:
  it behaves exactly like `*` (within-segment, never matches `/`).
- Character classes: `[abc]` matches one char from the set; `[a-z]` is a range;
  `[!...]` negates. A class never matches `/`. A `]` as the **first** class
  character is a literal (`[]]` matches `]`; `[!]]` matches any one char except
  `]`). A `-` first or last in the class is a literal. An unterminated `[` raises
  `ValueError`. A `/` inside a class is unsupported (the pattern is interpreted
  segment-wise); such patterns are not tested.
- There is no escape character; every other character matches itself literally.
- The empty pattern `""` matches only the empty path.
- Matching is anchored: the whole pattern must match the whole path.

Examples:
- `glob_match("*.py", "util.py")` -> True; `glob_match("*.py", "lib/util.py")` -> False
- `glob_match("**/*.py", "a/b/c.py")` -> True, and True for `"c.py"` too
- `glob_match("a/**/b", "a/b")` -> True; `glob_match("a/**", "a")` -> False
- `glob_match("[!a-c]x", "dx")` -> True; `glob_match("[!a-c]x", "ax")` -> False

No I/O. Standard library only. Only `glob_match` is tested.
