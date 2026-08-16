# Task: template_render

Write a Python module defining exactly one public function:

```python
def render(template: str, values: dict) -> str
```

A small placeholder-substitution engine. Do not use `str.format`, `string.Template`,
or f-string tricks — implement the scan yourself.

Rules:
- `{name}` is a placeholder. `name` must match `[A-Za-z_][A-Za-z0-9_]*`.
- `{{` renders as a literal `{`, and `}}` renders as a literal `}`.
- Substituted values are converted with `str()`.
- Scanning is left to right; escapes are consumed greedily: `{{{x}}}` with `x=1`
  renders `{1}` (that is: `{{`, then placeholder `{x}`, then `}}`).
- Errors:
  - Any `{` that does not start a valid placeholder or a `{{` escape raises
    `ValueError` (e.g. `"{1x}"`, `"{ name }"`, `"{"`, `"{unclosed"`).
  - Any `}` that is not closing a placeholder and not part of `}}` raises `ValueError`
    (e.g. `"a}b"`).
  - If the template is well-formed but one or more placeholder names are missing from
    `values`, raise `KeyError` whose first argument (`e.args[0]`) is exactly the string
    `"missing keys: "` followed by the sorted, comma-separated unique missing names —
    e.g. `KeyError("missing keys: age, name")`. All missing names are collected before
    raising (do not stop at the first).
- A well-formed template with all keys present renders fully; keys in `values` that are
  unused are fine.
- Examples:
  - `render("Hello {name}!", {"name": "Ada"})` -> `"Hello Ada!"`
  - `render("{{literal}}", {})` -> `"{literal}"`
  - `render("{a}{b}", {})` -> raises `KeyError("missing keys: a, b")`

No I/O. Standard library only (`re` allowed). Only `render` is tested.
