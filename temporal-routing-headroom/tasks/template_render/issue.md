# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
__________________________________ test_basic __________________________________
    def test_basic():
>       assert render("Hello {name}!", {"name": "Ada"}) == "Hello Ada!"
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_public.py:6: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
template = 'Hello {name}!', values = {'name': 'Ada'}
    def render(template: str, values: dict) -> str:
        parts = []
        missing = set()
        i = 0
        n = len(template)
        while i < n:
            ch = template[i]
            if ch == "{":
                if i + 1 < n and template[i + 1] == "{":
                    parts.append("{")
                    i += 2
                    continue
                m = _NAME.match(template, i + 1)
                if not m or m.end() >= n or template[m.end()] != "}":
>                   raise ValueError(f"malformed placeholder at index {i}")
...
FAILED tests/test_public.py::test_underscore_names - ValueError: malformed pl...
FAILED tests/test_public.py::test_missing_keys_message_exact - ValueError: ma...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `render`.
