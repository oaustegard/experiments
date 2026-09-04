# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
_____________________________ test_escaped_quotes ______________________________
    def test_escaped_quotes():
>       assert parse_csv_line('"he said ""hi""",x') == ['he said "hi"', "x"]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_public.py:6: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
line = '"he said ""hi""",x'
    def parse_csv_line(line: str) -> list[str]:
        fields = []
        buf = []
        i = 0
        n = len(line)
        while True:
            # parse one field starting at i
            if i < n and line[i] == '"':
                i += 1
                closed = False
                while i < n:
                    ch = line[i]
                    if ch == '"':
                        if i + 1 < n and line[i + 1] == "'":
                            buf.append('"')
...
FAILED tests/test_public.py::test_escaped_quotes - ValueError: character afte...
FAILED tests/test_public.py::test_only_escaped_quote - ValueError: character ...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `parse_csv_line`.
