# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
______________________________ test_exact_example ______________________________
    def test_exact_example():
        out = format_table([[1, "hi"], [22, "world!"]], ["n", "word"], max_col_width=5)
        expected = (
            "+----+-------+\n"
            "| n  | word  |\n"
            "+----+-------+\n"
            "| 1  | hi    |\n"
            "| 22 | world |\n"
            "|    | !     |\n"
            "+----+-------+"
        )
>       assert out == expected
E       AssertionError: assert '+----+------...----+-------+' == '+----+------...----+-------+'
E         
E           +----+-------+
E         - | n  | word  |
E         ?     -       -
E         + | n | word |
E           +----+-------+
E         - | 1  | hi    |...
E         
...
FAILED tests/test_public.py::test_header_wider_than_data - AssertionError: as...
FAILED tests/test_public.py::test_header_wraps_too - AssertionError: assert '...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `format_table`.
