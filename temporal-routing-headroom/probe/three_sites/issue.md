# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
_____________________ test_center_align_extra_space_right ______________________
    def test_center_align_extra_space_right():
        # width 3 ("abc"), value "a" -> leftover 2 -> 1 left 1 right: " a "
        # value "ab" -> leftover 1 -> extra goes right: "ab " with no left space
        out = format_table([["a"], ["ab"]], ["abc"], aligns=["c"])
        lines = out.split("\n")
>       assert lines[3] == "|  a  |"
E       AssertionError: assert '+----+' == '|  a  |'
E         
E         - |  a  |
E         + +----+
tests/test_public.py:10: AssertionError
______________________________ test_empty_rows_ok ______________________________
    def test_empty_rows_ok():
        out = format_table([], ["a", "b"])
        expected = (
            "+---+---+\n"
            "| a | b |\n"
            "+---+---+\n"
            "+---+---+"
        )
>       assert out == expected
...
FAILED tests/test_public.py::test_center_align_extra_space_right - AssertionE...
FAILED tests/test_public.py::test_empty_rows_ok - AssertionError: assert '+--...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: format_table.
