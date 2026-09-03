# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
____________________________ test_char_class_range _____________________________
    def test_char_class_range():
>       assert glob_match("[a-z]x", "mx")
E       AssertionError: assert False
E        +  where False = glob_match('[a-z]x', 'mx')
tests/test_public.py:6: AssertionError
___________________________ test_char_class_negation ___________________________
    def test_char_class_negation():
        assert glob_match("[!a-c]x", "dx")
>       assert not glob_match("[!a-c]x", "ax")
E       AssertionError: assert not True
E        +  where True = glob_match('[!a-c]x', 'ax')
tests/test_public.py:13: AssertionError
=========================== short test summary info ============================
FAILED tests/test_public.py::test_char_class_range - AssertionError: assert F...
FAILED tests/test_public.py::test_char_class_negation - AssertionError: asser...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `glob_match`.
