# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
___________________________ test_mixed_sorted_dedup ____________________________
    def test_mixed_sorted_dedup():
>       assert parse_range("7-9,5,1-3,8") == [1, 2, 3, 5, 7, 8, 9]
E       assert [1, 2, 5, 7, 8] == [1, 2, 3, 5, 7, 8, ...]
E         
E         At index 2 diff: 5 != 3
E         Right contains 2 more items, first extra item: 8
E         Use -v to get more diff
tests/test_public.py:6: AssertionError
_____________________________ test_negative_range ______________________________
    def test_negative_range():
>       assert parse_range("-3--1") == [-3, -2, -1]
E       assert [-3, -2] == [-3, -2, -1]
E         
E         Right contains one more item: -1
E         Use -v to get more diff
tests/test_public.py:10: AssertionError
__________________________ test_negative_to_positive ___________________________
    def test_negative_to_positive():
>       assert parse_range("-2-1") == [-2, -1, 0, 1]
E       assert [-2, -1, 0] == [-2, -1, 0, 1]
E         
...
FAILED tests/test_public.py::test_negative_range - assert [-3, -2] == [-3, -2...
FAILED tests/test_public.py::test_negative_to_positive - assert [-2, -1, 0] =...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `parse_range`.
