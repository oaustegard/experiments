# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
________________________ test_containment_always_merges ________________________
    def test_containment_always_merges():
>       assert merge([(1, 4), (2, 3)], join_touching=False) == [(1, 4)]
E       assert [(1, 3)] == [(1, 4)]
E         
E         At index 0 diff: (1, 3) != (1, 4)
E         Use -v to get more diff
tests/test_public.py:6: AssertionError
__________________________ test_point_inside_interval __________________________
    def test_point_inside_interval():
>       assert merge([(1, 3), (2, 2)], join_touching=False) == [(1, 3)]
E       assert [(1, 2)] == [(1, 3)]
E         
E         At index 0 diff: (1, 2) != (1, 3)
E         Use -v to get more diff
tests/test_public.py:10: AssertionError
=========================== short test summary info ============================
FAILED tests/test_public.py::test_containment_always_merges - assert [(1, 3)]...
FAILED tests/test_public.py::test_point_inside_interval - assert [(1, 2)] == ...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `merge`.
