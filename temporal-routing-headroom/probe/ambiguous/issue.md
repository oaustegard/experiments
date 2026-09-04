# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
F                                                                        [100%]
=================================== FAILURES ===================================
____________________________ test_transitive_chain _____________________________
    def test_transitive_chain():
>       assert merge([(3, 4), (1, 2), (2, 3)]) == [(1, 4)]
E       assert [(1, 2), (2, 3), (3, 4)] == [(1, 4)]
E         
E         At index 0 diff: (1, 2) != (1, 4)
E         Left contains 2 more items, first extra item: (2, 3)
E         Use -v to get more diff
tests/test_public.py:6: AssertionError
=========================== short test summary info ============================
FAILED tests/test_public.py::test_transitive_chain - assert [(1, 2), (2, 3), ...
1 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: merge.
