# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
___________________________ test_two_cycle_reported ____________________________
    def test_two_cycle_reported():
        edges = [("a", "b"), ("b", "a")]
        with pytest.raises(CycleError) as ei:
            toposort(edges)
>       _check_cycle(ei.value, edges)
        ^^^^^^^^^^^^
E       NameError: name '_check_cycle' is not defined
tests/test_public.py:9: NameError
___________________________ test_self_loop_reported ____________________________
    def test_self_loop_reported():
        edges = [("a", "a")]
        with pytest.raises(CycleError) as ei:
            toposort(edges)
>       _check_cycle(ei.value, edges)
        ^^^^^^^^^^^^
E       NameError: name '_check_cycle' is not defined
tests/test_public.py:16: NameError
__________________________ test_cycle_in_larger_graph __________________________
    def test_cycle_in_larger_graph():
        edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "b"), ("a", "e")]
        with pytest.raises(CycleError) as ei:
...
FAILED tests/test_public.py::test_self_loop_reported - NameError: name '_chec...
FAILED tests/test_public.py::test_cycle_in_larger_graph - NameError: name '_c...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `toposort, CycleError`.
