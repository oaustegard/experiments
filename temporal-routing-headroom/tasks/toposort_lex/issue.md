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
tests/test_public.py:17: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
exc = CycleError("cycle detected: ['b']"), edges = [('a', 'b'), ('b', 'a')]
    def _check_cycle(exc, edges):
        edge_set = set(edges)
        cyc = exc.cycle
        assert isinstance(cyc, list) and len(cyc) >= 1
        assert len(set(cyc)) == len(cyc), "cycle nodes must be distinct"
        for i in range(len(cyc) - 1):
            assert (cyc[i], cyc[i + 1]) in edge_set
>       assert (cyc[-1], cyc[0]) in edge_set
E       AssertionError: assert ('b', 'b') in {('a', 'b'), ('b', 'a')}
tests/test_public.py:10: AssertionError
___________________________ test_self_loop_reported ____________________________
    def test_self_loop_reported():
        edges = [("a", "a")]
...
FAILED tests/test_public.py::test_self_loop_reported - assert (False)
FAILED tests/test_public.py::test_cycle_in_larger_graph - AssertionError: ass...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `toposort, CycleError`.
