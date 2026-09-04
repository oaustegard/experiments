import pytest
from solution import toposort, CycleError
def _check_cycle(exc, edges):
    edge_set = set(edges)
    cyc = exc.cycle
    assert isinstance(cyc, list) and len(cyc) >= 1
    assert len(set(cyc)) == len(cyc), "cycle nodes must be distinct"
    for i in range(len(cyc) - 1):
        assert (cyc[i], cyc[i + 1]) in edge_set
    assert (cyc[-1], cyc[0]) in edge_set


def test_two_cycle_reported():
    edges = [("a", "b"), ("b", "a")]
    with pytest.raises(CycleError) as ei:
        toposort(edges)
    _check_cycle(ei.value, edges)


def test_self_loop_reported():
    edges = [("a", "a")]
    with pytest.raises(CycleError) as ei:
        toposort(edges)
    _check_cycle(ei.value, edges)


def test_cycle_in_larger_graph():
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "b"), ("a", "e")]
    with pytest.raises(CycleError) as ei:
        toposort(edges)
    _check_cycle(ei.value, edges)
