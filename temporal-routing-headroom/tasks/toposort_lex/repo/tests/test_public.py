import pytest
from solution import toposort, CycleError


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
