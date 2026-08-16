import pytest
from solution import toposort, CycleError


def test_empty():
    assert toposort([]) == []

def test_single_edge():
    assert toposort([("a", "b")]) == ["a", "b"]

def test_lexicographically_smallest():
    # both b and c available after a; must pick b first
    assert toposort([("a", "b"), ("a", "c")]) == ["a", "b", "c"]

def test_lex_smallest_across_components():
    assert toposort([("b", "d"), ("a", "c")]) == ["a", "b", "c", "d"]

def test_lex_pull_forward():
    # z->a: despite z being needed before a, ordering must be smallest overall
    assert toposort([("z", "a")]) == ["z", "a"]

def test_classic_diamond():
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    assert toposort(edges) == ["a", "b", "c", "d"]

def test_isolated_nodes_param():
    assert toposort([("b", "c")], nodes=["a", "z"]) == ["a", "b", "c", "z"]

def test_duplicate_edges_and_nodes():
    assert toposort([("a", "b"), ("a", "b")], nodes=["a", "b", "b"]) == ["a", "b"]

def test_cycle_error_is_valueerror():
    with pytest.raises(ValueError):
        toposort([("a", "b"), ("b", "a")])

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

def test_long_chain():
    edges = [(str(i), str(i + 1)) for i in range(9)]
    assert toposort(edges) == [str(i) for i in range(10)]
