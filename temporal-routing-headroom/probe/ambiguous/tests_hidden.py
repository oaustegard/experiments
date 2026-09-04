import pytest
from solution import merge


def test_empty():
    assert merge([]) == []

def test_single():
    assert merge([(1, 2)]) == [(1, 2)]

def test_overlap_merged():
    assert merge([(1, 3), (2, 5)]) == [(1, 5)]

def test_disjoint_sorted():
    assert merge([(5, 6), (1, 2)]) == [(1, 2), (5, 6)]

def test_touching_joined_by_default():
    assert merge([(1, 2), (2, 3)]) == [(1, 3)]

def test_touching_kept_when_flag_false():
    assert merge([(1, 2), (2, 3)], join_touching=False) == [(1, 2), (2, 3)]

def test_transitive_chain():
    assert merge([(3, 4), (1, 2), (2, 3)]) == [(1, 4)]

def test_containment_always_merges():
    assert merge([(1, 4), (2, 3)], join_touching=False) == [(1, 4)]

def test_duplicates_collapse():
    assert merge([(1, 2), (1, 2)], join_touching=False) == [(1, 2)]

def test_degenerate_point():
    assert merge([(2, 2)]) == [(2, 2)]

def test_point_on_end_touching():
    assert merge([(1, 2), (2, 2)]) == [(1, 2)]
    assert merge([(1, 2), (2, 2)], join_touching=False) == [(1, 2), (2, 2)]

def test_duplicate_points_collapse():
    assert merge([(2, 2), (2, 2)], join_touching=False) == [(2, 2)]

def test_point_inside_interval():
    assert merge([(1, 3), (2, 2)], join_touching=False) == [(1, 3)]

def test_floats():
    assert merge([(0.5, 1.5), (1.0, 2.5)]) == [(0.5, 2.5)]

def test_reversed_interval_raises():
    with pytest.raises(ValueError):
        merge([(3, 1)])

def test_result_is_tuples():
    out = merge([(1, 3), (2, 5)])
    assert all(isinstance(x, tuple) for x in out)

def test_unsorted_many():
    ivs = [(8, 10), (1, 3), (2, 6), (15, 18), (17, 20)]
    assert merge(ivs) == [(1, 6), (8, 10), (15, 20)]
