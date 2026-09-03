import pytest
from solution import merge


def test_containment_always_merges():
    assert merge([(1, 4), (2, 3)], join_touching=False) == [(1, 4)]


def test_point_inside_interval():
    assert merge([(1, 3), (2, 2)], join_touching=False) == [(1, 3)]
