import pytest
from solution import merge


def test_transitive_chain():
    assert merge([(3, 4), (1, 2), (2, 3)]) == [(1, 4)]
