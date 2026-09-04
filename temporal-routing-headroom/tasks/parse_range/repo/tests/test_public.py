import pytest
from solution import parse_range


def test_mixed_sorted_dedup():
    assert parse_range("7-9,5,1-3,8") == [1, 2, 3, 5, 7, 8, 9]


def test_negative_range():
    assert parse_range("-3--1") == [-3, -2, -1]


def test_negative_to_positive():
    assert parse_range("-2-1") == [-2, -1, 0, 1]
