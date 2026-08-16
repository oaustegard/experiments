import pytest
from solution import parse_range


def test_single_int():
    assert parse_range("5") == [5]

def test_simple_range():
    assert parse_range("1-3") == [1, 2, 3]

def test_mixed_sorted_dedup():
    assert parse_range("7-9,5,1-3,8") == [1, 2, 3, 5, 7, 8, 9]

def test_overlap_dedup():
    assert parse_range("3,1-3") == [1, 2, 3]

def test_negative_single():
    assert parse_range("-5") == [-5]

def test_negative_range():
    assert parse_range("-3--1") == [-3, -2, -1]

def test_negative_to_positive():
    assert parse_range("-2-1") == [-2, -1, 0, 1]

def test_whitespace():
    assert parse_range(" 1 - 3 , 5 ") == [1, 2, 3, 5]

def test_empty_string():
    assert parse_range("") == []

def test_whitespace_only():
    assert parse_range("   ") == []

def test_single_element_range():
    assert parse_range("4-4") == [4]

def test_reversed_range_raises():
    with pytest.raises(ValueError):
        parse_range("5-3")

def test_double_comma_raises():
    with pytest.raises(ValueError):
        parse_range("1,,3")

def test_leading_comma_raises():
    with pytest.raises(ValueError):
        parse_range(",1")

def test_trailing_comma_raises():
    with pytest.raises(ValueError):
        parse_range("1,")

def test_garbage_raises():
    with pytest.raises(ValueError):
        parse_range("a")

def test_triple_hyphen_ambiguous_raises():
    with pytest.raises(ValueError):
        parse_range("1-2-3")

def test_float_raises():
    with pytest.raises(ValueError):
        parse_range("1.5")

def test_double_negative_raises():
    with pytest.raises(ValueError):
        parse_range("--3")
