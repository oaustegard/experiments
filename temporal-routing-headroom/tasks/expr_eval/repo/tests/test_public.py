import pytest
from solution import evaluate


def test_unary_minus_looser_than_power():
    assert evaluate("-2**2") == -4.0
    assert evaluate("-2**2 + 1") == -3.0


def test_zero_power_zero():
    assert evaluate("0**0") == 1.0


def test_precedence_mul_vs_power():
    assert evaluate("2*3**2") == 18.0
    assert evaluate("3**2*2") == 18.0
