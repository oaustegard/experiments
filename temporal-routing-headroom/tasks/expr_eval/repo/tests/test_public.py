import pytest
from solution import evaluate


def test_power_right_assoc():
    assert evaluate("2**3**2") == 512.0


def test_unary_after_power_binds_tight():
    assert evaluate("2**-1") == 0.5
    assert evaluate("2**-2**2") == 0.0625
    assert evaluate("4**-0.5") == 0.5


def test_precedence_mul_vs_power():
    assert evaluate("2*3**2") == 18.0
    assert evaluate("3**2*2") == 18.0
