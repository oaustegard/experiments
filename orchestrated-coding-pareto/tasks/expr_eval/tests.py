import pytest
from solution import evaluate


def test_basic_precedence():
    assert evaluate("1+2*3") == 7.0
    assert evaluate("2*3+1") == 7.0
    assert evaluate("10-4/2") == 8.0

def test_left_assoc_add_sub():
    assert evaluate("10-3-2") == 5.0
    assert evaluate("100/10/5") == 2.0

def test_power_right_assoc():
    assert evaluate("2**3**2") == 512.0

def test_unary_minus_looser_than_power():
    assert evaluate("-2**2") == -4.0
    assert evaluate("-2**2 + 1") == -3.0

def test_unary_after_power_binds_tight():
    assert evaluate("2**-1") == 0.5
    assert evaluate("2**-2**2") == 0.0625
    assert evaluate("4**-0.5") == 0.5

def test_stacked_unary():
    assert evaluate("--3") == 3.0
    assert evaluate("+-+3") == -3.0
    assert evaluate("-(-3)") == 3.0

def test_modulo_python_sign():
    assert evaluate("-7 % 3") == 2.0
    assert evaluate("7 % -3") == -2.0

def test_parens():
    assert evaluate("(1+2)*3") == 9.0
    assert evaluate("((2))") == 2.0
    assert evaluate("2*(3+(4-1))") == 12.0

def test_decimals():
    assert evaluate("2.5*2") == 5.0
    assert evaluate(".5+.5") == 1.0
    assert evaluate("10.") == 10.0

def test_whitespace():
    assert evaluate("  1 +\t2 ") == 3.0

def test_zero_power_zero():
    assert evaluate("0**0") == 1.0

def test_returns_float():
    assert isinstance(evaluate("1+1"), float)

def test_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        evaluate("1/0")
    with pytest.raises(ZeroDivisionError):
        evaluate("1%0")

def test_precedence_mul_vs_power():
    assert evaluate("2*3**2") == 18.0
    assert evaluate("3**2*2") == 18.0

@pytest.mark.parametrize("bad", [
    "", "   ", "1+", "*3", "1 2", "1..2", "1.2.3", "()", "(1+2", "1+2)",
    "1+*2", "abc", "1&2", "2**", "((1)", "-", "+",
])
def test_malformed_raises(bad):
    with pytest.raises(ValueError):
        evaluate(bad)
