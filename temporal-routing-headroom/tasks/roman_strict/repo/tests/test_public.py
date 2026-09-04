import pytest
from solution import from_roman
_VALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
         (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
         (5, "V"), (4, "IV"), (1, "I")]
def _to_roman(n):
    out = []
    for v, sym in _VALS:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def test_basic_values():
    assert from_roman("I") == 1
    assert from_roman("IV") == 4
    assert from_roman("IX") == 9
    assert from_roman("XIV") == 14
    assert from_roman("XL") == 40
    assert from_roman("XC") == 90
    assert from_roman("CD") == 400
    assert from_roman("CM") == 900


def test_famous():
    assert from_roman("MCMXCIV") == 1994
    assert from_roman("MMMCMXCIX") == 3999
    assert from_roman("MMXXVI") == 2026


def test_full_roundtrip_1_to_3999():
    for n in range(1, 4000):
        assert from_roman(_to_roman(n)) == n
