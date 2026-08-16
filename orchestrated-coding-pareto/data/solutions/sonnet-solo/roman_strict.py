"""Strict classical Roman numeral parser.

Parses only the canonical (strict) Roman numeral representation of integers
in the range 1..3999, as produced by the standard greedy encoding:

    1000=M, 900=CM, 500=D, 400=CD, 100=C, 90=XC, 50=L, 40=XL,
    10=X, 9=IX, 5=V, 4=IV, 1=I
"""

_VALUES = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

_CANONICAL_NUMERALS = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def _to_canonical_roman(n: int) -> str:
    """Encode an integer in 1..3999 using the standard greedy algorithm."""
    parts = []
    remaining = n
    for value, symbol in _CANONICAL_NUMERALS:
        count, remaining = divmod(remaining, value)
        if count:
            parts.append(symbol * count)
    return "".join(parts)


def from_roman(s: str) -> int:
    """Parse a strictly classical Roman numeral string into an integer.

    Raises ValueError for empty strings, whitespace, non-Roman characters,
    lowercase/mixed case, non-canonical repetition, invalid subtractive
    pairs, wrong ordering, or values outside 1..3999.
    """
    if not isinstance(s, str):
        raise ValueError("Roman numeral must be a string")

    if not s:
        raise ValueError("Roman numeral must not be empty")

    for ch in s:
        if ch not in _VALUES:
            raise ValueError(f"invalid character in Roman numeral: {ch!r}")

    total = 0
    prev_value = 0
    for ch in reversed(s):
        value = _VALUES[ch]
        if value < prev_value:
            total -= value
        else:
            total += value
            prev_value = value

    if not (1 <= total <= 3999):
        raise ValueError("value out of range 1..3999")

    if _to_canonical_roman(total) != s:
        raise ValueError(f"not a strictly canonical Roman numeral: {s!r}")

    return total
