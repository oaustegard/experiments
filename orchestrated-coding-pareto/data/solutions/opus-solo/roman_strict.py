"""Strict classical Roman numeral parsing.

Only the canonical (greedy) encoding of an integer in 1..3999 is accepted.
Any other spelling raises ``ValueError``.
"""

__all__ = ["from_roman"]

_VALUES = (
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

_SYMBOLS = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _to_roman(n: int) -> str:
    """Canonical greedy encoding of ``n`` (assumed to be in 1..3999)."""
    out = []
    for value, glyph in _VALUES:
        count, n = divmod(n, value)
        if count:
            out.append(glyph * count)
    return "".join(out)


def from_roman(s: str) -> int:
    """Parse a strictly canonical Roman numeral and return its value.

    Raises ``ValueError`` for anything that is not the exact canonical
    representation of an integer in 1..3999.
    """
    if not isinstance(s, str):
        raise ValueError("roman numeral must be a string")
    if not s:
        raise ValueError("empty string is not a valid Roman numeral")

    # Reject whitespace, lowercase, and any non-Roman character up front.
    for ch in s:
        if ch not in _SYMBOLS:
            raise ValueError("invalid character in Roman numeral: %r" % (ch,))

    # Accumulate with the standard subtractive rule; the canonical check below
    # is what actually enforces strictness.
    total = 0
    i = 0
    length = len(s)
    while i < length:
        value = _SYMBOLS[s[i]]
        if i + 1 < length and _SYMBOLS[s[i + 1]] > value:
            total += _SYMBOLS[s[i + 1]] - value
            i += 2
        else:
            total += value
            i += 1

    if not 1 <= total <= 3999:
        raise ValueError("Roman numeral out of range 1..3999: %r" % (s,))

    if _to_roman(total) != s:
        raise ValueError("not a canonical Roman numeral: %r" % (s,))

    return total
