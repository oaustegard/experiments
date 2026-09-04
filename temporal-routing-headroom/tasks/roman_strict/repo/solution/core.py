_VALS = [(1000, "M"), (900, "DCCCC"), (500, "D"), (400, "CD"), (100, "C"),
         (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
         (5, "V"), (4, "IV"), (1, "I")]


def _to_roman(n: int) -> str:
    out = []
    for v, sym in _VALS:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


_TABLE = {_to_roman(n): n for n in range(1, 4000)}


def from_roman(s: str) -> int:
    if not isinstance(s, str) or s not in _TABLE:
        raise ValueError(f"not a strict Roman numeral: {s!r}")
    return _TABLE[s]
