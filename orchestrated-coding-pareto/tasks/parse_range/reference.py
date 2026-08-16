import re

_SINGLE = re.compile(r"^-?\d+$")
_RANGE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")


def parse_range(s: str) -> list[int]:
    if s.strip() == "":
        return []
    out = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            raise ValueError("empty token")
        if _SINGLE.match(token):
            out.add(int(token))
            continue
        m = _RANGE.match(token)
        if not m:
            raise ValueError(f"bad token: {token!r}")
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            raise ValueError(f"reversed range: {token!r}")
        out.update(range(a, b + 1))
    return sorted(out)
