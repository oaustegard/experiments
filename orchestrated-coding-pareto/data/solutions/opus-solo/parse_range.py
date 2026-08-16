"""Parse comma-separated range expressions into sorted, deduplicated integer lists."""

import re

__all__ = ["parse_range"]

_INT_RE = re.compile(r"^-?\d+$")
_RANGE_RE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")


def parse_range(s: str) -> list[int]:
    """Parse a comma-separated range expression into a sorted, deduplicated list.

    Each comma-separated token is either a single integer (``"5"``, ``"-5"``) or an
    inclusive range ``A-B`` (``"1-3"``, ``"-3--1"``). Whitespace around tokens and
    around the range hyphen is ignored. An empty or whitespace-only string yields
    an empty list.

    Raises:
        ValueError: if a range has A > B, if a token is empty (leading, trailing,
            or doubled commas), or if a token is not a valid integer or range.
    """
    if not isinstance(s, str):
        raise ValueError("input must be a string")

    if s.strip() == "":
        return []

    values: set[int] = set()

    for raw_token in s.split(","):
        token = raw_token.strip()

        if token == "":
            raise ValueError("empty token in range expression: {!r}".format(s))

        if _INT_RE.match(token):
            values.add(int(token))
            continue

        match = _RANGE_RE.match(token)
        if match is None:
            raise ValueError("invalid token: {!r}".format(token))

        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            raise ValueError(
                "invalid range (start greater than end): {!r}".format(token)
            )

        values.update(range(start, end + 1))

    return sorted(values)
