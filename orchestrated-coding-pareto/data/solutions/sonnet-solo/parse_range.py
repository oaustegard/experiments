import re

_INT_RE = re.compile(r'^-?\d+$')
_RANGE_RE = re.compile(r'^(-?\d+)\s*-\s*(-?\d+)$')


def parse_range(s: str) -> list[int]:
    """Parse a comma-separated range expression into a sorted, deduplicated
    list of integers.

    Tokens are separated by commas. Each token is either a single integer
    or an inclusive range 'A-B'. Whitespace around tokens and around the
    range hyphen is allowed. Raises ValueError for malformed input.
    """
    if s.strip() == '':
        return []

    values: set[int] = set()

    for raw_token in s.split(','):
        token = raw_token.strip()

        if token == '':
            raise ValueError(f'empty token in range expression: {s!r}')

        int_match = _INT_RE.match(token)
        if int_match is not None:
            values.add(int(token))
            continue

        range_match = _RANGE_RE.match(token)
        if range_match is not None:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start > end:
                raise ValueError(
                    f'invalid range (start > end): {token!r}'
                )
            values.update(range(start, end + 1))
            continue

        raise ValueError(f'invalid token in range expression: {token!r}')

    return sorted(values)
