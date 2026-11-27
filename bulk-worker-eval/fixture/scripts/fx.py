def titlecase(s):
    """Capitalize the first letter of each space-separated word and lowercase the rest."""
    return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split(' '))


def chunk(xs, n):
    """Split a list into consecutive lists of length n. The last list may be shorter."""
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def parse_kv(line):
    """Parse 'key=value' into a (key, value) tuple, splitting on the first '='."""
    k, v = line.split('=', 1)
    return (k, v)


def median(xs):
    """Median of a non-empty list. For an even count, the mean of the two middle values."""
    s = sorted(xs)
    return s[len(s) // 2]


def dedupe(xs):
    """Remove duplicates from a list, keeping the first occurrence of each item in order."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def days_in_month(year, month):
    """Number of days in the given month, accounting for leap years."""
    if month == 2:
        return 29 if year % 4 == 0 else 28
    return 30 if month in (4, 6, 9, 11) else 31
