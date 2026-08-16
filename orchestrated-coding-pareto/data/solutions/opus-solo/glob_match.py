"""Slash-separated path matching against git/.gitignore-style glob patterns.

Public API: ``glob_match(pattern, path) -> bool``.

Implemented from scratch: no ``fnmatch``, ``glob``, ``pathlib`` or ``re``.
"""

__all__ = ["glob_match"]


# --- token markers ---------------------------------------------------------

_STAR = object()          # '*' within a segment: zero+ chars, never '/'
_ANY = object()           # '?' within a segment: exactly one char, never '/'
_DOUBLESTAR = object()    # a whole pattern segment equal to '**'


class _CharClass(object):
    """A `[...]` character class: literal chars, ranges, optional negation."""

    __slots__ = ("negated", "chars", "ranges")

    def __init__(self, negated, chars, ranges):
        self.negated = negated
        self.chars = chars
        self.ranges = ranges

    def matches(self, ch):
        if ch == "/":
            return False
        found = ch in self.chars
        if not found:
            for lo, hi in self.ranges:
                if lo <= ch <= hi:
                    found = True
                    break
        return (not found) if self.negated else found


class _Literal(object):
    """A single literal character."""

    __slots__ = ("ch",)

    def __init__(self, ch):
        self.ch = ch

    def matches(self, ch):
        return ch == self.ch


# --- compilation -----------------------------------------------------------

def _parse_class(seg, i):
    """Parse a character class starting at ``seg[i] == '['``.

    Returns ``(_CharClass, next_index)``. Raises ValueError if unterminated.
    """
    n = len(seg)
    j = i + 1
    negated = False
    if j < n and seg[j] == "!":
        negated = True
        j += 1

    chars = set()
    ranges = []
    first = True
    while True:
        if j >= n:
            raise ValueError("unterminated character class in pattern segment: %r" % (seg,))
        c = seg[j]
        if c == "]" and not first:
            j += 1
            break
        first = False
        # A range is `x-y` where the char after '-' exists and is not the
        # closing ']' (so a trailing '-' stays literal).
        if j + 2 < n and seg[j + 1] == "-" and seg[j + 2] != "]":
            lo = c
            hi = seg[j + 2]
            if lo <= hi:
                ranges.append((lo, hi))
            j += 3
        else:
            chars.add(c)
            j += 1

    return _CharClass(negated, chars, ranges), j


def _compile_segment(seg):
    """Compile one pattern segment into a list of tokens."""
    tokens = []
    i = 0
    n = len(seg)
    while i < n:
        c = seg[i]
        if c == "*":
            # Collapse runs of '*' -- '**' inside a segment is just '*'.
            if not tokens or tokens[-1] is not _STAR:
                tokens.append(_STAR)
            i += 1
        elif c == "?":
            tokens.append(_ANY)
            i += 1
        elif c == "[":
            cls, i = _parse_class(seg, i)
            tokens.append(cls)
        else:
            tokens.append(_Literal(c))
            i += 1
    return tokens


def _compile_pattern(pattern):
    """Compile a whole pattern into a list of per-segment token lists.

    A segment that is exactly ``**`` becomes the ``_DOUBLESTAR`` marker.
    """
    if pattern == "":
        return []
    compiled = []
    for seg in pattern.split("/"):
        if seg == "**":
            compiled.append(_DOUBLESTAR)
        else:
            compiled.append(_compile_segment(seg))
    return compiled


# --- matching --------------------------------------------------------------

def _match_segment(tokens, s):
    """Match one path segment against compiled within-segment tokens."""
    ti = 0
    si = 0
    nt = len(tokens)
    ns = len(s)
    star_ti = -1
    star_si = 0

    while si < ns:
        if ti < nt:
            tok = tokens[ti]
            if tok is _STAR:
                star_ti = ti
                star_si = si
                ti += 1
                continue
            if tok is _ANY:
                if s[si] != "/":
                    ti += 1
                    si += 1
                    continue
            elif tok.matches(s[si]):
                ti += 1
                si += 1
                continue
        if star_ti >= 0:
            star_si += 1
            si = star_si
            ti = star_ti + 1
            continue
        return False

    while ti < nt and tokens[ti] is _STAR:
        ti += 1
    return ti == nt


def _match_from(pat_segs, path_segs, pi, si, memo):
    key = (pi, si)
    cached = memo.get(key)
    if cached is not None:
        return cached

    np = len(pat_segs)
    ns = len(path_segs)

    while True:
        if pi == np:
            result = si == ns
            memo[(pi, si)] = result
            return result

        p = pat_segs[pi]
        if p is _DOUBLESTAR:
            if pi == np - 1:
                # Trailing '**'. The pattern '**' alone matches every path,
                # including the empty path; otherwise a trailing '/**'
                # requires at least one following path segment.
                result = True if np == 1 else si < ns
                memo[(pi, si)] = result
                return result
            for k in range(si, ns + 1):
                if _match_from(pat_segs, path_segs, pi + 1, k, memo):
                    memo[(pi, si)] = True
                    return True
            memo[(pi, si)] = False
            return False

        if si >= ns or not _match_segment(p, path_segs[si]):
            memo[(pi, si)] = False
            return False

        pi += 1
        si += 1


def glob_match(pattern, path):
    """Return True if ``path`` matches the glob ``pattern`` entirely.

    Raises ValueError if the pattern contains an unterminated ``[`` class.
    """
    pat_segs = _compile_pattern(pattern)
    path_segs = [] if path == "" else path.split("/")
    return _match_from(pat_segs, path_segs, 0, 0, {})
