def _parse_class(seg: str, i: int):
    """Parse a class starting at seg[i] == '['; return (matcher, next_index)."""
    j = i + 1
    negate = False
    if j < len(seg) and seg[j] == "!":
        negate = True
        j += 1
    chars = set()
    ranges = []
    first = True
    while True:
        if j >= len(seg):
            raise ValueError("unterminated character class")
        c = seg[j]
        if c == "]" and not first:
            j += 1
            break
        first = False
        if (j + 2 < len(seg) and seg[j + 1] == "-" and seg[j + 2] != "]"):
            ranges.append((c, seg[j + 2]))
            j += 3
        else:
            chars.add(c)
            j += 1

    def match(ch):
        if ch == "/":
            return False
        hit = ch in chars or any(a <= ch <= b for a, b in ranges)
        return hit != negate

    return match, j


def _seg_match(pat: str, s: str) -> bool:
    """Match one pattern segment against one path segment ('*'/'?'/classes)."""
    # dynamic programming over (pi, si)
    from functools import lru_cache

    # pre-parse pattern into token list
    toks = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "*":
            # collapse runs of *; ** inside a segment == *
            while i < len(pat) and pat[i] == "*":
                i += 1
            toks.append(("star", None))
        elif c == "?":
            toks.append(("one", None))
            i += 1
        elif c == "[":
            m, i = _parse_class(pat, i)
            toks.append(("class", m))
        else:
            toks.append(("lit", c))
            i += 1

    @lru_cache(maxsize=None)
    def go(ti, si):
        if ti == len(toks):
            return si == len(s)
        kind, val = toks[ti]
        if kind == "star":
            return any(go(ti + 1, k) for k in range(si, len(s) + 1))
        if si >= len(s):
            return False
        ch = s[si]
        if kind == "one":
            ok = ch != "/"
        elif kind == "class":
            ok = val(ch)
        else:
            ok = ch == val
        return ok and go(ti + 1, si + 1)

    return go(0, 0)


def glob_match(pattern: str, path: str) -> bool:
    psegs = pattern.split("/") if pattern != "" else []
    ssegs = path.split("/") if path != "" else []

    # validate classes in ** segments too (spec: unterminated [ raises)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def go(pi, si):
        if pi == len(psegs):
            return si == len(ssegs)
        seg = psegs[pi]
        if seg == "**":
            # zero or more whole segments; a trailing "/**" (i.e. ** as the last
            # of several segments) must consume at least one
            lo = si + 1 if (pi == len(psegs) - 1 and pi > 0) else si
            return any(go(pi + 1, k) for k in range(lo, len(ssegs) + 1))
        if si >= len(ssegs):
            return False
        return _seg_match(seg, ssegs[si]) and go(pi + 1, si + 1)

    return go(0, 0)
