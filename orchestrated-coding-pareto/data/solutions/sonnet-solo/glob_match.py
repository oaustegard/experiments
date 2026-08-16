"""
glob_match: git/.gitignore-style glob matching against slash-separated paths.

Implements matching from scratch (no fnmatch/glob/pathlib). Supports:
  - '?'   matches exactly one character within a path segment (never '/').
  - '*'   matches zero or more characters within a path segment (never '/').
  - '**'  as a whole pattern segment matches zero or more whole path
          segments. A trailing '/**' (preceded by another segment) requires
          at least one following segment. The pattern '**' alone matches
          every path, including the empty path.
  - '**' occurring inside a segment (e.g. 'a**b') is not special and
          behaves like '*'.
  - '[...]' character classes, with '[!...]' negation, '-' ranges, a
          leading ']' treated as a literal, and a leading/trailing '-'
          treated as a literal. An unterminated '[' raises ValueError.
          A class never matches '/'.
  - Every other character matches itself literally (no escape character).

Matching is anchored: the whole pattern must match the whole path.
"""


def _parse_class(pattern, i):
    """Parse a '[...]' character class starting at pattern[i] == '['.

    Returns (token, next_index) where token is
    ('class', negate, frozenset_of_chars, tuple_of_(lo, hi)_ranges) and
    next_index is the index just past the closing ']'.

    Raises ValueError if the class is unterminated.
    """
    n = len(pattern)
    j = i + 1
    negate = False
    if j < n and pattern[j] == '!':
        negate = True
        j += 1

    chars = set()
    ranges = []
    first = True

    while True:
        if j >= n:
            raise ValueError(
                "unterminated '[' in pattern: {!r}".format(pattern)
            )
        c = pattern[j]
        if c == ']' and not first:
            break
        if c == ']' and first:
            # ']' as the first class character is a literal.
            chars.add(']')
            j += 1
            first = False
            continue
        if c == '-':
            # A '-' first or last in the class is a literal.
            chars.add('-')
            j += 1
            first = False
            continue
        if j + 2 < n and pattern[j + 1] == '-' and pattern[j + 2] != ']':
            # c '-' d : a range.
            ranges.append((c, pattern[j + 2]))
            j += 3
            first = False
            continue
        chars.add(c)
        j += 1
        first = False

    end = j  # index of the closing ']'
    token = ('class', negate, frozenset(chars), tuple(ranges))
    return token, end + 1


def _tokenize_segment(seg):
    """Tokenize one '/'-free pattern segment into a list of match tokens.

    Each token is one of:
      '?'                                   -- single-char wildcard
      '*'                                   -- zero-or-more wildcard
      ('class', negate, chars, ranges)      -- character class
      any other single character            -- literal
    """
    tokens = []
    i = 0
    n = len(seg)
    while i < n:
        c = seg[i]
        if c == '?':
            tokens.append('?')
            i += 1
        elif c == '*':
            tokens.append('*')
            i += 1
        elif c == '[':
            token, i = _parse_class(seg, i)
            tokens.append(token)
        else:
            tokens.append(c)
            i += 1
    return tokens


def _segment_matches(tokens, s):
    """Match a tokenized pattern segment against one path segment string."""
    memo = {}

    def rec(ti, si):
        key = (ti, si)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if ti == len(tokens):
            result = si == len(s)
            memo[key] = result
            return result

        tok = tokens[ti]

        if tok == '*':
            result = False
            for k in range(si, len(s) + 1):
                if rec(ti + 1, k):
                    result = True
                    break
            memo[key] = result
            return result

        if si >= len(s):
            memo[key] = False
            return False

        ch = s[si]

        if tok == '?':
            result = rec(ti + 1, si + 1)
        elif isinstance(tok, tuple) and tok[0] == 'class':
            _, negate, chars, ranges = tok
            if ch == '/':
                matched = False
            else:
                matched = ch in chars or any(
                    lo <= ch <= hi for lo, hi in ranges
                )
                if negate:
                    matched = not matched
            result = rec(ti + 1, si + 1) if matched else False
        else:
            result = rec(ti + 1, si + 1) if ch == tok else False

        memo[key] = result
        return result

    return rec(0, 0)


def glob_match(pattern: str, path: str) -> bool:
    """Match a slash-separated path against a glob pattern.

    Semantics follow git/.gitignore-style globbing; see module docstring.
    Raises ValueError if the pattern contains an unterminated '[' class.
    """
    pattern_segs = pattern.split('/') if pattern != '' else []
    path_segs = path.split('/') if path != '' else []

    # Pre-tokenize every non-'**' pattern segment. This also validates
    # character classes up front, so a malformed '[' always raises
    # ValueError regardless of whether that segment is ever reached
    # during matching.
    tokenized = []
    for seg in pattern_segs:
        if seg == '**':
            tokenized.append('**')
        else:
            tokenized.append(_tokenize_segment(seg))

    memo = {}

    def match_segs(pi, si):
        key = (pi, si)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if pi == len(tokenized):
            result = si == len(path_segs)
            memo[key] = result
            return result

        seg = tokenized[pi]

        if seg == '**':
            if pi + 1 == len(tokenized):
                # Trailing '**': the pattern '**' alone (pi == 0) matches
                # everything, including zero remaining segments. A '**'
                # preceded by another segment requires at least one
                # remaining segment (no dangling trailing slash).
                if pi == 0:
                    result = True
                else:
                    result = si < len(path_segs)
                memo[key] = result
                return result
            else:
                result = False
                for k in range(si, len(path_segs) + 1):
                    if match_segs(pi + 1, k):
                        result = True
                        break
                memo[key] = result
                return result
        else:
            if si >= len(path_segs):
                result = False
            else:
                result = (
                    _segment_matches(seg, path_segs[si])
                    and match_segs(pi + 1, si + 1)
                )
            memo[key] = result
            return result

    return match_segs(0, 0)
