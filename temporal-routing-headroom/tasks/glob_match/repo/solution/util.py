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
        hit = ch in chars
        return hit != negate

    return match, j
