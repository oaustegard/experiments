def parse_range(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []

    tokens = s.split(',')
    result = []

    for token in tokens:
        token = token.strip()
        if not token:
            raise ValueError("empty token")

        # Try single integer
        try:
            result.append(int(token))
            continue
        except ValueError:
            pass

        # Try range
        if token[0] == '-':
            sep_idx = token.find('-', 1)
        else:
            sep_idx = token.find('-')

        if sep_idx == -1:
            raise ValueError("invalid token")

        a = token[:sep_idx].strip()
        b = token[sep_idx+1:].strip()

        try:
            a_int = int(a)
            b_int = int(b)
        except ValueError:
            raise ValueError("invalid token")

        if a_int > b_int:
            raise ValueError("invalid range")

        result.extend(range(a_int, b_int + 1))

    return sorted(list(set(result)))
