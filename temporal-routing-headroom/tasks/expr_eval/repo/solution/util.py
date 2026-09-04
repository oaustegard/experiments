import re


_TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+\.?|\d*\.\d+)|([+\-*/%()]|\*\*))")


def _tokenize(s: str):
    tokens = []
    i = 0
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m:
            if s[i:].strip() == "":
                break
            raise ValueError(f"bad character at {i}: {s[i]!r}")
        num, op = m.group(1), m.group(2)
        if num is not None:
            # reject things like "1.2.3" -- next char must not be a dot/digit
            j = m.end()
            if j < len(s) and s[j] == ".":
                raise ValueError(f"malformed number near {num!r}")
            tokens.append(("num", float(num)))
        else:
            tokens.append(("op", op))
        i = m.end()
    return tokens
