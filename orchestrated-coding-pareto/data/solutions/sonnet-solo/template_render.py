import re

_NAME_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def render(template: str, values: dict) -> str:
    tokens = []
    i = 0
    n = len(template)

    while i < n:
        c = template[i]

        if c == '{':
            if i + 1 < n and template[i + 1] == '{':
                tokens.append(('lit', '{'))
                i += 2
                continue

            m = _NAME_RE.match(template, i + 1)
            if m is None:
                raise ValueError(
                    "invalid '{' at position %d in template" % i
                )
            end = m.end()
            if end >= n or template[end] != '}':
                raise ValueError(
                    "invalid '{' at position %d in template" % i
                )
            tokens.append(('ph', m.group(0)))
            i = end + 1
            continue

        if c == '}':
            if i + 1 < n and template[i + 1] == '}':
                tokens.append(('lit', '}'))
                i += 2
                continue
            raise ValueError(
                "invalid '}' at position %d in template" % i
            )

        tokens.append(('lit', c))
        i += 1

    missing = set()
    for kind, val in tokens:
        if kind == 'ph' and val not in values:
            missing.add(val)

    if missing:
        raise KeyError("missing keys: " + ", ".join(sorted(missing)))

    parts = []
    for kind, val in tokens:
        if kind == 'lit':
            parts.append(val)
        else:
            parts.append(str(values[val]))

    return ''.join(parts)
