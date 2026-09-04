import re
from .util import _NAME


def render(template: str, values: dict) -> str:
    parts = []
    missing = set()
    i = 0
    n = len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            if i + 1 < n and template[i + 1] == "{":
                parts.append("{")
                i += 2
                continue
            m = _NAME.match(template, i + 1)
            if not m or m.end() >= n or template[m.end()] != "}":
                raise ValueError(f"malformed placeholder at index {i}")
            name = m.group(0)
            if name in values:
                parts.append(str(values[name]))
            else:
                missing.add(name)
            i = m.end() + 1
        elif ch == "}":
            if i + 1 < n and template[i + 1] == "}":
                parts.append("}")
                i += 2
                continue
            raise ValueError(f"stray '}}' at index {i}")
        else:
            parts.append(ch)
            i += 1
    if missing:
        raise KeyError("missing keys: " + ", ".join(sorted(missing)))
    return "".join(parts)
