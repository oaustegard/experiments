"""Minimal placeholder-substitution engine.

Public API: ``render(template, values)``.
"""

import re

__all__ = ["render"]

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def render(template: str, values: dict) -> str:
    """Render ``template``, substituting ``{name}`` placeholders from ``values``.

    - ``{{`` renders a literal ``{``; ``}}`` renders a literal ``}``.
    - Values are converted with ``str()``.
    - Malformed braces raise ``ValueError``.
    - Missing placeholder names raise ``KeyError`` listing all of them.
    """
    out = []
    missing = set()
    i = 0
    n = len(template)

    while i < n:
        ch = template[i]

        if ch == "{":
            # Escape: "{{" -> literal "{"
            if i + 1 < n and template[i + 1] == "{":
                out.append("{")
                i += 2
                continue

            match = _NAME_RE.match(template, i + 1)
            if match is None:
                raise ValueError(
                    "invalid placeholder at index %d: expected a name matching "
                    "[A-Za-z_][A-Za-z0-9_]* after '{'" % i
                )

            end = match.end()
            if end >= n or template[end] != "}":
                raise ValueError(
                    "unclosed placeholder starting at index %d: expected '}'" % i
                )

            name = match.group(0)
            if name in values:
                out.append(str(values[name]))
            else:
                missing.add(name)

            i = end + 1
            continue

        if ch == "}":
            # Escape: "}}" -> literal "}"
            if i + 1 < n and template[i + 1] == "}":
                out.append("}")
                i += 2
                continue
            raise ValueError("unmatched '}' at index %d" % i)

        out.append(ch)
        i += 1

    if missing:
        raise KeyError("missing keys: " + ", ".join(sorted(missing)))

    return "".join(out)
