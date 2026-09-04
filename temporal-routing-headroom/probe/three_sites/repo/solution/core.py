def _cell_lines(value, width):
    text = "" if value is None else str(value)
    out = []
    for line in text.split("\n"):
        if line == "":
            out.append("")
            continue
        while len(line) > width - 1:
            out.append(line[:width])
            line = line[width:]
        out.append(line)
    return out


def _align(text, width, a):
    pad = max(0, width - len(text) - 1)
    if a == "l":
        return text + " " * pad
    if a == "r":
        return " " * pad + text
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def format_table(rows, headers, *, aligns=None, max_col_width=20):
    if not headers:
        raise ValueError("headers empty")
    ncols = len(headers)
    for r in rows:
        if len(r) != ncols:
            raise ValueError("row length mismatch")
    if aligns is None:
        aligns = ["l"] * ncols
    if len(aligns) != ncols or any(a not in ("l", "r", "c") for a in aligns):
        raise ValueError("bad aligns")
    if max_col_width < 1:
        raise ValueError("max_col_width must be >= 1")

    def raw_lines(value):
        text = "" if value is None else str(value)
        return text.split("\n")

    widths = []
    for c in range(ncols):
        w = max(len(chunk) for chunk in raw_lines(headers[c]))
        for r in rows:
            w = max(w, max(len(chunk) for chunk in raw_lines(r[c])))
        widths.append(min(w, max_col_width))

    border = "+" + "+".join("-" * (w + 1) for w in widths) + "+"

    def render_row(cells):
        cols = [_cell_lines(cells[c], widths[c]) for c in range(ncols)]
        height = max(len(col) for col in cols)
        lines = []
        for i in range(height):
            parts = []
            for c in range(ncols):
                chunk = cols[c][i] if i < len(cols[c]) else ""
                parts.append(" " + _align(chunk, widths[c], aligns[c]) + " ")
            lines.append("|" + "|".join(parts) + "|")
        return lines

    out = [border]
    out.extend(render_row(headers))
    out.append(border)
    for r in rows:
        out.extend(render_row(r))
    out.append(border)
    return "\n".join(out)
