"""Render ASCII tables with exact, deterministic layout rules."""

__all__ = ["format_table"]

_VALID_ALIGNS = ("l", "r", "c")


def _cell_text(value):
    """Convert a raw cell value to its text form."""
    if value is None:
        return ""
    return str(value)


def _split_lines(text):
    """Split a cell's text on explicit newlines."""
    return text.split("\n")


def _hard_wrap(line, width):
    """Hard-break a single line into chunks of exactly `width` characters."""
    if width <= 0:
        return [""]
    if len(line) <= width:
        return [line]
    return [line[i:i + width] for i in range(0, len(line), width)]


def _wrap_cell(text, width):
    """Split on newlines, then hard-wrap each resulting line."""
    chunks = []
    for line in _split_lines(text):
        chunks.extend(_hard_wrap(line, width))
    return chunks or [""]


def _align(chunk, width, how):
    """Pad `chunk` to `width` according to alignment `how`."""
    pad = width - len(chunk)
    if pad <= 0:
        return chunk
    if how == "r":
        return " " * pad + chunk
    if how == "c":
        left = pad // 2
        right = pad - left
        return " " * left + chunk + " " * right
    return chunk + " " * pad


def _render_row(cells, widths, aligns):
    """Render one logical row (list of text cells) as physical lines."""
    wrapped = [_wrap_cell(text, widths[i]) for i, text in enumerate(cells)]
    height = max(len(chunks) for chunks in wrapped)
    lines = []
    for row_line in range(height):
        parts = []
        for col, chunks in enumerate(wrapped):
            chunk = chunks[row_line] if row_line < len(chunks) else ""
            parts.append(" " + _align(chunk, widths[col], aligns[col]) + " ")
        lines.append("|" + "|".join(parts) + "|")
    return lines


def _border(widths):
    return "+" + "+".join("-" * (w + 2) for w in widths) + "+"


def format_table(rows, headers, *, aligns=None, max_col_width=20):
    """Render `rows` as an ASCII table under `headers`.

    Cells are stringified (``None`` becomes ""), hard-wrapped to the column
    width, and padded with one space on each side. Column width is the larger
    of the header width and the widest cell line, capped at `max_col_width`.
    """
    if not headers:
        raise ValueError("headers must not be empty")

    ncols = len(headers)

    if not isinstance(max_col_width, int) or isinstance(max_col_width, bool):
        max_col_width = int(max_col_width)
    if max_col_width < 1:
        raise ValueError("max_col_width must be at least 1")

    if aligns is None:
        aligns = ["l"] * ncols
    else:
        aligns = list(aligns)
        if len(aligns) != ncols:
            raise ValueError("aligns must have one entry per column")
        for a in aligns:
            if a not in _VALID_ALIGNS:
                raise ValueError("aligns entries must be 'l', 'r' or 'c'")

    header_texts = [_cell_text(h) for h in headers]

    body = []
    for row in rows:
        row = list(row)
        if len(row) != ncols:
            raise ValueError("every row must have exactly len(headers) cells")
        body.append([_cell_text(c) for c in row])

    widths = []
    for col in range(ncols):
        widest = 0
        for line in _split_lines(header_texts[col]):
            widest = max(widest, len(line))
        for row in body:
            for line in _split_lines(row[col]):
                widest = max(widest, len(line))
        widths.append(min(widest, max_col_width))

    out = [_border(widths)]
    out.extend(_render_row(header_texts, widths, aligns))
    out.append(_border(widths))
    for row in body:
        out.extend(_render_row(row, widths, aligns))
    out.append(_border(widths))

    return "\n".join(out)
