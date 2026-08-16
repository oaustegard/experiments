def format_table(rows: list[list], headers: list[str], *,
                  aligns: list[str] | None = None,
                  max_col_width: int = 20) -> str:
    if not headers:
        raise ValueError("headers must not be empty")

    n_cols = len(headers)

    for row in rows:
        if len(row) != n_cols:
            raise ValueError("row length does not match number of headers")

    if aligns is not None:
        if len(aligns) != n_cols:
            raise ValueError("aligns length does not match number of headers")
        for a in aligns:
            if a not in ("l", "r", "c"):
                raise ValueError("aligns must contain only 'l', 'r', or 'c'")
    else:
        aligns = ["l"] * n_cols

    if max_col_width < 1:
        raise ValueError("max_col_width must be >= 1")

    def to_text(value) -> str:
        if value is None:
            return ""
        return str(value)

    def raw_lines(text: str) -> list[str]:
        return text.split("\n")

    def max_line_len(text: str) -> int:
        lines = raw_lines(text)
        if not lines:
            return 0
        return max(len(line) for line in lines)

    def wrap_cell(text: str, width: int) -> list[str]:
        chunks: list[str] = []
        for line in raw_lines(text):
            if len(line) == 0:
                chunks.append("")
            else:
                for i in range(0, len(line), width):
                    chunks.append(line[i:i + width])
        return chunks

    def align_text(s: str, width: int, align: str) -> str:
        if align == "l":
            return s.ljust(width)
        if align == "r":
            return s.rjust(width)
        # center
        total_pad = width - len(s)
        if total_pad <= 0:
            return s
        left = total_pad // 2
        right = total_pad - left
        return (" " * left) + s + (" " * right)

    header_texts = [to_text(h) for h in headers]
    row_texts = [[to_text(v) for v in row] for row in rows]

    col_widths = []
    for j in range(n_cols):
        widest = max_line_len(header_texts[j])
        for row in row_texts:
            widest = max(widest, max_line_len(row[j]))
        col_widths.append(min(widest, max_col_width))

    def build_physical_row(texts: list[str]) -> list[list[str]]:
        col_chunks = [wrap_cell(texts[j], col_widths[j]) for j in range(n_cols)]
        height = max((len(c) for c in col_chunks), default=1)
        height = max(height, 1)
        for c in col_chunks:
            while len(c) < height:
                c.append("")
        return col_chunks

    def render_physical_row(col_chunks: list[list[str]]) -> list[str]:
        height = len(col_chunks[0]) if col_chunks else 0
        lines = []
        for line_idx in range(height):
            cells = []
            for j in range(n_cols):
                content = align_text(col_chunks[j][line_idx], col_widths[j], aligns[j])
                cells.append(" " + content + " ")
            lines.append("|" + "|".join(cells) + "|")
        return lines

    def border_line() -> str:
        return "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    output_lines: list[str] = []
    output_lines.append(border_line())

    header_chunks = build_physical_row(header_texts)
    output_lines.extend(render_physical_row(header_chunks))

    output_lines.append(border_line())

    for row in row_texts:
        row_chunks = build_physical_row(row)
        output_lines.extend(render_physical_row(row_chunks))

    output_lines.append(border_line())

    return "\n".join(output_lines)
