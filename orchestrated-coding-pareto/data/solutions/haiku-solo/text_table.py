def format_table(rows: list[list], headers: list[str], *,
                 aligns: list[str] | None = None,
                 max_col_width: int = 20) -> str:
    """Render an ASCII table with specified alignment and column width constraints."""

    # Validation
    if not headers:
        raise ValueError("headers cannot be empty")

    if max_col_width < 1:
        raise ValueError("max_col_width must be >= 1")

    if aligns is None:
        aligns = ["l"] * len(headers)
    else:
        if len(aligns) != len(headers):
            raise ValueError("aligns length must match headers length")
        for align in aligns:
            if align not in ("l", "r", "c"):
                raise ValueError("aligns must contain only 'l', 'r', or 'c'")

    # Check all rows have correct length
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("row length must match headers length")

    # Helper: wrap text at column width
    def wrap_text(text: str, width: int) -> list[str]:
        """Split by newlines first, then hard-wrap each line to width."""
        if not text:
            return [""]
        lines = text.split("\n")
        wrapped = []
        for line in lines:
            if not line:
                wrapped.append("")
            else:
                for i in range(0, len(line), width):
                    wrapped.append(line[i:i+width])
        return wrapped if wrapped else [""]

    # Helper: calculate display width (longest line after split by newlines)
    def display_width(text: str) -> int:
        lines = text.split("\n")
        return max((len(line) for line in lines), default=0)

    # Convert all cell values to strings
    cell_data = []
    for row in rows:
        converted = [str(cell) if cell is not None else "" for cell in row]
        cell_data.append(converted)

    # Convert headers to strings
    header_strs = [str(h) for h in headers]

    # Calculate column widths
    col_widths = []
    for col_idx in range(len(headers)):
        max_width = display_width(header_strs[col_idx])
        for row_data in cell_data:
            max_width = max(max_width, display_width(row_data[col_idx]))
        col_widths.append(min(max_width, max_col_width))

    # Build output
    result = []

    # Helper: build border
    def build_border():
        parts = []
        for width in col_widths:
            parts.append("+" + "-" * (width + 2))
        parts.append("+")
        return "".join(parts)

    # Helper: align text to width
    def align_text(text: str, width: int, align: str) -> str:
        if align == "l":
            return text.ljust(width)
        elif align == "r":
            return text.rjust(width)
        else:  # "c"
            total_padding = width - len(text)
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            return " " * left_padding + text + " " * right_padding

    # Top border
    result.append(build_border())

    # Header rows (with wrapping)
    header_wrapped = []
    max_header_lines = 0
    for col_idx, header in enumerate(header_strs):
        lines = wrap_text(header, col_widths[col_idx])
        header_wrapped.append(lines)
        max_header_lines = max(max_header_lines, len(lines))

    for line_idx in range(max_header_lines):
        row_parts = []
        for col_idx in range(len(headers)):
            lines = header_wrapped[col_idx]
            text = lines[line_idx] if line_idx < len(lines) else ""
            aligned = align_text(text, col_widths[col_idx], aligns[col_idx])
            row_parts.append("| " + aligned + " ")
        row_parts.append("|")
        result.append("".join(row_parts))

    # Header separator
    result.append(build_border())

    # Data rows
    for row_data in cell_data:
        # Wrap each cell
        wrapped_cells = []
        max_lines = 0
        for col_idx, cell_text in enumerate(row_data):
            lines = wrap_text(cell_text, col_widths[col_idx])
            wrapped_cells.append(lines)
            max_lines = max(max_lines, len(lines))

        # Output physical lines for this row
        for line_idx in range(max_lines):
            row_parts = []
            for col_idx in range(len(headers)):
                lines = wrapped_cells[col_idx]
                text = lines[line_idx] if line_idx < len(lines) else ""
                aligned = align_text(text, col_widths[col_idx], aligns[col_idx])
                row_parts.append("| " + aligned + " ")
            row_parts.append("|")
            result.append("".join(row_parts))

    # Bottom border
    result.append(build_border())

    return "\n".join(result)
