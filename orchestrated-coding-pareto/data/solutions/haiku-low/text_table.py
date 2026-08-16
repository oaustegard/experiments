def format_table(rows: list[list], headers: list[str], *,
                 aligns: list[str] | None = None,
                 max_col_width: int = 20) -> str:
    # Validation
    if not headers:
        raise ValueError("headers cannot be empty")

    if max_col_width < 1:
        raise ValueError("max_col_width must be >= 1")

    for row in rows:
        if len(row) != len(headers):
            raise ValueError("row length mismatch")

    if aligns is None:
        aligns = ["l"] * len(headers)
    elif len(aligns) != len(headers):
        raise ValueError("aligns length mismatch")

    for align in aligns:
        if align not in ("l", "r", "c"):
            raise ValueError("invalid alignment")

    num_cols = len(headers)

    # Split cells by newlines
    header_cell_lines = [h.split('\n') for h in headers]
    data_cell_lines = []
    for row in rows:
        row_cell_lines = []
        for cell in row:
            cell_str = "" if cell is None else str(cell)
            row_cell_lines.append(cell_str.split('\n'))
        data_cell_lines.append(row_cell_lines)

    # Calculate column widths
    col_widths = []
    for col_idx in range(num_cols):
        max_width = 0
        for line in header_cell_lines[col_idx]:
            max_width = max(max_width, len(line))
        for row_cell_lines in data_cell_lines:
            for line in row_cell_lines[col_idx]:
                max_width = max(max_width, len(line))
        col_widths.append(min(max_width, max_col_width))

    # Helper to hard-wrap and align a single line
    def wrap_and_align(line, col_width, align):
        chunks = []
        for i in range(0, len(line), col_width):
            chunks.append(line[i:i+col_width])
        if not chunks:
            chunks = [""]

        aligned = []
        for chunk in chunks:
            if align == "l":
                aligned.append(chunk.ljust(col_width))
            elif align == "r":
                aligned.append(chunk.rjust(col_width))
            else:  # "c"
                left_pad = (col_width - len(chunk)) // 2
                right_pad = col_width - len(chunk) - left_pad
                aligned.append(" " * left_pad + chunk + " " * right_pad)
        return aligned

    # Wrap header cells
    header_wrapped = []
    max_header_height = 0
    for col_idx in range(num_cols):
        col_wrapped = []
        for line in header_cell_lines[col_idx]:
            col_wrapped.extend(wrap_and_align(line, col_widths[col_idx], aligns[col_idx]))
        header_wrapped.append(col_wrapped)
        max_header_height = max(max_header_height, len(col_wrapped))

    # Pad header columns
    for col_idx in range(num_cols):
        while len(header_wrapped[col_idx]) < max_header_height:
            header_wrapped[col_idx].append(" " * col_widths[col_idx])

    # Wrap data cells
    data_wrapped = []
    for row_cell_lines in data_cell_lines:
        row_wrapped = []
        max_row_height = 0
        for col_idx in range(num_cols):
            col_wrapped = []
            for line in row_cell_lines[col_idx]:
                col_wrapped.extend(wrap_and_align(line, col_widths[col_idx], aligns[col_idx]))
            row_wrapped.append(col_wrapped)
            max_row_height = max(max_row_height, len(col_wrapped))

        # Pad columns in this row
        for col_idx in range(num_cols):
            while len(row_wrapped[col_idx]) < max_row_height:
                row_wrapped[col_idx].append(" " * col_widths[col_idx])

        data_wrapped.append(row_wrapped)

    # Build output
    result = []

    # Border
    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    result.append(border)

    # Header rows
    for line_idx in range(max_header_height):
        cells = [" " + header_wrapped[col_idx][line_idx] + " " for col_idx in range(num_cols)]
        result.append("|" + "|".join(cells) + "|")

    # Header separator
    result.append(border)

    # Data rows
    for row_wrapped_cols in data_wrapped:
        for line_idx in range(len(row_wrapped_cols[0])):
            cells = [" " + row_wrapped_cols[col_idx][line_idx] + " " for col_idx in range(num_cols)]
            result.append("|" + "|".join(cells) + "|")

    # Bottom border
    result.append(border)

    return "\n".join(result)
