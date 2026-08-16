"""Parse a single RFC 4180-style CSV record without using the `csv` module."""

__all__ = ["parse_csv_line"]


def parse_csv_line(line: str) -> list[str]:
    """Parse one CSV record into its list of fields.

    Fields are comma-separated. A field is quoted when its first character is a
    double quote; inside such a field commas and newlines are literal and a pair
    of double quotes denotes one literal double quote.

    Raises:
        ValueError: if a quote appears inside an unquoted field, if any
            character other than a comma follows the closing quote of a quoted
            field, or if a quoted field is never terminated.
    """
    fields: list[str] = []
    i = 0
    n = len(line)

    while True:
        if i < n and line[i] == '"':
            # Quoted field.
            i += 1
            buf: list[str] = []
            closed = False
            while i < n:
                ch = line[i]
                if ch == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                buf.append(ch)
                i += 1
            if not closed:
                raise ValueError(
                    "unterminated quoted field starting at index %d" % (i,)
                )
            if i < n and line[i] != ",":
                raise ValueError(
                    "unexpected character %r after closing quote at index %d"
                    % (line[i], i)
                )
            fields.append("".join(buf))
        else:
            # Unquoted field: runs to the next comma or the end of the record.
            start = i
            while i < n and line[i] != ",":
                if line[i] == '"':
                    raise ValueError(
                        "unexpected quote inside unquoted field at index %d" % (i,)
                    )
                i += 1
            fields.append(line[start:i])

        if i >= n:
            return fields
        # Skip the field separator and parse the following field.
        i += 1
