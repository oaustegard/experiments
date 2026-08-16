"""Parse a single CSV record (RFC 4180 style) without using the csv module."""


def parse_csv_line(line: str) -> list[str]:
    """Parse a single CSV record into its fields.

    Rules:
    - Fields are separated by commas.
    - A field may be quoted: it starts with a double quote as its first
      character and ends with a matching double quote. Inside a quoted
      field, commas are literal, and an escaped quote is written as two
      double quotes ("" -> one literal ").
    - Quoted fields may contain any characters, including newlines.
    - The empty string parses to [""] (one empty field).
    - A double quote appearing anywhere inside an unquoted field is an
      error.
    - Any character other than a comma (or end of string) immediately
      after the closing quote of a quoted field is an error.
    - An unterminated quoted field is an error.

    Raises:
        ValueError: if the line does not conform to the rules above.
    """
    fields: list[str] = []
    i = 0
    n = len(line)

    while True:
        if i < n and line[i] == '"':
            # Quoted field.
            i += 1
            chars: list[str] = []
            while True:
                if i >= n:
                    raise ValueError("Unterminated quoted field")
                c = line[i]
                if c == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        chars.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                chars.append(c)
                i += 1
            fields.append("".join(chars))

            if i >= n:
                break
            if line[i] == ",":
                i += 1
                continue
            raise ValueError(
                "Unexpected character after closing quote of quoted field"
            )
        else:
            # Unquoted field.
            chars = []
            while i < n and line[i] != ",":
                if line[i] == '"':
                    raise ValueError('Unexpected quote in unquoted field')
                chars.append(line[i])
                i += 1
            fields.append("".join(chars))

            if i >= n:
                break
            i += 1  # Skip the comma.

    return fields
