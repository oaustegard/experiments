def parse_csv_line(line: str) -> list[str]:
    """Parse a single CSV record (RFC 4180 style) into its fields."""
    if len(line) == 0:
        return [""]

    fields = []
    i = 0

    while True:
        # Check if this is a quoted field
        if i < len(line) and line[i] == '"':
            # Parse quoted field
            field = ""
            i += 1  # Move past opening quote

            while i < len(line):
                if line[i] == '"':
                    # Check for escaped quote or end of field
                    if i + 1 < len(line) and line[i + 1] == '"':
                        # Escaped quote (two quotes become one)
                        field += '"'
                        i += 2
                    else:
                        # End of quoted field
                        i += 1  # Move past closing quote

                        # Check what comes next (must be comma or end of string)
                        if i < len(line) and line[i] != ',':
                            raise ValueError("Character after closing quote is not comma or end of string")

                        fields.append(field)

                        # Move past comma if present
                        if i < len(line) and line[i] == ',':
                            i += 1
                        else:
                            # End of line after quoted field
                            return fields

                        break
                else:
                    field += line[i]
                    i += 1
            else:
                # Reached end of string without closing quote (while loop exited naturally)
                raise ValueError("Unterminated quoted field")

        elif i < len(line):
            # Parse unquoted field
            field = ""

            while i < len(line) and line[i] != ',':
                if line[i] == '"':
                    raise ValueError("Quote in unquoted field")
                field += line[i]
                i += 1

            fields.append(field)

            # Check if there's a comma
            if i < len(line) and line[i] == ',':
                i += 1
            else:
                # End of line after unquoted field
                return fields

        else:
            # i >= len(line), we've reached the end
            # This happens when the previous field ended with a comma
            fields.append("")
            return fields
