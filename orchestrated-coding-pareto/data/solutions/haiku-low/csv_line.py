def parse_csv_line(line: str) -> list[str]:
    fields = []
    current_field = []
    in_quotes = False
    i = 0

    while i < len(line):
        char = line[i]

        if in_quotes:
            if char == '"':
                # Check what comes after the quote
                if i + 1 < len(line) and line[i + 1] == '"':
                    # Escaped quote
                    current_field.append('"')
                    i += 2
                else:
                    # End of quoted field
                    in_quotes = False
                    i += 1
                    # After a closing quote, next character must be comma or end
                    if i < len(line) and line[i] != ',':
                        raise ValueError("Expected comma or end of string after closing quote")
            else:
                current_field.append(char)
                i += 1
        else:
            # Not in quotes
            if char == '"':
                # Quote can only start a field if field is empty
                if len(current_field) == 0:
                    in_quotes = True
                    i += 1
                else:
                    # Quote in the middle of unquoted field
                    raise ValueError("Unexpected quote in unquoted field")
            elif char == ',':
                fields.append(''.join(current_field))
                current_field = []
                i += 1
            else:
                current_field.append(char)
                i += 1

    # Check if we ended while still in quotes
    if in_quotes:
        raise ValueError("Unterminated quoted field")

    # Add the last field
    fields.append(''.join(current_field))

    return fields
