def render(template: str, values: dict) -> str:
    """
    Render a template with placeholder substitution.

    Placeholders are {name} where name matches [A-Za-z_][A-Za-z0-9_]*.
    {{ renders as {, }} renders as }.
    """
    # First pass: validate template and collect placeholder names
    placeholders = set()
    i = 0

    while i < len(template):
        c = template[i]

        if c == '{':
            if i + 1 < len(template) and template[i + 1] == '{':
                # Escape: {{
                i += 2
            else:
                # Potential placeholder
                j = i + 1
                if j < len(template) and (template[j].isalpha() or template[j] == '_'):
                    # Read identifier
                    ident_start = j
                    while j < len(template) and (template[j].isalnum() or template[j] == '_'):
                        j += 1

                    if j < len(template) and template[j] == '}':
                        # Valid placeholder
                        name = template[ident_start:j]
                        placeholders.add(name)
                        i = j + 1
                    else:
                        # Unclosed placeholder
                        raise ValueError(f"Invalid placeholder at position {i}")
                else:
                    # Invalid placeholder start
                    raise ValueError(f"Invalid placeholder at position {i}")

        elif c == '}':
            if i + 1 < len(template) and template[i + 1] == '}':
                # Escape: }}
                i += 2
            else:
                # Unmatched }
                raise ValueError(f"Unmatched '}}' at position {i}")

        else:
            i += 1

    # Check for missing keys
    missing_keys = placeholders - set(values.keys())
    if missing_keys:
        sorted_missing = sorted(missing_keys)
        raise KeyError("missing keys: " + ", ".join(sorted_missing))

    # Second pass: render
    result = []
    i = 0

    while i < len(template):
        c = template[i]

        if c == '{':
            if i + 1 < len(template) and template[i + 1] == '{':
                # Escape: {{
                result.append('{')
                i += 2
            else:
                # Placeholder
                j = i + 1
                ident_start = j
                while j < len(template) and (template[j].isalnum() or template[j] == '_'):
                    j += 1

                name = template[ident_start:j]
                result.append(str(values[name]))
                i = j + 1

        elif c == '}':
            if i + 1 < len(template) and template[i + 1] == '}':
                # Escape: }}
                result.append('}')
                i += 2
            else:
                # This should not happen if first pass succeeded
                raise ValueError(f"Unmatched '}}' at position {i}")

        else:
            result.append(c)
            i += 1

    return ''.join(result)
