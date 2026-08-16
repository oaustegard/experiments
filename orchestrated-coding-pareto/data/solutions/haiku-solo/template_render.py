def render(template: str, values: dict) -> str:
    result = []
    i = 0
    missing_keys = set()

    while i < len(template):
        if template[i] == '{':
            # Check for {{ escape
            if i + 1 < len(template) and template[i + 1] == '{':
                result.append('{')
                i += 2
            else:
                # Try to parse placeholder {name}
                j = i + 1

                # Check if we can read a name
                if j >= len(template):
                    raise ValueError("Unclosed placeholder")

                # First char of name must be letter or underscore
                if not (template[j].isalpha() or template[j] == '_'):
                    raise ValueError("Invalid placeholder")

                # Scan while valid name characters
                name_end = j
                while name_end < len(template) and (template[name_end].isalnum() or template[name_end] == '_'):
                    name_end += 1

                # Must be followed by }
                if name_end >= len(template) or template[name_end] != '}':
                    raise ValueError("Invalid placeholder")

                name = template[j:name_end]

                # Track value or missing key
                if name in values:
                    result.append(str(values[name]))
                else:
                    missing_keys.add(name)

                i = name_end + 1
        elif template[i] == '}':
            # Check for }} escape
            if i + 1 < len(template) and template[i + 1] == '}':
                result.append('}')
                i += 2
            else:
                raise ValueError("Unmatched closing brace")
        else:
            result.append(template[i])
            i += 1

    # Raise KeyError if any keys are missing
    if missing_keys:
        sorted_keys = sorted(missing_keys)
        raise KeyError("missing keys: " + ", ".join(sorted_keys))

    return ''.join(result)
