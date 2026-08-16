def parse_range(s: str) -> list[int]:
    s = s.strip()
    if not s:
        return []

    tokens = s.split(',')

    # Check for empty tokens (leading/trailing/double commas)
    for token in tokens:
        if not token.strip():
            raise ValueError("Empty token")

    result = []

    for token in tokens:
        token = token.strip()
        # Remove internal spaces
        token = token.replace(' ', '')

        # Try to parse as a range by finding a hyphen separator
        found_range = False

        if '-' in token:
            hyphen_positions = [i for i, c in enumerate(token) if c == '-']

            for pos in hyphen_positions:
                left = token[:pos]
                right = token[pos+1:]

                # Both sides must be non-empty
                if not left or not right:
                    continue

                # Try to parse both sides as integers
                try:
                    left_val = int(left)
                    right_val = int(right)
                except ValueError:
                    # This hyphen doesn't work as a range separator
                    continue

                # Both sides parsed successfully
                if left_val > right_val:
                    raise ValueError("Range A > B")

                result.extend(range(left_val, right_val + 1))
                found_range = True
                break

        if not found_range:
            # Try to parse as a single integer
            try:
                result.append(int(token))
            except ValueError:
                raise ValueError("Invalid token")

    # Remove duplicates and sort
    return sorted(set(result))
