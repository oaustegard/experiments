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
        # Remove internal spaces around the range hyphen
        token = token.replace(' ', '')

        # Try to parse as a range by finding a valid split point
        valid_splits = []
        for i in range(1, len(token)):
            left = token[:i]
            right = token[i:]
            try:
                left_val = int(left)
                right_val = int(right)
                valid_splits.append((left_val, right_val))
            except ValueError:
                pass

        if valid_splits:
            # Found a valid range
            left_val, right_val = valid_splits[0]
            if left_val > right_val:
                raise ValueError("Range A > B")
            result.extend(range(left_val, right_val + 1))
        else:
            # Try to parse as a single integer
            try:
                result.append(int(token))
            except ValueError:
                raise ValueError("Invalid token")

    # Remove duplicates and sort
    return sorted(set(result))
