import re


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

        # Try to parse as a range first using a grammar-based approach
        # Pattern: optional sign + digits, optional whitespace, hyphen, optional whitespace, optional sign + digits
        range_match = re.match(r'^(-?\d+)\s*-\s*(-?\d+)$', token)
        if range_match:
            left_val = int(range_match.group(1))
            right_val = int(range_match.group(2))
            if left_val > right_val:
                raise ValueError("Range A > B")
            result.extend(range(left_val, right_val + 1))
        else:
            # Try to parse as a single integer
            # Validate it matches the strict pattern: optional sign + digits
            if re.match(r'^-?\d+$', token):
                result.append(int(token))
            else:
                raise ValueError("Invalid token")

    # Remove duplicates and sort
    return sorted(set(result))
