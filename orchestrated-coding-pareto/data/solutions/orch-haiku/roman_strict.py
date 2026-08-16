def from_roman(s: str) -> int:
    """
    Parse a strictly classical Roman numeral into an integer in 1..3999.

    Args:
        s: A string representing a Roman numeral in canonical form.

    Returns:
        An integer in the range 1..3999.

    Raises:
        ValueError: If the input is not a valid canonical Roman numeral.
    """
    # Check for empty string
    if not s:
        raise ValueError("empty string")

    # Check that all characters are valid uppercase Roman numeral symbols
    valid_chars = set('IVXLCDM')
    if not all(c in valid_chars for c in s):
        raise ValueError("invalid characters or whitespace")

    # Define which symbols can be used in subtractive notation and what they can precede
    # I can go before V(5) or X(10)
    # X can go before L(50) or C(100)
    # C can go before D(500) or M(1000)
    valid_subtractive = {
        'I': {'V', 'X'},
        'X': {'L', 'C'},
        'C': {'D', 'M'}
    }

    # Symbol values
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}

    # Parse the Roman numeral left to right
    total = 0
    i = 0
    while i < len(s):
        # Check if this is a subtractive pair (current symbol < next symbol)
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            # This must be a valid subtractive pair
            if s[i] not in valid_subtractive or s[i + 1] not in valid_subtractive[s[i]]:
                raise ValueError("invalid subtractive pair")
            # Add the difference (e.g., IV = 5 - 1 = 4)
            total += values[s[i + 1]] - values[s[i]]
            i += 2
        else:
            # Regular additive case
            total += values[s[i]]
            i += 1

    # Verify that the input is in canonical form by converting the result back to Roman
    canonical = _to_roman(total)
    if canonical != s:
        raise ValueError("not in canonical form")

    # Enforce the output range bound 1..3999
    if not (1 <= total <= 3999):
        raise ValueError("result out of range")

    return total


def _to_roman(n: int) -> str:
    """
    Convert an integer to its canonical Roman numeral form.
    Uses the standard greedy algorithm with values:
    1000=M, 900=CM, 500=D, 400=CD, 100=C, 90=XC, 50=L, 40=XL, 10=X, 9=IX, 5=V, 4=IV, 1=I
    """
    values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    symbols = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

    result = ''
    for val, sym in zip(values, symbols):
        count = n // val
        if count:
            result += sym * count
            n -= val * count

    return result
