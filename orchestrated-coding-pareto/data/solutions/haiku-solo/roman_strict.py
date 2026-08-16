def from_roman(s: str) -> int:
    """Parse a strictly classical Roman numeral into an integer in 1..3999."""

    # Check for empty string
    if not s:
        raise ValueError("empty string")

    # Check for whitespace
    if any(c.isspace() for c in s):
        raise ValueError("whitespace in string")

    # Check for invalid characters (also catches lowercase)
    roman_chars = set('IVXLCDM')
    for c in s:
        if c not in roman_chars:
            raise ValueError("invalid character")

    # Value map for each Roman numeral character
    value_map = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }

    # Parse the Roman numeral using right-to-left algorithm
    total = 0
    prev_value = 0

    for i in range(len(s) - 1, -1, -1):
        current_value = value_map[s[i]]

        if current_value < prev_value:
            total -= current_value
        else:
            total += current_value

        prev_value = current_value

    # Validate range
    if total < 1 or total > 3999:
        raise ValueError("value out of range")

    # Helper function to convert integer to canonical Roman numeral
    def to_roman(num):
        val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        roman_num = ''
        for i in range(len(val)):
            count = num // val[i]
            roman_num += syms[i] * count
            num -= val[i] * count
        return roman_num

    # Validate that the input is in canonical form
    if to_roman(total) != s:
        raise ValueError("non-canonical form")

    return total
