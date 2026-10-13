"""C11 Annex D.1 ranges of characters allowed in identifiers (not as the first
character). gcc accepts these in identifiers in every C11+ mode (e.g. `a¾`);
they are added to the XID_Continue test used by the lexer."""

_C11_D1 = (
    (0xA8, 0xA8), (0xAA, 0xAA), (0xAD, 0xAD), (0xAF, 0xAF), (0xB2, 0xB5),
    (0xB7, 0xBA), (0xBC, 0xBE), (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0xFF),
    (0x100, 0x167F), (0x1681, 0x180D), (0x180F, 0x1FFF),
    (0x200B, 0x200D), (0x202A, 0x202E), (0x203F, 0x2040), (0x2054, 0x2054),
    (0x2060, 0x206F), (0x2070, 0x218F), (0x2460, 0x24FF), (0x2776, 0x2793),
    (0x2C00, 0x2DFF), (0x2E80, 0x2FFF), (0x3004, 0x3007), (0x3021, 0x302F),
    (0x3031, 0x303F), (0x3040, 0xD7FF), (0xF900, 0xFD3D), (0xFD40, 0xFDCF),
    (0xFDF0, 0xFE44), (0xFE47, 0xFFFD),
) + tuple((base, base + 0xFFFD)
          for base in range(0x10000, 0xE0000, 0x10000)) + ((0xE0000, 0xEFFFD),)


# C11 Annex D.2: D.1 characters that may not begin an identifier (combining marks)
_C11_D2 = ((0x300, 0x36F), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F))


def c11_identifier_start(c: str) -> bool:
    """True if c may begin an identifier under C11 Annex D (D.1 minus D.2). gcc 13 accepts
    these as the first character before C23 (`int ² = 42;`, `int 😀x;` in -std=c99..gnu17)."""
    o = ord(c)
    return in_c11_identifier_range(c) and not any(lo <= o <= hi for lo, hi in _C11_D2)


def in_c11_identifier_range(c: str) -> bool:
    """True if character c may appear in an identifier after the first character (C11 D.1)."""
    o = ord(c)
    for lo, hi in _C11_D1:
        if lo <= o <= hi:
            return True
    return False
