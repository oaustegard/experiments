"""Literal decoding and final token conversion (between preprocess and parse).

Public API:
    convert_tokens(toks) -> list[Token]   mark keywords (canonicalizing GNU
        aliases like __inline__ -> inline), set value/ty on NUMBER/CHAR/STRING,
        concatenate adjacent string literals.
    parse_int_literal(tok) -> (int, Type)
    parse_number(tok) -> (int|float, Type)
    parse_char_literal(tok) -> (int, Type)
    parse_string_literal(tok) -> (list[int] code units, elem Type)
"""
from __future__ import annotations

from hcc import diag
from hcc.tokens import KEYWORDS, Token, TokenKind
from hcc.halfbits import round_to_half
from hcc.typesys import (TY_BOOL, TY_CHAR, TY_UCHAR, TY_DOUBLE, TY_FLOAT, TY_FLOAT16, TY_FLOAT128,
                       TY_INT, TY_LDOUBLE, TY_LLONG, TY_LONG, TY_UINT, TY_ULLONG, TY_ULONG,
                       TY_USHORT, Type, TypeKind, array_of)

KEYWORD_ALIASES = {
    "__restrict": "restrict", "__restrict__": "restrict",
    "__inline": "inline", "__inline__": "inline",
    "__const": "const", "__const__": "const",
    "__volatile": "volatile", "__volatile__": "volatile",
    "__signed": "signed", "__signed__": "signed",
    "__alignof": "_Alignof", "__alignof__": "_Alignof",
    "__typeof": "typeof", "__typeof__": "typeof",
    "__asm": "asm", "__asm__": "asm",
    "__attribute": "__attribute__",
    "__thread": "_Thread_local",
    "__complex__": "_Complex", "__complex": "_Complex",
    "__real": "__real__", "__imag": "__imag__",
    "__int128_t": "__int128",
}

_ESCAPES = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11, "e": 27, "E": 27,
            "\\": 92, "'": 39, '"': 34, "?": 63}


def parse_int_literal(tok: Token) -> tuple[int, Type]:
    s = tok.text.lower()
    suffix_start = len(s)
    if s.startswith(("0x", "0b")):
        digits_ok = "0123456789abcdef" if s[1] == "x" else "01"
        i = 2
        while i < len(s) and s[i] in digits_ok:
            i += 1
        base = 16 if s[1] == "x" else 2
        digits = s[2:i]
    else:
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        digits = s[:i]
        base = 8 if digits.startswith("0") and len(digits) > 1 else 10
    suffix_start = i
    suffix = s[suffix_start:]
    if suffix not in ("", "u", "l", "ul", "lu", "ll", "ull", "llu"):
        diag.error(tok, f"invalid integer constant '{tok.text}'")
    try:
        val = int(digits, base) if digits else 0
    except ValueError:
        diag.error(tok, f"invalid integer constant '{tok.text}'")
    is_u = "u" in suffix
    nl = suffix.count("l")
    # C11 6.4.4.1: first type in the list that can hold the value
    if base == 10:
        cands = {0: [TY_INT, TY_LONG], 1: [TY_LONG], 2: [TY_LLONG]}[nl]
    else:
        cands = {0: [TY_INT, TY_UINT, TY_LONG, TY_ULONG], 1: [TY_LONG, TY_ULONG],
                 2: [TY_LLONG, TY_ULLONG]}[nl]
    if is_u:
        cands = {0: [TY_UINT, TY_ULONG], 1: [TY_ULONG], 2: [TY_ULLONG]}[nl]
    for t in cands:
        limit = 1 << (t.size * 8 - (0 if t.is_unsigned else 1))
        if val < limit:
            return val, t
    ty = TY_ULLONG if nl == 2 else TY_ULONG
    return val & ((1 << 64) - 1), ty


# floating suffixes, longest first; the _FloatN/_FloatNx forms are GCC's TS 18661 suffixes
_FLOAT_SUFFIXES = (("f16", TY_FLOAT16), ("f32x", TY_DOUBLE), ("f32", TY_FLOAT),
                   ("f64x", TY_LDOUBLE), ("f64", TY_DOUBLE), ("f128x", TY_FLOAT128),
                   ("f128", TY_FLOAT128), ("f", TY_FLOAT), ("l", TY_LDOUBLE))


def parse_number(tok: Token):
    s = tok.text
    low = s.lower()
    is_hex = low.startswith("0x")
    if not is_hex and ("." in s or "e" in low) or is_hex and ("." in s or "p" in low):
        ty, body = TY_DOUBLE, s
        for suf, suf_ty in _FLOAT_SUFFIXES:
            if low.endswith(suf):
                ty, body = suf_ty, s[: -len(suf)]
                break
        try:
            val = float.fromhex(body) if is_hex else float(body)
        except ValueError:
            diag.error(tok, f"invalid floating constant '{s}'")
        if ty.kind == TypeKind.FLOAT16:
            val = round_to_half(val)
        return val, ty
    return parse_int_literal(tok)


def _decode(tok: Token, body: str, wide: bool) -> list[int]:
    """Decode escapes; returns code units (bytes for narrow, code points for wide)."""
    out: list[int] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\":
            j = i
            while j < n and body[j] != "\\":
                j += 1
            chunk = body[i:j]
            if wide:
                out.extend(ord(ch) for ch in chunk)
            else:
                out.extend(chunk.encode("utf-8", errors="surrogateescape"))
            i = j
            continue
        i += 1
        if i >= n:
            diag.error(tok, "bad escape at end of literal")
        c = body[i]
        if c in _ESCAPES:
            out.append(_ESCAPES[c])
            i += 1
        elif c in "01234567":
            j = i
            while j < n and j < i + 3 and body[j] in "01234567":
                j += 1
            out.append(int(body[i:j], 8) & (0xFFFFFFFF if wide else 0xFF))
            i = j
        elif c == "x":
            j = i + 1
            while j < n and body[j] in "0123456789abcdefABCDEF":
                j += 1
            if j == i + 1:
                diag.error(tok, "\\x used with no following hex digits")
            out.append(int(body[i + 1:j], 16) & (0xFFFFFFFF if wide else 0xFF))
            i = j
        elif c in "uU":
            k = 4 if c == "u" else 8
            cp = int(body[i + 1:i + 1 + k], 16)
            i += 1 + k
            if wide:
                out.append(cp)
            else:
                out.extend(chr(cp).encode("utf-8", errors="surrogatepass"))
        else:
            out.append(ord(c))     # unknown escape: gcc warns, keeps char
            i += 1
    return out


def _prefix(text: str) -> tuple[str, str]:
    q = text.index(text[-1])
    return text[:q], text[q + 1:-1]


def parse_char_literal(tok: Token) -> tuple[int, Type]:
    prefix, body = _prefix(tok.text)
    if prefix == "":
        units = _decode(tok, body, False)
        if not units:
            diag.error(tok, "empty character constant")
        if len(units) == 1:
            v = units[0]
            return (v - 256 if v >= 128 else v), TY_INT
        v = 0
        for u in units:
            v = (v << 8) | u
        v &= 0xFFFFFFFF
        return (v - (1 << 32) if v >= 1 << 31 else v), TY_INT
    units = _decode(tok, body, True)
    v = units[-1] if units else 0
    ty = {"L": TY_INT, "u": TY_USHORT, "U": TY_UINT, "u8": TY_UCHAR}[prefix]
    return v & ((1 << (ty.size * 8)) - 1), ty


STRING_ELEM = {"": TY_CHAR, "u8": TY_CHAR, "L": TY_INT, "u": TY_USHORT, "U": TY_UINT}


def parse_string_literal(tok: Token) -> tuple[list[int], Type, str]:
    prefix, body = _prefix(tok.text)
    wide = prefix in ("L", "u", "U")
    units = _decode(tok, body, wide)
    if prefix == "u":
        u16 = []
        for cp in units:
            if cp >= 0x10000:
                cp -= 0x10000
                u16 += [0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)]
            else:
                u16.append(cp)
        units = u16
    return units, STRING_ELEM[prefix], prefix


def string_bytes(units: list[int], elem: Type) -> bytes:
    """Encode code units (plus terminating NUL) as little-endian bytes."""
    sz = elem.size
    if sz == 1:
        return bytes(u & 0xFF for u in units) + b"\0"
    return b"".join((u & ((1 << (8 * sz)) - 1)).to_bytes(sz, "little") for u in units + [0])


def convert_tokens(toks: list[Token], u8_unsigned: bool = False,
                   c23: bool = False, off_keywords: frozenset = frozenset()) -> list[Token]:
    """`u8_unsigned`: C23 types u8"" literals as unsigned char arrays.
    `c23`: `constexpr` is a keyword (it is an ordinary identifier otherwise).
    `off_keywords`: spellings that stay identifiers (e.g. `typeof` in strict ISO modes)."""
    out: list[Token] = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        k = t.kind
        if k == TokenKind.IDENT:
            if c23 and t.text in ("true", "false"):
                t = _c23_bool_const(t)
            elif (t.text in KEYWORDS and t.text not in off_keywords) or (c23 and t.text == "constexpr"):
                t = t.copy()
                t.kind = TokenKind.KEYWORD
                t.text = KEYWORD_ALIASES.get(t.text, t.text)
        elif k == TokenKind.NUMBER:
            t = t.copy()
            t.value, t.ty = parse_number(t)
        elif k == TokenKind.CHAR:
            t = t.copy()
            t.value, t.ty = parse_char_literal(t)
        elif k == TokenKind.STRING:
            j = i + 1
            while j < n and toks[j].kind == TokenKind.STRING:
                j += 1
            t = _concat_strings(toks[i:j], TY_UCHAR if u8_unsigned else TY_CHAR)
            out.append(t)
            i = j
            continue
        elif k == TokenKind.PUNCT and t.text in ("\\", "@", "`", "$", "'", '"'):
            diag.error(t, f"stray '{t.text}' in program")
        out.append(t)
        i += 1
    return out


def _c23_bool_const(t: Token) -> Token:
    """C23: `true`/`false` are constants of type bool (value 1/0)."""
    t = t.copy()
    t.kind = TokenKind.NUMBER
    t.value = int(t.text == "true")
    t.ty = TY_BOOL
    return t


def _concat_strings(pieces: list[Token], u8_elem: Type = TY_CHAR) -> Token:
    """Concatenate adjacent string literals (C11 6.4.5p5)."""
    prefix = ""
    has_u8 = False
    for p in pieces:
        pre, _ = _prefix(p.text)
        has_u8 = has_u8 or pre == "u8"
        if pre and pre != "u8":
            if prefix and prefix != pre:
                diag.error(p, "unsupported non-standard concatenation of string literals")
            prefix = pre
    units: list[int] = []
    for p in pieces:
        if prefix:
            pre, body = _prefix(p.text)
            fake = Token(p.kind, prefix + '"' + body + '"', p.loc)
            units += parse_string_literal(fake)[0]
        else:
            units += parse_string_literal(p)[0]
    elem = STRING_ELEM[prefix] if prefix else (u8_elem if has_u8 else TY_CHAR)
    t = pieces[0].copy()
    t.value = string_bytes(units, elem)
    t.ty = array_of(elem, len(t.value) // elem.size)
    return t
