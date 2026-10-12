"""Tokens and source locations (STABLE, see docs/CONTRACT.md section 3).

A Token is produced by hcc.lexer, rewritten by hcc.preprocess, then
finalized by hcc.literals.convert_tokens (keywords marked, literal values set).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TokenKind(Enum):
    IDENT = "ident"
    KEYWORD = "keyword"
    PUNCT = "punct"
    NUMBER = "number"    # preprocessing number; value/ty set by literals.py
    STRING = "string"
    CHAR = "char"
    EOF = "eof"


@dataclass(slots=True, frozen=True)
class Loc:
    file: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col}"


class Token:
    """One token. `hideset` is used only by the preprocessor."""

    __slots__ = ("kind", "text", "loc", "bol", "space", "hideset", "value", "ty")

    def __init__(self, kind: TokenKind, text: str, loc: Loc,
                 bol: bool = False, space: bool = False,
                 hideset: frozenset = frozenset()):
        self.kind = kind
        self.text = text
        self.loc = loc
        self.bol = bol
        self.space = space
        self.hideset = hideset
        self.value = None
        self.ty = None

    def copy(self) -> "Token":
        t = Token(self.kind, self.text, self.loc, self.bol, self.space, self.hideset)
        t.value = self.value
        t.ty = self.ty
        return t

    def is_(self, text: str) -> bool:
        """True if this is a punctuator/keyword/identifier spelled `text`."""
        return self.text == text and self.kind not in (TokenKind.STRING, TokenKind.CHAR)

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.text!r}, {self.loc})"


KEYWORDS = frozenset("""
auto break case char const continue default do double else enum extern float
for goto if inline int long register restrict return short signed sizeof static
struct switch typedef union unsigned void volatile while
_Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn
_Static_assert _Thread_local __complex__ __complex
__attribute__ __attribute __asm__ __asm asm __extension__ __typeof__ __typeof typeof
__restrict __restrict__ __inline __inline__ __const __const__ __volatile__ __volatile
__signed__ __signed __alignof__ __alignof __thread __label__
__builtin_va_list __int128 __int128_t __uint128_t __float128
__auto_type __builtin_va_arg __builtin_offsetof
__builtin_types_compatible_p __real__ __imag__ __real __imag
""".split())
