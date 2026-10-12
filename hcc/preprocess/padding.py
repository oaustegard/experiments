"""GCC-style padding tokens for macro expansion (libcpp padding_token/avoid_paste).

A padding token is a PUNCT token with empty text. Its `value` is the
PREV_WHITE flag of its source token (True/False), or None for the
"avoid_paste" padding that ends a macro context. Padding only matters to
`#` stringification (spaces between the tokens of an expansion); every other
consumer drops it (see strip_pads / the main loop in preprocess/__init__.py).
"""
from __future__ import annotations

from hcc.tokens import Loc, Token, TokenKind


def is_pad(t: Token) -> bool:
    return t.kind == TokenKind.PUNCT and not t.text


def make_pad(src: Token) -> Token:
    """padding_token(source): carries the whitespace flag of `src`."""
    return _pad(src.loc, bool(src.space))


def make_avoid(loc: Loc) -> Token:
    """avoid_paste: a padding with no source (ends a macro expansion or argument)."""
    return _pad(loc, None)


def _pad(loc: Loc, value) -> Token:
    t = Token(TokenKind.PUNCT, "", loc)
    t.value = value
    return t


def strip_pads(toks: list[Token]) -> list[Token]:
    return [t for t in toks if not is_pad(t)]


def fold_pad(source, pad: Token):
    """Apply one padding token to the pending source state (libcpp stringify_arg /
    scan_translation_unit): None = no source, else the PREV_WHITE of the source."""
    if source is None or (source is False and pad.value is None):
        return pad.value
    return source
