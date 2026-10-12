"""Lexer: source text -> list of preprocessing tokens.

Public API:
    tokenize(text, path, line_comments=True, u8_chars=False) -> list[Token]
    tokenize_file(path, line_comments=True, u8_chars=False) -> list[Token]
line_comments=False (C89/C90 mode) makes `//` two `/` punctuators and bars
non-ASCII identifiers. u8_chars=True (C23) makes u8'x' a char constant.

Tokens are pp-tokens (CONTRACT section 3): identifiers (keywords are NOT yet
marked), pp-numbers, string/char literals (raw spelling incl. quotes and
prefix), punctuators. Unknown characters (e.g. a stray `'` or `@` inside
`#if 0`) become single-char PUNCT tokens; the parser rejects them later.
Line splices (backslash-newline) are removed; to keep line numbers right the
removed newlines are re-inserted after the end of the logical line.
"""
from __future__ import annotations

import re

from hcc import diag
from hcc.tokens import Loc, Token, TokenKind

PUNCTUATORS = [
    "...", "<<=", ">>=", "%:%:",
    "->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "##",
    "<:", ":>", "<%", "%>", "%:",
]
DIGRAPHS = {"<:": "[", ":>": "]", "<%": "{", "%>": "}", "%:": "#", "%:%:": "##"}

_TOKEN_SRC = r"""
    (?P<nl>\n)
  | (?P<ws>[ \t\f\v\r]+)
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<ubc>/\*)
  | (?P<num>\.?0[xX](?:[pP][+-]|DIGSEP[0-9A-Za-z_.\u0080-￿]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*
            |\.?[0-9](?:[eE][+-]|DIGSEP[0-9A-Za-z_.\u0080-￿]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*)
  | (?P<str>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
  | (?P<chr>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)*')
  | (?P<id>(?:[A-Za-z_$\u0080-\U0010ffff]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})
             (?:[A-Za-z0-9_$\u0080-\U0010ffff]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*)
  | (?P<punct>""" + "|".join(re.escape(p) for p in PUNCTUATORS) + r"""|[\[\](){}.&*+\-~!/%<>^|?:;=,#\\])
  | (?P<other>.)
"""
def _make_scanner(line_comments: bool, u8_chars: bool) -> "re.Pattern[str]":
    src = _TOKEN_SRC
    if not line_comments:
        # C89/C90 have no `//` comments: drop the `lc` alternative.
        src = src.replace(r"  | (?P<lc>//[^\n]*)" + "\n", "")
        # C89/C90 do not allow extended (non-ASCII) characters in identifiers.
        src = src.replace(r"\u0080-\U0010ffff", "")
        # ... and no universal character names (\uXXXX) inside identifiers
        # either: `aª` lexes as `a`, `\`, `u00aa` (GCC -std=c89).
        head, sep, tail = src.partition("(?P<id>")
        tail = tail.replace(r"|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}", "")
        src = head + sep + tail
        # ... and only `L` is a literal prefix (C90 has no u8/u/U prefixes):
        # gnu89 `u'a'` lexes as identifier `u` then char constant (dg utf-dflt).
        src = src.replace("(?:u8|u|U|L)?", "(?:L)?")
    if not u8_chars:
        # u8'x' character constants exist only in C23; before that `u8` is an identifier.
        src = src.replace("(?P<chr>(?:u8|u|U|L)?", "(?P<chr>(?:u|U|L)?")
        # C23 digit separators (`1'000`) exist only in C23; before that `'` ends a number.
        src = src.replace("DIGSEP", "")
    else:
        # C23 digit separator: pp-number ' digit / pp-number ' nondigit (6.4.9).
        src = src.replace("DIGSEP", "'(?=[0-9A-Za-z_\u0080-\U0010ffff])|")
    return re.compile(src, re.VERBOSE | re.DOTALL)


_SCANNERS = {(lc, u8): _make_scanner(lc, u8) for lc in (True, False) for u8 in (True, False)}
# -std values that do not recognise `//` comments
C90_STDS = frozenset({"c89", "c90", "gnu89", "gnu90", "iso9899:1990", "ansi"})

_UCN_RE = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")


def _decode_ucns(s: str) -> str:
    """Replace universal character names (C11 6.4.3) in an identifier."""
    if "\\" not in s:
        return s
    return _UCN_RE.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), s)


_KIND = {"num": TokenKind.NUMBER, "str": TokenKind.STRING, "chr": TokenKind.CHAR,
         "id": TokenKind.IDENT, "punct": TokenKind.PUNCT, "other": TokenKind.PUNCT}


def remove_splices(text: str) -> str:
    """Delete backslash-newlines, re-adding the lost newlines afterwards."""
    if "\\" not in text:
        return text
    text = text.replace("\r\n", "\n")
    if "\\\n" not in text and not re.search(r"\\[ \t]+\n", text):
        return text
    out = []
    pending = 0
    for line in text.split("\n"):
        stripped = line.rstrip(" \t")
        if stripped.endswith("\\"):
            out.append(stripped[:-1])
            pending += 1
            continue
        out.append(line + "\n" * (pending + 1))
        pending = 0
    result = "".join(out)
    return result[:-1] if not text.endswith("\n") and result.endswith("\n") else result


_TRIGRAPHS = {"=": "#", "(": "[", "/": "\\", ")": "]", "'": "^",
              "<": "{", "!": "|", ">": "}", "-": "~"}
_TRIGRAPH_RE = re.compile(r"\?\?([=(/)'<!>-])")
# -std values in which trigraphs are enabled (strict ISO modes); gnu* disable them
TRIGRAPH_STDS = frozenset({"c89", "c90", "c99", "c9x", "c11", "c1x", "c17", "c18",
                           "c2x", "c23", "ansi", "iso9899:1990", "iso9899:199409",
                           "iso9899:1999", "iso9899:199x", "iso9899:2011",
                           "iso9899:2017", "iso9899:2018", "iso9899:2024"})


def replace_trigraphs(text: str) -> str:
    """Translation phase 1: replace the nine trigraph sequences (C11 5.2.1.1)."""
    if "??" not in text:
        return text
    return _TRIGRAPH_RE.sub(lambda m: _TRIGRAPHS[m.group(1)], text)


def tokenize(text: str, path: str, line_comments: bool = True,
             u8_chars: bool = False, trigraphs: bool = False) -> list[Token]:
    if trigraphs:
        text = replace_trigraphs(text)
    text = remove_splices(text)
    toks: list[Token] = []
    line = 1
    line_start = 0
    bol = True
    space = False
    scanner = _SCANNERS[(bool(line_comments), bool(u8_chars))]
    for m in scanner.finditer(text):
        kind = m.lastgroup
        if kind == "nl":
            line += 1
            line_start = m.end()
            bol = True
            space = False
            continue
        if kind in ("ws", "lc"):
            space = True
            continue
        if kind == "bc":
            n = m.group().count("\n")
            if n:
                line += n
                line_start = m.start() + m.group().rfind("\n") + 1
            space = True
            continue
        loc = Loc(path, line, m.start() - line_start + 1)
        if kind == "ubc":
            diag.error(loc, "unterminated comment", "preprocess")
        s = m.group()
        spell = None
        if kind == "punct" and s in DIGRAPHS:
            spell = s
            s = DIGRAPHS[s]
        elif kind == "id":
            s = _decode_ucns(s)
        tok = Token(_KIND[kind], s, loc, bol, space)
        tok.spell = spell
        toks.append(tok)
        bol = False
        space = False
    toks.append(Token(TokenKind.EOF, "", Loc(path, line, 1), True, False))
    return toks


def read_source(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="surrogateescape")
    if text.startswith("﻿"):
        text = text[1:]
    return text


def tokenize_file(path: str, line_comments: bool = True,
                  u8_chars: bool = False, trigraphs: bool = False) -> list[Token]:
    return tokenize(read_source(path), path, line_comments, u8_chars, trigraphs)
