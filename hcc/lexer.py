"""Lexer: source text -> list of preprocessing tokens.

Public API:
    tokenize(text, path, line_comments=True) -> list[Token]   (ends with EOF)
    tokenize_file(path, line_comments=True) -> list[Token]
line_comments=False (C89/C90 mode) makes `//` two `/` punctuators.

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
  | (?P<num>\.?[0-9](?:[eEpP][+-]|[0-9A-Za-z_.\u0080-￿])*)
  | (?P<str>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
  | (?P<chr>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)*')
  | (?P<id>[A-Za-z_$\u0080-\U0010ffff][A-Za-z0-9_$\u0080-\U0010ffff]*)
  | (?P<punct>""" + "|".join(re.escape(p) for p in PUNCTUATORS) + r"""|[\[\](){}.&*+\-~!/%<>^|?:;=,#\\])
  | (?P<other>.)
"""
_TOKEN_RE = re.compile(_TOKEN_SRC, re.VERBOSE | re.DOTALL)
# C89/C90 have no `//` comments: the same scanner without the `lc` alternative.
_TOKEN_RE_C90 = re.compile(_TOKEN_SRC.replace(r"  | (?P<lc>//[^\n]*)" + "\n", ""),
                           re.VERBOSE | re.DOTALL)
# -std values that do not recognise `//` comments
C90_STDS = frozenset({"c89", "c90", "gnu89", "gnu90", "iso9899:1990", "ansi"})

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


def tokenize(text: str, path: str, line_comments: bool = True) -> list[Token]:
    text = remove_splices(text)
    toks: list[Token] = []
    line = 1
    line_start = 0
    bol = True
    space = False
    scanner = _TOKEN_RE if line_comments else _TOKEN_RE_C90
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
        if kind == "punct" and s in DIGRAPHS:
            s = DIGRAPHS[s]
        toks.append(Token(_KIND[kind], s, loc, bol, space))
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


def tokenize_file(path: str, line_comments: bool = True) -> list[Token]:
    return tokenize(read_source(path), path, line_comments)
