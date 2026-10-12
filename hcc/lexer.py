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
from hcc.ident_ranges import in_c11_identifier_range
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
  | (?P<bad>[\udc80-\udcff])
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<ubc>/\*)
  | (?P<num>\.?0[xX](?:[pP][+-]|DIGSEP[0-9A-Za-z_.\u0080-￿]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*
            |\.?[0-9](?:[eE][+-]|DIGSEP[0-9A-Za-z_.\u0080-￿]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*)
  | (?P<str>(?:u8|u|U|L)?"(?:[^"\\\n]|\\.)*")
  | (?P<chr>(?:u8|u|U|L)?'(?:[^'\\\n]|\\.)*')
  | (?P<id>(?:[A-Za-z_$\u0080-\udc7f\udd00-\U0010ffff]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})
             (?:[A-Za-z0-9_$\u0080-\udc7f\udd00-\U0010ffff]|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8})*)
  | (?P<punct>""" + "|".join(re.escape(p) for p in PUNCTUATORS) + r"""|[\[\](){}.&*+\-~!/%<>^|?:;=,#\\])
  | (?P<other>.)
"""
def _make_scanner(line_comments: bool, u8_chars: bool, c90_literals: bool = False) -> "re.Pattern[str]":
    src = _TOKEN_SRC
    if c90_literals:
        # gnu89/gnu90 (and C89/C90) have no u8/u/U literal prefixes: `u"a"` is the
        # identifier `u` followed by a string (dg utf-dflt, utf-dflt2).
        src = src.replace("(?:u8|u|U|L)?", "(?:L)?")
    if not line_comments:
        # C89/C90 have no `//` comments: drop the `lc` alternative.
        src = src.replace(r"  | (?P<lc>//[^\n]*)" + "\n", "")
        # C89/C90 do not allow extended (non-ASCII) characters in identifiers.
        src = src.replace(r"\u0080-\udc7f\udd00-\U0010ffff", "")
        # ... and no universal character names (\uXXXX) inside identifiers
        # either: `aª` lexes as `a`, `\`, `u00aa` (GCC -std=c89).
        head, sep, tail = src.partition("(?P<id>")
        tail = tail.replace(r"|\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}", "")
        src = head + sep + tail
        # ... and only `L` is a literal prefix (C90 has no u8/u/U prefixes):
        # gnu89 `u'a'` lexes as identifier `u` then char constant (dg utf-dflt).
        src = src.replace("(?:u8|u|U|L)?", "(?:L)?")
        # C90 pp-numbers have no hex exponent sign: `0x1p+f` is `0x1p`, `+`, `f`
        # (the sign is part of a pp-number only from C99 on).
        src = src.replace("[pP][+-]|", "")
    if not u8_chars:
        # u8'x' character constants exist only in C23; before that `u8` is an identifier.
        src = src.replace("(?P<chr>(?:u8|u|U|L)?", "(?P<chr>(?:u|U|L)?")
        # C23 digit separators (`1'000`) exist only in C23; before that `'` ends a number.
        src = src.replace("DIGSEP", "")
    else:
        # C23 digit separator: pp-number ' digit / pp-number ' nondigit (6.4.9).
        src = src.replace("DIGSEP", "'(?=[0-9A-Za-z_\u0080-\U0010ffff])|")
    return re.compile(src, re.VERBOSE | re.DOTALL)


_SCANNERS = {(lc, u8, c90): _make_scanner(lc, u8, c90) for lc in (True, False)
             for u8 in (True, False) for c90 in (True, False)}
# Strict -std values that do not recognise `//` comments (GCC: gnu89/gnu90 keep them)
C90_STDS = frozenset({"c89", "c90", "iso9899:1990", "iso9899:199409", "ansi"})
# -std values without u8/u/U literal prefixes (pre-C11; gnu89/gnu90 included)
C90_LITERAL_STDS = C90_STDS | frozenset({"gnu89", "gnu90"})

_UCN_RE = re.compile(r"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})")


def _decode_ucns(s: str) -> str:
    """Replace universal character names (C11 6.4.3) in an identifier."""
    if "\\" not in s:
        return s
    return _UCN_RE.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), s)


_UCN_TAIL_RE = re.compile(r"\\u[0-9A-Fa-f]{4}|\\U[0-9A-Fa-f]{8}")


def _xid_ok(c: str, start: bool) -> bool:
    """C23 6.4.2.1 / XID_Start, XID_Continue: may character c begin/continue an identifier."""
    if start:
        return c.isidentifier()
    # Continuation: XID_Continue, or the C11 Annex D.1 ranges gcc accepts (e.g. `a¾`).
    return ("a" + c).isidentifier() or in_c11_identifier_range(c)


def _split_id(s: str) -> list[tuple[str, str]]:
    """Split a raw identifier match into (kind, text) pieces. Non-ASCII characters
    that are not XID characters end the identifier and become single stray tokens
    (GCC: `int ² = 42;` says "stray '\\302' in program")."""
    if s.isascii():
        return [("id", _decode_ucns(s))]
    out: list[tuple[str, str]] = []
    cur = ""
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = _UCN_TAIL_RE.match(s, i)
            if m:
                cur += m.group()
                i = m.end()
                continue
        if ord(c) < 0x80 or _xid_ok(c, start=(cur == "")):
            cur += c
        else:
            if cur:
                out.append(("id", _decode_ucns(cur)))
                cur = ""
            out.append(("other", c))
        i += 1
    if cur:
        out.append(("id", _decode_ucns(cur)))
    return out


_KIND = {"num": TokenKind.NUMBER, "str": TokenKind.STRING, "chr": TokenKind.CHAR,
         "id": TokenKind.IDENT, "punct": TokenKind.PUNCT, "other": TokenKind.PUNCT,
         "bad": TokenKind.PUNCT}


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
             u8_chars: bool = False, trigraphs: bool = False,
             c90_literals: bool = False) -> list[Token]:
    if trigraphs:
        text = replace_trigraphs(text)
    text = remove_splices(text)
    toks: list[Token] = []
    line = 1
    line_start = 0
    bol = True
    space = False
    scanner = _SCANNERS[(bool(line_comments), bool(u8_chars), bool(c90_literals))]
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
        pieces = _split_id(s) if kind == "id" else [(kind, s)]
        for idx, (pkind, s) in enumerate(pieces):
            spell = None
            if pkind == "punct" and s in DIGRAPHS:
                spell = s
                s = DIGRAPHS[s]
            tok = Token(_KIND[pkind], s, loc, bol if idx == 0 else False,
                        space if idx == 0 else False)
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
                  u8_chars: bool = False, trigraphs: bool = False,
                  c90_literals: bool = False) -> list[Token]:
    return tokenize(read_source(path), path, line_comments, u8_chars, trigraphs, c90_literals)
