"""The C preprocessor (hcc's own; never calls cpp).

Public API:
    pp = Preprocessor(opts)
    tokens = pp.run(path)            # list[Token], last is EOF
    text = format_tokens(tokens)     # for -E output

Implementation: a reversed token stack (`self.stack`, stack[-1] is next)
fed by files and macro expansions. The main loop is `_loop`.
Mixins (every `*Mixin` class of this package, found by hcc.registry): MacroMixin
(macros.py), DirectiveMixin (directives.py), ConditionalMixin (conditionals.py),
IncludeMixin (includes.py).
"""
from __future__ import annotations

import os

from hcc import diag
from hcc.lexer import C90_LITERAL_STDS, C90_STDS, TRIGRAPH_STDS, tokenize
from hcc.preprocess.macros import Macro
from hcc.preprocess.padding import fold_pad, is_pad
from hcc.preprocess.predefined import DYNAMIC_MACROS, predefined_source
from hcc.registry import assemble_mixins
from hcc.tokens import Token, TokenKind


# Every `*Mixin` class in a module of hcc/preprocess/ is a base (hcc/registry.py):
# a new directive family is a new file with a mixin defining `dir_<name>`.
class Preprocessor(*assemble_mixins("hcc.preprocess", object)):

    def __init__(self, opts):
        self.opts = opts
        self.macros: dict[str, Macro] = {}
        self.pushed_macros: dict[str, list] = {}   # #pragma push_macro stacks
        self.assertions: dict[str, list[str]] = {}   # GNU #assert predicate -> answers
        self.stack: list[Token] = []
        self.cond_stack = []
        self.counter = 0
        self.line_marks: dict[str, list[tuple[int, int, str]]] = {}   # #line marks per file
        self.include_depth = 0
        self.base_file = ""
        self.init_search_path()
        for name, fn in DYNAMIC_MACROS.items():
            self.macros[name] = Macro(name, True, handler=fn)

    def _command_line_source(self) -> str:
        lines = [predefined_source(self.opts)]
        for kind, text in self.opts.defines:
            if kind == "-D":
                name, _, val = text.partition("=")
                lines.append(f"#define {name} {val if '=' in text else '1'}")
            else:
                lines.append(f"#undef {text}")
        for inc in self.opts.includes:
            lines.append(f'#include "{os.path.abspath(inc)}"')
        return "\n".join(lines) + "\n"

    @property
    def line_comments(self) -> bool:
        """`//` starts a comment except in C89/C90 modes (-std=c89, -ansi)."""
        return (self.opts.std or "gnu11").lower() not in C90_STDS

    @property
    def u8_chars(self) -> bool:
        """`u8'x'` is a character constant only in C23 modes."""
        return (self.opts.std or "").lower() in ("c23", "gnu23", "c2x", "gnu2x")

    @property
    def c90_literals(self) -> bool:
        """No u8/u/U literal prefixes before C11 (gnu89 `u"a"` is identifier + string)."""
        return (self.opts.std or "gnu11").lower() in C90_LITERAL_STDS

    @property
    def trigraphs(self) -> bool:
        """Trigraphs are on in strict ISO modes (-std=c11, -ansi) or with -trigraphs."""
        return self.opts.trigraphs or (self.opts.std or "").lower() in TRIGRAPH_STDS

    def run(self, path: str) -> list[Token]:
        self.base_file = path
        main = self.file_tokens(path)
        cmdline = tokenize(self._command_line_source(), "<command-line>", self.line_comments,
                           self.u8_chars, self.trigraphs, self.c90_literals)
        self.stack = []
        self.push_tokens(main)
        cmd_eof = cmdline.pop()      # drop its EOF so it flows into the main file
        self.push_tokens(cmdline)
        out = self._loop()
        if self.cond_stack:
            diag.error(self.cond_stack[-1].tok, "unterminated conditional directive")
        return out

    def run_text(self, text: str, path: str = "<string>") -> list[Token]:
        """Preprocess a string (used by tests and -x c -)."""
        self.base_file = path
        self.stack = []
        self.push_tokens(tokenize(text, path, self.line_comments, self.u8_chars,
                                  self.trigraphs, self.c90_literals))
        cmdline = tokenize(self._command_line_source(), "<command-line>", self.line_comments,
                           self.u8_chars, self.trigraphs, self.c90_literals)
        cmdline.pop()
        self.push_tokens(cmdline)
        return self._loop()

    def _loop(self) -> list[Token]:
        out: list[Token] = []
        stack = self.stack
        macros = self.macros
        IDENT, EOF, PUNCT = TokenKind.IDENT, TokenKind.EOF, TokenKind.PUNCT
        pad_src = None      # padding seen since the last token: sets that token's space
        while True:
            stack = self.stack
            tok = stack.pop()
            kind = tok.kind
            if kind == PUNCT and not tok.text:
                pad_src = fold_pad(pad_src, tok)      # padding (preprocess/padding.py)
                continue
            if pad_src is not None:
                tok.space, pad_src = pad_src, None
            if kind == IDENT:
                if tok.text in macros and self.expand_macro(tok):
                    continue
                if tok.text == "_Pragma" and self._lparen_index() >= 0:
                    self._skip_pragma_operator(tok)
                    continue
                out.append(tok)
                continue
            if kind == PUNCT and tok.bol and tok.text == "#" and not tok.hideset:
                self.directive(tok)
                continue
            if kind == EOF:
                if stack:
                    self.include_depth -= 1
                    continue
                out.append(tok)
                return out
            out.append(tok)

    def _skip_pragma_operator(self, tok: Token) -> None:
        del self.stack[self._lparen_index():]    # ( (and padding before it)
        depth = 1
        while depth:
            t = self.stack.pop()
            if t.kind == TokenKind.EOF:
                diag.error(tok, "unterminated _Pragma")
            if t.text == "(":
                depth += 1
            elif t.text == ")":
                depth -= 1


def format_tokens(toks: list[Token]) -> str:
    """Render tokens as text for `-E` (newline at bol, space where `space`)."""
    parts = []
    line = None
    file = None
    for t in toks:
        if t.kind == TokenKind.EOF:
            break
        if t.bol and parts:
            if t.loc.file == file and line is not None and 0 < t.loc.line - line <= 8:
                parts.append("\n" * (t.loc.line - line))
            else:
                parts.append("\n")
        elif t.space and parts:
            parts.append(" ")
        parts.append(t.text)
        line, file = t.loc.line, t.loc.file
    parts.append("\n")
    return "".join(parts)
