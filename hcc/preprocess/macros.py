"""Macro definitions and expansion (Prosser's hideset algorithm, chibicc style).

MacroMixin is mixed into Preprocessor. It owns:
  * Macro dataclass
  * expand_macro(tok): try to expand `tok` (pushes result back on the stack)
  * argument collection, substitution, # stringify, ## paste,
    __VA_ARGS__, __VA_OPT__, GNU `, ## __VA_ARGS__`.
The pending-token stack `self.stack` is REVERSED: stack[-1] is the next token.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hcc import diag
from hcc.lexer import tokenize
from hcc.preprocess.padding import fold_pad, is_pad, make_avoid, make_pad, strip_pads
from hcc.tokens import Token, TokenKind


def _real(toks: list[Token]) -> list[Token]:
    """The non-padding tokens of an argument (empty test, paste operands)."""
    return [t for t in toks if not is_pad(t)]


@dataclass
class Macro:
    name: str
    is_objlike: bool
    params: list[str] = field(default_factory=list)
    va_name: str | None = None          # "__VA_ARGS__" or GNU named variadic
    body: list[Token] = field(default_factory=list)
    handler: object = None              # dynamic macro: fn(pp, tok) -> list[Token]


class MacroMixin:

    # ---- expansion entry point -------------------------------------------
    def expand_macro(self, tok: Token) -> bool:
        name = tok.text
        if name in tok.hideset:
            return False
        m = self.macros.get(name)
        if m is None:
            return False
        if m.handler is not None:
            self.push_tokens(m.handler(self, tok))
            return True
        if m.is_objlike:
            hs = tok.hideset | {name}
            out = self.subst(m, {}, tok)
            for c in out:
                c.hideset = c.hideset | hs
                c.loc = tok.loc
                c.bol = False
            self._finish_expansion(tok, out)
            return True
        # function-like: must be followed by '(' (padding in between is skipped)
        j = self._lparen_index()
        if j < 0:
            return False
        del self.stack[j:]
        args, rparen = self.read_macro_args(m, tok)
        hs = (tok.hideset & rparen.hideset) | {name}
        out = self.subst(m, args, tok)
        for t in out:
            t.hideset = t.hideset | hs
            t.loc = tok.loc
            t.bol = False
        self._finish_expansion(tok, out)
        return True

    def _lparen_index(self) -> int:
        """Stack index of the '(' that follows a function-like macro name, or -1."""
        stack = self.stack
        j = len(stack) - 1
        while j >= 0 and is_pad(stack[j]):
            j -= 1
        if j >= 0 and stack[j].kind != TokenKind.EOF and stack[j].is_("("):
            return j
        return -1

    def _finish_expansion(self, tok: Token, out: list[Token]) -> None:
        if out:
            out[0].space = tok.space
        # libcpp: padding for the macro name, then avoid_paste when the context ends
        self.push_tokens([make_pad(tok)] + out + [make_avoid(tok.loc)])

    def push_tokens(self, toks: list[Token]) -> None:
        """Make `toks` the next tokens to be read."""
        self.stack.extend(reversed(toks))

    # ---- argument collection ----------------------------------------------
    def read_macro_args(self, m: Macro, name_tok: Token):
        """Read args after '('. Returns (dict param->list[Token], rparen)."""
        args: list[list[Token]] = []
        cur: list[Token] = []
        depth = 0
        stack = self.stack
        nparams = len(m.params)
        while True:
            if not stack:
                diag.error(name_tok, f"unterminated call to macro '{m.name}'")
            t = stack.pop()
            if t.kind == TokenKind.EOF:
                diag.error(name_tok, f"unterminated call to macro '{m.name}'")
            if is_pad(t) and not cur:
                continue            # collect_args drops leading padding
            if t.kind == TokenKind.PUNCT and t.bol and t.text == "#" and not t.hideset:
                # directives may appear inside macro arguments (C11 6.10.3p11)
                self.directive(t)
                continue
            if t.kind == TokenKind.PUNCT:
                if t.text == "(":
                    depth += 1
                elif t.text == ")":
                    if depth == 0:
                        args.append(cur)
                        rparen = t
                        break
                    depth -= 1
                elif t.text == "," and depth == 0 and not (m.va_name and len(args) == nparams):
                    args.append(cur)
                    cur = []
                    continue
            cur.append(t)
        result: dict[str, list[Token]] = {}
        if nparams == 0 and not m.va_name and len(args) == 1 and not _real(args[0]):
            return result, rparen
        if m.va_name:
            if len(args) == nparams:          # variadic part omitted
                args.append([])
            if len(args) != nparams + 1:
                diag.error(name_tok, f"macro '{m.name}' requires {nparams}+ arguments, got {len(args)}")
            result[m.va_name] = args[nparams]
        elif len(args) != nparams:
            diag.error(name_tok, f"macro '{m.name}' requires {nparams} arguments, but {len(args)} given")
        for p, a in zip(m.params, args):
            result[p] = a
        return result, rparen

    # ---- substitution -----------------------------------------------------
    def subst(self, m: Macro, args: dict, call_tok: Token) -> list[Token]:
        body = m.body
        out: list[Token] = []
        expanded: dict[str, list[Token]] = {}
        n = len(body)
        i = 0

        def is_param(j):
            return j < n and body[j].kind == TokenKind.IDENT and body[j].text in args

        def nxt_is(j, text):
            return j < n and body[j].kind == TokenKind.PUNCT and body[j].text == text

        while i < n:
            t = body[i]
            if t.kind == TokenKind.PUNCT and t.text == "#" and not m.is_objlike:
                if not is_param(i + 1):
                    diag.error(t, "'#' is not followed by a macro parameter")
                out.append(self.stringize(t, args[body[i + 1].text]))
                i += 2
                continue
            if t.text == "," and nxt_is(i + 1, "##") and i + 2 < n and body[i + 2].text == m.va_name:
                if not _real(args[m.va_name]):
                    i += 3
                else:
                    out.append(t.copy())
                    i += 2
                continue
            if t.kind == TokenKind.PUNCT and t.text == "##":
                if not out:
                    diag.error(t, "'##' cannot appear at start of macro expansion")
                # A run of consecutive `##` operators acts as a single one (GCC).
                j = i + 1
                while j < n and body[j].kind == TokenKind.PUNCT and body[j].text == "##":
                    j += 1
                if j >= n:
                    diag.error(t, "'##' cannot appear at end of macro expansion")
                if is_param(j):
                    a = _real(args[body[j].text])
                    if a:
                        out[-1] = self.paste(out[-1], a[0])
                        out.extend(x.copy() for x in a[1:])
                else:
                    out[-1] = self.paste(out[-1], body[j])
                i = j + 1
                continue
            if t.kind == TokenKind.IDENT and t.text == "__VA_OPT__" and nxt_is(i + 1, "("):
                j, depth, inner = i + 2, 0, []
                while j < n and not (depth == 0 and nxt_is(j, ")")):
                    if nxt_is(j, "("):
                        depth += 1
                    elif nxt_is(j, ")"):
                        depth -= 1
                    inner.append(body[j])
                    j += 1
                if m.va_name and _real(args.get(m.va_name, [])):
                    sub = Macro(m.name, False, m.params, m.va_name, inner)
                    out.extend(self.subst(sub, args, call_tok))
                i = j + 1
                continue
            if is_param(i):
                a = _real(args[t.text])
                if nxt_is(i + 1, "##"):
                    if not a:
                        rhs_i = i + 2
                        if is_param(rhs_i) and nxt_is(rhs_i + 1, "##"):
                            # Empty lhs placemarker: the rhs operand is handled
                            # by the next iteration so that a chain such as
                            # `x ## y ## z` with x and y empty still pastes.
                            i = rhs_i
                            continue
                        if is_param(rhs_i):
                            out.extend(x.copy() for x in args[body[rhs_i].text])
                        elif rhs_i < n:
                            out.append(body[rhs_i].copy())
                        i = rhs_i + 1
                        continue
                    out.extend(x.copy() for x in a)
                    i += 1
                    continue
                if t.text not in expanded:
                    expanded[t.text] = self._expand_arg(args[t.text])
                ex = [x.copy() for x in expanded[t.text]]
                if ex:
                    ex[0].space = t.space
                # libcpp replace_args: padding around a substituted argument
                if i:
                    out.append(make_pad(t))
                out.extend(ex)
                out.append(make_avoid(t.loc))
                i += 1
                continue
            out.append(t.copy())
            i += 1
        return out

    def stringize(self, hash_tok: Token, arg: list[Token]) -> Token:
        parts = []
        source = None       # libcpp stringify_arg: PREV_WHITE of pending padding source
        for t in arg:
            if is_pad(t):
                source = fold_pad(source, t)
                continue
            if parts and (t.space if source is None else source):
                parts.append(" ")
            source = None
            if t.kind in (TokenKind.STRING, TokenKind.CHAR):
                parts.append(t.text.replace("\\", "\\\\").replace('"', '\\"'))
            else:
                parts.append(t.spell or t.text)
        s = "".join(parts)
        # GCC: an odd run of trailing backslashes would escape the closing
        # quote; warn and drop the final backslash (as cpp does).
        tail = len(s) - len(s.rstrip("\\"))
        if tail % 2 == 1:
            diag.warn(hash_tok, "invalid string literal, ignoring final '\\'")
            s = s[:-1]
        return Token(TokenKind.STRING, f'"{s}"', hash_tok.loc, False, hash_tok.space)

    def paste(self, lhs: Token, rhs: Token) -> Token:
        text = (lhs.spell or lhs.text) + (rhs.spell or rhs.text)
        toks = tokenize(text, lhs.loc.file)
        if len(toks) != 2:
            diag.error(lhs, f"pasting forms '{text}', an invalid token")
        t = toks[0]
        t.loc, t.bol, t.space, t.hideset = lhs.loc, lhs.bol, lhs.space, lhs.hideset
        return t

    # ---- helpers ------------------------------------------------------------
    def expand_list(self, toks: list[Token]) -> list[Token]:
        """Fully macro-expand a token list in isolation (no directives)."""
        return strip_pads(self._expand_arg(toks))

    def _expand_arg(self, toks: list[Token]) -> list[Token]:
        """expand_list keeping the padding that stringify needs (macro arguments)."""
        if not toks:
            return []
        if not any(t.kind == TokenKind.IDENT and t.text in self.macros for t in toks):
            return toks
        saved = self.stack
        eof = Token(TokenKind.EOF, "", toks[-1].loc, True)
        self.stack = [eof] + list(reversed(toks))
        out = []
        try:
            while True:
                t = self.stack.pop()
                if t.kind == TokenKind.EOF and t is eof:
                    break
                if t.kind == TokenKind.IDENT and self.expand_macro(t):
                    continue
                out.append(t)
        finally:
            self.stack = saved
        return out
