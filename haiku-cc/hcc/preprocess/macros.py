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
from hcc.tokens import Token, TokenKind


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
            out = []
            for t in m.body:
                c = t.copy()
                c.hideset = c.hideset | hs
                c.loc = tok.loc
                c.bol = False
                out.append(c)
            self._finish_expansion(tok, out)
            return True
        # function-like: must be followed by '('
        if not self.stack or not self.stack[-1].is_("(") or self.stack[-1].kind == TokenKind.EOF:
            return False
        self.stack.pop()
        args, rparen = self.read_macro_args(m, tok)
        hs = (tok.hideset & rparen.hideset) | {name}
        out = self.subst(m, args, tok)
        for t in out:
            t.hideset = t.hideset | hs
            t.loc = tok.loc
            t.bol = False
        self._finish_expansion(tok, out)
        return True

    def _finish_expansion(self, tok: Token, out: list[Token]) -> None:
        if out:
            out[0].space = tok.space
        self.push_tokens(out)

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
        if nparams == 0 and not m.va_name and len(args) == 1 and not args[0]:
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
            if t.kind == TokenKind.PUNCT and t.text == "#":
                if not is_param(i + 1):
                    diag.error(t, "'#' is not followed by a macro parameter")
                out.append(self.stringize(t, args[body[i + 1].text]))
                i += 2
                continue
            if t.text == "," and nxt_is(i + 1, "##") and i + 2 < n and body[i + 2].text == m.va_name:
                if not args[m.va_name]:
                    i += 3
                else:
                    out.append(t.copy())
                    i += 2
                continue
            if t.kind == TokenKind.PUNCT and t.text == "##":
                if not out:
                    diag.error(t, "'##' cannot appear at start of macro expansion")
                if i + 1 >= n:
                    diag.error(t, "'##' cannot appear at end of macro expansion")
                if is_param(i + 1):
                    a = args[body[i + 1].text]
                    if a:
                        out[-1] = self.paste(out[-1], a[0])
                        out.extend(x.copy() for x in a[1:])
                else:
                    out[-1] = self.paste(out[-1], body[i + 1])
                i += 2
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
                if m.va_name and args.get(m.va_name):
                    sub = Macro(m.name, False, m.params, m.va_name, inner)
                    out.extend(self.subst(sub, args, call_tok))
                i = j + 1
                continue
            if is_param(i):
                a = args[t.text]
                if nxt_is(i + 1, "##"):
                    if not a:
                        rhs_i = i + 2
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
                    expanded[t.text] = self.expand_list(a)
                ex = [x.copy() for x in expanded[t.text]]
                if ex:
                    ex[0].space = t.space
                out.extend(ex)
                i += 1
                continue
            out.append(t.copy())
            i += 1
        return out

    def stringize(self, hash_tok: Token, arg: list[Token]) -> Token:
        parts = []
        for k, t in enumerate(arg):
            if k and t.space:
                parts.append(" ")
            if t.kind in (TokenKind.STRING, TokenKind.CHAR):
                parts.append(t.text.replace("\\", "\\\\").replace('"', '\\"'))
            else:
                parts.append(t.text)
        s = "".join(parts)
        return Token(TokenKind.STRING, f'"{s}"', hash_tok.loc, False, hash_tok.space)

    def paste(self, lhs: Token, rhs: Token) -> Token:
        text = lhs.text + rhs.text
        toks = tokenize(text, lhs.loc.file)
        if len(toks) != 2:
            diag.error(lhs, f"pasting forms '{text}', an invalid token")
        t = toks[0]
        t.loc, t.bol, t.space, t.hideset = lhs.loc, lhs.bol, lhs.space, lhs.hideset
        return t

    # ---- helpers ------------------------------------------------------------
    def expand_list(self, toks: list[Token]) -> list[Token]:
        """Fully macro-expand a token list in isolation (no directives)."""
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
