"""Preprocessing directives: #define #undef #include #include_next #if #ifdef
#ifndef #elif #else #endif #line #error #warning #pragma #ident.

DirectiveMixin is mixed into Preprocessor. `directive(hash_tok)` is called by
the main loop when it sees a `#` at the beginning of a line.
Unknown pragmas are dropped (TODO: forward `#pragma pack` to the parser).
"""
from __future__ import annotations

from dataclasses import dataclass

from hcc import diag
from hcc.preprocess.macros import Macro
from hcc.tokens import Token, TokenKind


@dataclass
class CondIncl:
    ctx: str          # "if" | "elif" | "else"
    tok: Token
    included: bool


class DirectiveMixin:

    def read_line(self) -> list[Token]:
        """Pop and return the remaining tokens of the current line."""
        out = []
        stack = self.stack
        while stack and not stack[-1].bol:
            out.append(stack.pop())
        return out

    def directive(self, hash_tok: Token) -> None:
        stack = self.stack
        if stack[-1].bol:          # null directive "#"
            return
        name_tok = stack.pop()
        name = name_tok.text
        handler = getattr(self, "dir_" + name, None) if name_tok.kind == TokenKind.IDENT or name in ("if", "else") else None
        if name_tok.kind == TokenKind.NUMBER:     # GNU line marker "# 12 "file""
            self.apply_line_directive(hash_tok, [name_tok] + self.read_line())
            return
        if handler is None:
            if name in ("if", "else"):
                handler = getattr(self, "dir_" + name)
            else:
                diag.error(name_tok, f"invalid preprocessing directive #{name}")
        handler(hash_tok, name_tok)

    # ---- macros -------------------------------------------------------------
    def dir_define(self, hash_tok, name_tok):
        line = self.read_line()
        if not line or line[0].kind not in (TokenKind.IDENT, TokenKind.KEYWORD):
            diag.error(name_tok, "macro name must be an identifier")
        name = line[0].text
        if len(line) > 1 and line[1].text == "(" and not line[1].space:
            params, va_name, i = [], None, 2
            while True:
                if i >= len(line):
                    diag.error(line[0], "unterminated macro parameter list")
                t = line[i]
                if t.text == ")":
                    i += 1
                    break
                if params or va_name:
                    if t.text != ",":
                        diag.error(t, "expected ',' in macro parameter list")
                    i += 1
                    t = line[i]
                if t.text == "...":
                    va_name = "__VA_ARGS__"
                    i += 1
                    continue
                if t.kind != TokenKind.IDENT:
                    diag.error(t, "expected parameter name")
                if i + 1 < len(line) and line[i + 1].text == "...":
                    va_name = t.text
                    i += 2
                    continue
                params.append(t.text)
                i += 1
            self.macros[name] = Macro(name, False, params, va_name, line[i:])
        else:
            self.macros[name] = Macro(name, True, body=line[1:])

    def dir_undef(self, hash_tok, name_tok):
        line = self.read_line()
        if not line:
            diag.error(name_tok, "macro name missing")
        self.macros.pop(line[0].text, None)

    # ---- include ------------------------------------------------------------
    def dir_include(self, hash_tok, name_tok, next_=False):
        line = self.read_line()
        if line and line[0].kind not in (TokenKind.STRING,) and line[0].text != "<":
            line = self.expand_list(line)
        if not line:
            diag.error(name_tok, "#include expects \"FILENAME\" or <FILENAME>")
        if line[0].kind == TokenKind.STRING:
            fname, quoted = line[0].text[1:-1], True
        elif line[0].text == "<":
            parts = []
            for t in line[1:]:
                if t.text == ">":
                    break
                if t.space and parts:
                    parts.append(" ")
                parts.append(t.text)
            fname, quoted = "".join(parts), False
        else:
            diag.error(line[0], "#include expects \"FILENAME\" or <FILENAME>")
        path, idx = self.find_include(fname, quoted, hash_tok.loc.file, next_)
        if path is None:
            diag.error(hash_tok, f"'{fname}' file not found")
        if self.include_depth > 200:
            diag.error(hash_tok, "#include nested too deeply")
        self.include_file(path, idx)

    def dir_include_next(self, hash_tok, name_tok):
        self.dir_include(hash_tok, name_tok, next_=True)

    def dir_import(self, hash_tok, name_tok):
        self.dir_include(hash_tok, name_tok)

    # ---- conditionals ---------------------------------------------------------
    def dir_if(self, hash_tok, name_tok):
        val = self.eval_condition(hash_tok, self.read_line())
        self.cond_stack.append(CondIncl("if", hash_tok, val))
        if not val:
            self.skip_cond_incl()

    def dir_ifdef(self, hash_tok, name_tok, negate=False):
        line = self.read_line()
        if not line:
            diag.error(name_tok, "no macro name given")
        val = (line[0].text in self.macros) != negate
        self.cond_stack.append(CondIncl("if", hash_tok, val))
        if not val:
            self.skip_cond_incl()

    def dir_ifndef(self, hash_tok, name_tok):
        self.dir_ifdef(hash_tok, name_tok, negate=True)

    def dir_elif(self, hash_tok, name_tok):
        if not self.cond_stack or self.cond_stack[-1].ctx == "else":
            diag.error(hash_tok, "stray #elif")
        c = self.cond_stack[-1]
        c.ctx = "elif"
        line = self.read_line()
        if not c.included and self.eval_condition(hash_tok, line):
            c.included = True
        else:
            self.skip_cond_incl()

    def dir_elifdef(self, hash_tok, name_tok, negate=False):
        if not self.cond_stack or self.cond_stack[-1].ctx == "else":
            diag.error(hash_tok, "stray #elifdef")
        c = self.cond_stack[-1]
        c.ctx = "elif"
        line = self.read_line()
        if not c.included and line and (line[0].text in self.macros) != negate:
            c.included = True
        else:
            self.skip_cond_incl()

    def dir_elifndef(self, hash_tok, name_tok):
        self.dir_elifdef(hash_tok, name_tok, negate=True)

    def dir_else(self, hash_tok, name_tok):
        self.read_line()
        if not self.cond_stack or self.cond_stack[-1].ctx == "else":
            diag.error(hash_tok, "stray #else")
        c = self.cond_stack[-1]
        c.ctx = "else"
        if c.included:
            self.skip_cond_incl()
        c.included = True

    def dir_endif(self, hash_tok, name_tok):
        self.read_line()
        if not self.cond_stack:
            diag.error(hash_tok, "stray #endif")
        self.cond_stack.pop()

    def skip_cond_incl(self) -> None:
        """Skip tokens until the matching #elif/#else/#endif (left unread)."""
        stack = self.stack
        depth = 0
        while stack:
            t = stack.pop()
            if t.kind == TokenKind.EOF:
                diag.error(self.cond_stack[-1].tok, "unterminated conditional directive")
            if not (t.bol and t.text == "#" and t.kind == TokenKind.PUNCT):
                continue
            nxt = stack[-1]
            if nxt.bol:
                continue
            d = nxt.text
            if d in ("if", "ifdef", "ifndef"):
                depth += 1
            elif d in ("elif", "else", "elifdef", "elifndef") and depth == 0:
                stack.append(t)
                return
            elif d == "endif":
                if depth == 0:
                    stack.append(t)
                    return
                depth -= 1

    # ---- misc -----------------------------------------------------------------
    def dir_line(self, hash_tok, name_tok):
        self.apply_line_directive(hash_tok, self.read_line())

    def apply_line_directive(self, hash_tok, line: list[Token]) -> None:
        """Record a `#line N "name"` (or GNU `# N "name"`) for later lines."""
        if line and line[0].kind != TokenKind.STRING:
            line = self.expand_list(line)
        if not line or line[0].kind != TokenKind.NUMBER or not line[0].text.isdigit():
            diag.error(hash_tok, "invalid line number in #line directive")
        n = int(line[0].text)
        if len(line) > 1 and line[1].kind != TokenKind.STRING:
            diag.error(line[1], "invalid filename in #line directive")
        start = hash_tok.loc.line + 1          # first physical line affected
        name = line[1].text[1:-1] if len(line) > 1 else self.presumed(hash_tok.loc)[0]
        marks = self.line_marks.setdefault(hash_tok.loc.file, [])
        while marks and marks[-1][0] >= start:
            marks.pop()
        marks.append((start, n, name))

    def presumed(self, loc) -> tuple[str, int]:
        """(file, line) as seen by __FILE__/__LINE__ after any #line marks."""
        for start, n, name in reversed(self.line_marks.get(loc.file, ())):
            if start <= loc.line:
                return name, n + (loc.line - start)
        return loc.file, loc.line

    def dir_error(self, hash_tok, name_tok):
        msg = " ".join(t.text for t in self.read_line())
        diag.error(hash_tok, f"#error {msg}")

    def dir_warning(self, hash_tok, name_tok):
        msg = " ".join(t.text for t in self.read_line())
        diag.warn(hash_tok, f"#warning {msg}")

    def dir_pragma(self, hash_tok, name_tok):
        line = self.read_line()
        if line and line[0].text == "once":
            self.pragma_once.add(hash_tok.loc.file)
        elif line and line[0].text in ("push_macro", "pop_macro"):
            self._macro_pragma(line)

    def _macro_pragma(self, line: list[Token]) -> None:
        """#pragma push_macro("NAME") / pop_macro("NAME")."""
        if len(line) < 4 or line[1].text != "(" or line[3].text != ")":
            return
        if line[2].kind != TokenKind.STRING:
            return
        name = line[2].text[1:-1]
        saved = self.pushed_macros.setdefault(name, [])
        if line[0].text == "push_macro":
            saved.append(self.macros.get(name))
        elif saved:
            prev = saved.pop()
            if prev is None:
                self.macros.pop(name, None)
            else:
                self.macros[name] = prev

    def dir_ident(self, hash_tok, name_tok):
        self.read_line()

    dir_sccs = dir_ident
    dir_assert = dir_ident
    dir_unassert = dir_ident
