"""#if / #elif constant-expression evaluation.

ConditionalMixin is mixed into Preprocessor and provides eval_condition(line).
Handles `defined X`, `defined(X)`, __has_include, __has_include_next,
__has_builtin, __has_attribute (0), __has_feature (0), then macro-expands,
maps remaining identifiers to 0 and evaluates with intmax_t/uintmax_t rules.
"""
from __future__ import annotations

from hcc import diag
from hcc.tokens import Token, TokenKind

MASK64 = (1 << 64) - 1

# builtins that __has_builtin reports as available (keep in sync with codegen/builtins.py)
KNOWN_BUILTINS = frozenset("""
__builtin_va_start __builtin_va_end __builtin_va_arg __builtin_va_copy __builtin_expect
__builtin_offsetof __builtin_types_compatible_p __builtin_constant_p __builtin_unreachable
__builtin_trap __builtin_alloca
""".split())


def _wrap(v: int, unsigned: bool) -> int:
    v &= MASK64
    if not unsigned and v >= 1 << 63:
        v -= 1 << 64
    return v


class _Eval:
    """Recursive-descent evaluator over a token list. Values are (int, unsigned)."""

    def __init__(self, toks: list[Token], where: Token):
        self.toks = toks
        self.i = 0
        self.where = where
        self.skip = 0  # >0 while evaluating an unevaluated operand

    def peek(self) -> str:
        return self.toks[self.i].text if self.i < len(self.toks) else ""

    def take(self) -> Token:
        if self.i >= len(self.toks):
            diag.error(self.where, "#if: unexpected end of expression")
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect(self, s: str):
        t = self.take()
        if t.text != s:
            diag.error(t, f"#if: expected '{s}'")

    def run(self) -> int:
        v, _ = self.cond()
        if self.i != len(self.toks):
            diag.error(self.toks[self.i], f"#if: unexpected token '{self.toks[self.i].text}'")
        return v

    def cond(self):
        c = self.binary(0)
        if self.peek() == "?":
            self.take()
            self.skip += (not c[0])
            a = self.cond()
            self.skip -= (not c[0])
            self.expect(":")
            self.skip += bool(c[0])
            b = self.cond()
            self.skip -= bool(c[0])
            u = a[1] or b[1]
            return (_wrap((a if c[0] else b)[0], u), u)
        return c

    PREC = [["||"], ["&&"], ["|"], ["^"], ["&"], ["==", "!="], ["<", "<=", ">", ">="],
            ["<<", ">>"], ["+", "-"], ["*", "/", "%"]]

    def binary(self, level):
        if level == len(self.PREC):
            return self.unary()
        lhs = self.binary(level + 1)
        while self.peek() in self.PREC[level] and self.i < len(self.toks) and self.toks[self.i].kind == TokenKind.PUNCT:
            op = self.take().text
            if op in ("&&", "||"):
                short = (not lhs[0]) if op == "&&" else bool(lhs[0])
                self.skip += short
                rhs = self.binary(level + 1)
                self.skip -= short
                r = bool(rhs[0]) if not short else False
                lhs = (int(bool(lhs[0]) and r) if op == "&&" else int(bool(lhs[0]) or r), False)
                continue
            rhs = self.binary(level + 1)
            lhs = self.apply(op, lhs, rhs)
        return lhs

    def apply(self, op, a, b):
        u = a[1] or b[1]
        x, y = (a[0] & MASK64, b[0] & MASK64) if u else (a[0], b[0])
        if op in ("<<", ">>"):
            u = a[1]
            x = a[0] & MASK64 if u else a[0]
            return (_wrap(x << y if op == "<<" else x >> y, u), u)
        if op in ("/", "%"):
            if y == 0 and self.skip:
                return (0, u)
            if y == 0:
                diag.error(self.where, "#if: division by zero")
            q = abs(x) // abs(y) * (1 if (x >= 0) == (y >= 0) else -1)
            return (_wrap(q if op == "/" else x - q * y, u), u)
        r = {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y, "&": lambda: x & y,
             "|": lambda: x | y, "^": lambda: x ^ y, "==": lambda: int(x == y),
             "!=": lambda: int(x != y), "<": lambda: int(x < y), "<=": lambda: int(x <= y),
             ">": lambda: int(x > y), ">=": lambda: int(x >= y)}[op]()
        if op in ("==", "!=", "<", "<=", ">", ">="):
            return (r, False)
        return (_wrap(r, u), u)

    def unary(self):
        t = self.take()
        if t.kind == TokenKind.PUNCT:
            if t.text == "(":
                v = self.cond()
                self.expect(")")
                return v
            if t.text == "!":
                v = self.unary()
                return (int(not v[0]), False)
            if t.text == "-":
                v = self.unary()
                return (_wrap(-v[0], v[1]), v[1])
            if t.text == "+":
                return self.unary()
            if t.text == "~":
                v = self.unary()
                return (_wrap(~v[0], v[1]), v[1])
        if t.kind == TokenKind.NUMBER:
            from hcc.literals import parse_int_literal
            val, ty = parse_int_literal(t)
            return (_wrap(val, ty.is_unsigned and ty.size == 8), ty.is_unsigned and ty.size == 8)
        if t.kind == TokenKind.CHAR:
            from hcc.literals import parse_char_literal
            val, _ = parse_char_literal(t)
            # u8'x' and U'x' are unsigned in #if (C23 / gcc), plain and L/u are signed
            return (val, t.text.startswith(("u8", "U")))
        if t.kind == TokenKind.IDENT:
            return (0, False)
        diag.error(t, f"#if: unexpected token '{t.text}'")


class ConditionalMixin:

    def eval_condition(self, hash_tok: Token, line: list[Token]) -> bool:
        toks = self._replace_defined(line)
        toks = self.expand_list(toks)
        toks = self._replace_defined(toks)
        if not toks:
            diag.error(hash_tok, "#if with no expression")
        return _Eval(toks, hash_tok).run() != 0

    def _one(self, tok: Token, v: int) -> Token:
        return Token(TokenKind.NUMBER, str(v), tok.loc, False, True)

    def _replace_defined(self, line: list[Token]) -> list[Token]:
        out = []
        i = 0
        n = len(line)
        while i < n:
            t = line[i]
            if t.kind != TokenKind.IDENT:
                out.append(t)
                i += 1
                continue
            name = t.text
            if name == "defined":
                i += 1
                paren = i < n and line[i].text == "("
                if paren:
                    i += 1
                if i >= n or line[i].kind not in (TokenKind.IDENT, TokenKind.KEYWORD):
                    diag.error(t, "macro name must be an identifier")
                d = line[i].text
                v = d in self.macros or d in ("__has_include", "__has_include_next", "__has_builtin",
                                              "__has_attribute", "__has_feature", "__has_extension",
                                              "__has_c_attribute", "__has_cpp_attribute")
                i += 1
                if paren:
                    if i >= n or line[i].text != ")":
                        diag.error(t, "expected ')' after defined")
                    i += 1
                out.append(self._one(t, int(v)))
                continue
            if name in ("__has_include", "__has_include_next", "__has_builtin", "__has_attribute",
                        "__has_feature", "__has_extension", "__has_c_attribute", "__has_cpp_attribute") \
                    and i + 1 < n and line[i + 1].text == "(":
                j = i + 2
                depth = 1
                arg = []
                while j < n:
                    if line[j].text == "(":
                        depth += 1
                    elif line[j].text == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    arg.append(line[j])
                    j += 1
                i = j + 1
                out.append(self._one(t, int(self._has(name, arg, t))))
                continue
            out.append(t)
            i += 1
        return out

    def _has(self, name: str, arg: list[Token], t: Token) -> bool:
        if name.startswith("__has_include"):
            if arg and arg[0].kind == TokenKind.STRING:
                fname, quoted = arg[0].text[1:-1], True
            else:
                arg = self.expand_list(arg)
                if arg and arg[0].kind == TokenKind.STRING:
                    fname, quoted = arg[0].text[1:-1], True
                else:
                    fname, quoted = "".join(x.text for x in arg[1:-1]), False
            path, _ = self.find_include(fname, quoted, t.loc.file, name.endswith("_next"))
            return path is not None
        if name == "__has_builtin":
            return bool(arg) and arg[0].text in KNOWN_BUILTINS
        return False
