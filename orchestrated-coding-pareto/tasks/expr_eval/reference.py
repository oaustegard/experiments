import re

_TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+\.?|\d*\.\d+)|(\*\*|[+\-*/%()]))")


def _tokenize(s: str):
    tokens = []
    i = 0
    while i < len(s):
        m = _TOKEN.match(s, i)
        if not m:
            if s[i:].strip() == "":
                break
            raise ValueError(f"bad character at {i}: {s[i]!r}")
        num, op = m.group(1), m.group(2)
        if num is not None:
            # reject things like "1.2.3" -- next char must not be a dot/digit
            j = m.end()
            if j < len(s) and s[j] == ".":
                raise ValueError(f"malformed number near {num!r}")
            tokens.append(("num", float(num)))
        else:
            tokens.append(("op", op))
        i = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        if t is None:
            raise ValueError("unexpected end of expression")
        self.i += 1
        return t

    # expr := term (('+'|'-') term)*
    def expr(self):
        v = self.term()
        while (t := self.peek()) and t == ("op", "+") or (t and t == ("op", "-")):
            self.next()
            rhs = self.term()
            v = v + rhs if t[1] == "+" else v - rhs
        return v

    # term := unary (('*'|'/'|'%') unary)*
    def term(self):
        v = self.unary()
        while (t := self.peek()) and t[0] == "op" and t[1] in ("*", "/", "%"):
            self.next()
            rhs = self.unary()
            if t[1] == "*":
                v = v * rhs
            elif t[1] == "/":
                if rhs == 0:
                    raise ZeroDivisionError("division by zero")
                v = v / rhs
            else:
                if rhs == 0:
                    raise ZeroDivisionError("modulo by zero")
                v = v % rhs
        return v

    # unary := ('+'|'-')* power ; -x**y == -(x**y)
    def unary(self):
        t = self.peek()
        if t and t[0] == "op" and t[1] in ("+", "-"):
            self.next()
            v = self.unary()
            return -v if t[1] == "-" else v
        return self.power()

    # power := atom ('**' unary)?   (right assoc; rhs allows unary minus)
    def power(self):
        v = self.atom()
        t = self.peek()
        if t == ("op", "**"):
            self.next()
            rhs = self.unary()
            return v ** rhs
        return v

    def atom(self):
        t = self.next()
        if t[0] == "num":
            return t[1]
        if t == ("op", "("):
            v = self.expr()
            if self.peek() != ("op", ")"):
                raise ValueError("unbalanced parentheses")
            self.next()
            return v
        raise ValueError(f"unexpected token {t[1]!r}")


def evaluate(expr: str) -> float:
    tokens = _tokenize(expr)
    if not tokens:
        raise ValueError("empty expression")
    p = _Parser(tokens)
    v = p.expr()
    if p.peek() is not None:
        raise ValueError(f"trailing input: {p.peek()!r}")
    return float(v)
