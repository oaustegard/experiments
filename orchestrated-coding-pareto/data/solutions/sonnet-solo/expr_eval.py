"""
Arithmetic expression evaluator.

Implements a hand-written tokenizer and recursive-descent parser for a small
arithmetic grammar. No `eval`, `exec`, `ast`, `compile`, or expression
libraries are used.

Grammar (low to high precedence):
    expr   := term (('+' | '-') term)*
    term   := unary (('*' | '/' | '%') unary)*
    unary  := ('+' | '-') unary | power
    power  := atom ('**' unary)?
    atom   := NUMBER | '(' expr ')'

Notes:
    - '**' is right-associative.
    - Unary minus binds looser than '**' on its left ("-2**2" == -(2**2))
      but a unary minus immediately after '**' binds tighter
      ("2**-2**2" == 2**(-(2**2))), achieved by having 'power' recurse into
      'unary' (not 'power') for its right-hand operand.
    - '/' is true division, '%' is Python modulo; division/modulo by zero
      raises ZeroDivisionError (not converted to ValueError).
"""

import re

_TOKEN_SPEC = [
    ("NUMBER", r"\d+\.\d*|\.\d+|\d+"),
    ("DSTAR", r"\*\*"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("PERCENT", r"%"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SKIP", r"[ \t]+"),
    ("MISMATCH", r"."),
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC),
    re.DOTALL,
)


def _tokenize(expr):
    tokens = []
    for match in _MASTER_RE.finditer(expr):
        kind = match.lastgroup
        text = match.group()
        if kind == "SKIP":
            continue
        if kind == "MISMATCH":
            raise ValueError(f"invalid character {text!r} in expression")
        tokens.append((kind, text))
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_expr(self):
        value = self.parse_term()
        while True:
            tok = self.peek()
            if tok is None or tok[0] not in ("PLUS", "MINUS"):
                break
            self.advance()
            rhs = self.parse_term()
            if tok[0] == "PLUS":
                value = value + rhs
            else:
                value = value - rhs
        return value

    def parse_term(self):
        value = self.parse_unary()
        while True:
            tok = self.peek()
            if tok is None or tok[0] not in ("STAR", "SLASH", "PERCENT"):
                break
            self.advance()
            rhs = self.parse_unary()
            if tok[0] == "STAR":
                value = value * rhs
            elif tok[0] == "SLASH":
                value = value / rhs
            else:
                value = value % rhs
        return value

    def parse_unary(self):
        tok = self.peek()
        if tok is not None and tok[0] in ("PLUS", "MINUS"):
            self.advance()
            operand = self.parse_unary()
            if tok[0] == "MINUS":
                return -operand
            return operand
        return self.parse_power()

    def parse_power(self):
        base = self.parse_atom()
        tok = self.peek()
        if tok is not None and tok[0] == "DSTAR":
            self.advance()
            exponent = self.parse_unary()
            return base ** exponent
        return base

    def parse_atom(self):
        tok = self.peek()
        if tok is None:
            raise ValueError("unexpected end of expression")
        kind, text = tok
        if kind == "NUMBER":
            self.advance()
            return float(text)
        if kind == "LPAREN":
            self.advance()
            value = self.parse_expr()
            close = self.peek()
            if close is None or close[0] != "RPAREN":
                raise ValueError("unbalanced parentheses")
            self.advance()
            return value
        raise ValueError(f"unexpected token {text!r} in expression")


def evaluate(expr: str) -> float:
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression is empty or whitespace-only")

    tokens = _tokenize(expr)
    if not tokens:
        raise ValueError("expression is empty or whitespace-only")

    parser = _Parser(tokens)
    result = parser.parse_expr()

    if parser.peek() is not None:
        leftover = parser.peek()[1]
        raise ValueError(f"unexpected trailing token {leftover!r}")

    return float(result)
