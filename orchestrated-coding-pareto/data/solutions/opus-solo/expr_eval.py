"""Arithmetic expression evaluator.

Implements a hand-written tokenizer and recursive-descent parser reproducing
Python's arithmetic semantics for + - * / % ** with parentheses and stacked
unary +/-.  No use of eval/exec/ast/compile.
"""

import re

__all__ = ["evaluate"]

_NUMBER_RE = re.compile(r"\d+\.\d*|\.\d+|\d+")
_SKIP = " \t\r\n\f\v"

# Token kinds
_NUM = "NUM"
_OP = "OP"
_LPAR = "LPAR"
_RPAR = "RPAR"


def _tokenize(expr):
    """Turn *expr* into a list of (kind, value) tuples.

    Raises ValueError on any character that cannot start a valid token.
    """
    if not isinstance(expr, str):
        raise ValueError("expression must be a string")

    tokens = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch in _SKIP:
            i += 1
            continue
        if ch == "(":
            tokens.append((_LPAR, "("))
            i += 1
            continue
        if ch == ")":
            tokens.append((_RPAR, ")"))
            i += 1
            continue
        if ch == "*":
            if expr.startswith("**", i):
                tokens.append((_OP, "**"))
                i += 2
            else:
                tokens.append((_OP, "*"))
                i += 1
            continue
        if ch in "+-/%":
            tokens.append((_OP, ch))
            i += 1
            continue
        if ch.isdigit() or ch == ".":
            m = _NUMBER_RE.match(expr, i)
            if m is None:
                raise ValueError("malformed number at position %d" % i)
            text = m.group(0)
            try:
                value = float(text)
            except (ValueError, OverflowError):
                raise ValueError("malformed number %r" % text)
            tokens.append((_NUM, value))
            i = m.end()
            continue
        raise ValueError("invalid character %r at position %d" % (ch, i))

    return tokens


class _Parser(object):
    """Recursive-descent parser.

    Grammar (lowest to highest precedence)::

        expr   := term (('+' | '-') term)*
        term   := power (('*' | '/' | '%') power)*   -- with unary handled below
        unary  := ('+' | '-') unary | power
        power  := atom ('**' unary)?
        atom   := NUMBER | '(' expr ')'

    ``term`` actually parses ``unary`` operands, which makes unary minus bind
    tighter than ``*``/``/``/``%`` but looser than ``**`` on its left, while
    ``power``'s right operand being ``unary`` allows ``2**-1``.
    """

    def __init__(self, tokens):
        self._tokens = tokens
        self._pos = 0

    # -- token helpers -------------------------------------------------
    def _peek(self):
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return (None, None)

    def _advance(self):
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _at_end(self):
        return self._pos >= len(self._tokens)

    # -- grammar rules -------------------------------------------------
    def parse(self):
        if not self._tokens:
            raise ValueError("empty expression")
        value = self._parse_expr()
        if not self._at_end():
            kind, val = self._peek()
            if kind == _RPAR:
                raise ValueError("unbalanced parentheses: unexpected ')'")
            raise ValueError("unexpected token %r" % (val,))
        return value

    def _parse_expr(self):
        value = self._parse_term()
        while True:
            kind, val = self._peek()
            if kind == _OP and val in ("+", "-"):
                self._advance()
                right = self._parse_term()
                if val == "+":
                    value = value + right
                else:
                    value = value - right
            else:
                return value

    def _parse_term(self):
        value = self._parse_unary()
        while True:
            kind, val = self._peek()
            if kind == _OP and val in ("*", "/", "%"):
                self._advance()
                right = self._parse_unary()
                if val == "*":
                    value = value * right
                elif val == "/":
                    if right == 0.0:
                        raise ZeroDivisionError("float division by zero")
                    value = value / right
                else:
                    if right == 0.0:
                        raise ZeroDivisionError("float modulo by zero")
                    value = value % right
            else:
                return value

    def _parse_unary(self):
        kind, val = self._peek()
        if kind == _OP and val in ("+", "-"):
            self._advance()
            operand = self._parse_unary()
            return -operand if val == "-" else +operand
        return self._parse_power()

    def _parse_power(self):
        base = self._parse_atom()
        kind, val = self._peek()
        if kind == _OP and val == "**":
            self._advance()
            # Right operand parsed as ``unary`` => right-associative and
            # permits a unary sign directly after '**'.
            exponent = self._parse_unary()
            return base ** exponent
        return base

    def _parse_atom(self):
        if self._at_end():
            raise ValueError("unexpected end of expression")
        kind, val = self._advance()
        if kind == _NUM:
            return val
        if kind == _LPAR:
            if self._at_end():
                raise ValueError("unbalanced parentheses: missing ')'")
            if self._peek()[0] == _RPAR:
                raise ValueError("empty parentheses")
            value = self._parse_expr()
            if self._at_end() or self._peek()[0] != _RPAR:
                raise ValueError("unbalanced parentheses: missing ')'")
            self._advance()
            return value
        if kind == _RPAR:
            raise ValueError("unbalanced parentheses: unexpected ')'")
        raise ValueError("unexpected operator %r" % (val,))


def evaluate(expr):
    """Evaluate the arithmetic expression *expr* and return a float.

    Supports ``+ - * / % **``, parentheses, stacked unary ``+``/``-`` and
    integer/decimal literals.  Raises ``ValueError`` for malformed input and
    ``ZeroDivisionError`` for division or modulo by zero.
    """
    tokens = _tokenize(expr)
    if not tokens:
        raise ValueError("empty expression")
    result = _Parser(tokens).parse()
    return float(result)
