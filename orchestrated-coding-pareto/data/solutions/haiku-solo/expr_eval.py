def evaluate(expr: str) -> float:
    """Evaluate an arithmetic expression string."""

    # Check for empty expression
    if not expr or expr.isspace():
        raise ValueError("Empty expression")

    # Tokenize
    tokens = []
    i = 0
    while i < len(expr):
        # Skip whitespace
        if expr[i] in ' \t':
            i += 1
            continue

        # Numbers
        if expr[i].isdigit() or (expr[i] == '.' and i + 1 < len(expr) and expr[i + 1].isdigit()):
            start = i
            if expr[i] == '.':
                # .5 format
                i += 1
                while i < len(expr) and expr[i].isdigit():
                    i += 1
            else:
                # 10 or 10.5 format
                while i < len(expr) and expr[i].isdigit():
                    i += 1
                if i < len(expr) and expr[i] == '.':
                    i += 1
                    while i < len(expr) and expr[i].isdigit():
                        i += 1
            tokens.append(('NUMBER', float(expr[start:i])))
            continue

        # Two-character operators
        if i + 1 < len(expr) and expr[i:i + 2] == '**':
            tokens.append(('**', None))
            i += 2
            continue

        # Single-character operators and parentheses
        if expr[i] in '+-*/%()':
            tokens.append((expr[i], None))
            i += 1
            continue

        # Invalid character
        raise ValueError("Invalid character")

    # Parse
    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0

        def current_token(self):
            return self.tokens[self.pos] if self.pos < len(self.tokens) else None

        def consume(self):
            token = self.current_token()
            self.pos += 1
            return token

        def expect(self, token_type):
            token = self.current_token()
            if token is None or token[0] != token_type:
                raise ValueError("Unexpected token")
            return self.consume()

        def parse(self):
            result = self.parse_expression()
            if self.current_token() is not None:
                raise ValueError("Unexpected tokens")
            return result

        def parse_expression(self):
            left = self.parse_term()
            while self.current_token() and self.current_token()[0] in ('+', '-'):
                op = self.consume()[0]
                right = self.parse_term()
                left = left + right if op == '+' else left - right
            return left

        def parse_term(self):
            left = self.parse_factor()
            while self.current_token() and self.current_token()[0] in ('*', '/', '%'):
                op = self.consume()[0]
                right = self.parse_factor()
                if op == '*':
                    left = left * right
                elif op == '/':
                    left = left / right
                else:
                    left = left % right
            return left

        def parse_factor(self):
            if self.current_token() and self.current_token()[0] in ('+', '-'):
                op = self.consume()[0]
                value = self.parse_factor()
                return -value if op == '-' else value
            return self.parse_power()

        def parse_power(self):
            base = self.parse_atom()
            if self.current_token() and self.current_token()[0] == '**':
                self.consume()
                exponent = self.parse_factor()
                return base ** exponent
            return base

        def parse_atom(self):
            token = self.current_token()
            if token is None:
                raise ValueError("Unexpected end of expression")

            if token[0] == 'NUMBER':
                return self.consume()[1]
            elif token[0] == '(':
                self.consume()
                if self.current_token() and self.current_token()[0] == ')':
                    raise ValueError("Empty parentheses")
                value = self.parse_expression()
                self.expect(')')
                return value
            else:
                raise ValueError("Unexpected token")

    parser = Parser(tokens)
    return parser.parse()
