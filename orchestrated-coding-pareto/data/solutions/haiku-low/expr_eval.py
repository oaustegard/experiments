def evaluate(expr: str) -> float:
    expr = expr.strip()

    if not expr:
        raise ValueError("Empty expression")

    tokens = tokenize(expr)

    if not tokens:
        raise ValueError("Empty expression")

    parser = Parser(tokens)
    result = parser.parse_expr()

    if parser.pos < len(tokens):
        raise ValueError("Extra tokens after expression")

    return result


def tokenize(expr: str):
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i] in ' \t':
            i += 1
            continue

        if expr[i].isdigit():
            j = i
            has_dot = False
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                if expr[j] == '.':
                    if has_dot:
                        raise ValueError("Malformed number")
                    has_dot = True
                j += 1
            tokens.append(('NUMBER', float(expr[i:j])))
            i = j
        elif expr[i] == '.':
            if i + 1 < len(expr) and expr[i + 1].isdigit():
                j = i + 1
                while j < len(expr) and expr[j].isdigit():
                    j += 1
                tokens.append(('NUMBER', float(expr[i:j])))
                i = j
            else:
                raise ValueError("Invalid character")
        elif expr[i] == '+':
            tokens.append(('PLUS', '+'))
            i += 1
        elif expr[i] == '-':
            tokens.append(('MINUS', '-'))
            i += 1
        elif expr[i] == '*':
            if i + 1 < len(expr) and expr[i + 1] == '*':
                tokens.append(('POWER', '**'))
                i += 2
            else:
                tokens.append(('STAR', '*'))
                i += 1
        elif expr[i] == '/':
            tokens.append(('SLASH', '/'))
            i += 1
        elif expr[i] == '%':
            tokens.append(('PERCENT', '%'))
            i += 1
        elif expr[i] == '(':
            tokens.append(('LPAREN', '('))
            i += 1
        elif expr[i] == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
        else:
            raise ValueError("Invalid character")

    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        token = self.peek()
        self.pos += 1
        return token

    def expect(self, token_type):
        token = self.peek()
        if token is None or token[0] != token_type:
            raise ValueError(f"Expected {token_type}")
        return self.consume()

    def parse_expr(self):
        left = self.parse_term()

        while self.peek() and self.peek()[0] in ('PLUS', 'MINUS'):
            op = self.consume()[0]
            right = self.parse_term()
            if op == 'PLUS':
                left = left + right
            else:
                left = left - right

        return left

    def parse_term(self):
        left = self.parse_unary_minus()

        while self.peek() and self.peek()[0] in ('STAR', 'SLASH', 'PERCENT'):
            op = self.consume()[0]
            right = self.parse_unary_minus()
            if op == 'STAR':
                left = left * right
            elif op == 'SLASH':
                left = left / right
            else:
                left = left % right

        return left

    def parse_unary_minus(self):
        sign = 1
        while self.peek() and self.peek()[0] in ('PLUS', 'MINUS'):
            op = self.consume()[0]
            if op == 'MINUS':
                sign = -sign

        result = self.parse_power()
        return sign * result

    def parse_power(self):
        left = self.parse_atom()

        if self.peek() and self.peek()[0] == 'POWER':
            self.consume()
            right = self.parse_unary_minus()
            left = left ** right

        return left

    def parse_atom(self):
        token = self.peek()

        if token is None:
            raise ValueError("Unexpected end of expression")

        if token[0] == 'NUMBER':
            self.consume()
            return token[1]
        elif token[0] == 'LPAREN':
            self.consume()

            if self.peek() and self.peek()[0] == 'RPAREN':
                raise ValueError("Empty parentheses")

            result = self.parse_expr()
            self.expect('RPAREN')
            return result
        else:
            raise ValueError("Unexpected token")
