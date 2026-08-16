class VMError(Exception):
    pass


class StackUnderflow(VMError):
    pass


class BadJump(VMError):
    pass


class BadOpcode(VMError):
    pass


class StepLimitExceeded(VMError):
    pass


class VMZeroDivision(VMError):
    pass


def run(program: list[tuple], max_steps: int = 10_000) -> list[int]:
    operand_stack = []
    store = {}
    call_stack = []
    pc = 0
    steps = 0

    while True:
        if steps >= max_steps:
            raise StepLimitExceeded()

        if pc == len(program):
            return operand_stack

        if pc < 0 or pc > len(program):
            raise BadJump()

        instruction = program[pc]
        op = instruction[0]
        arg = instruction[1] if len(instruction) > 1 else None

        steps += 1
        pc += 1

        if op == "PUSH":
            operand_stack.append(arg)
        elif op == "POP":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            operand_stack.pop()
        elif op == "DUP":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            operand_stack.append(operand_stack[-1])
        elif op == "SWAP":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            operand_stack[-1], operand_stack[-2] = operand_stack[-2], operand_stack[-1]
        elif op == "ADD":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(a + b)
        elif op == "SUB":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(a - b)
        elif op == "MUL":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(a * b)
        elif op == "DIV":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            if b == 0:
                raise VMZeroDivision()
            operand_stack.append(a // b)
        elif op == "MOD":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            if b == 0:
                raise VMZeroDivision()
            operand_stack.append(a % b)
        elif op == "NEG":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            a = operand_stack.pop()
            operand_stack.append(-a)
        elif op == "EQ":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(1 if a == b else 0)
        elif op == "LT":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(1 if a < b else 0)
        elif op == "GT":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(1 if a > b else 0)
        elif op == "LOAD":
            operand_stack.append(store.get(arg, 0))
        elif op == "STORE":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            a = operand_stack.pop()
            store[arg] = a
        elif op == "JMP":
            if arg < 0 or arg > len(program):
                raise BadJump()
            pc = arg
        elif op == "JZ":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            a = operand_stack.pop()
            if a == 0:
                if arg < 0 or arg > len(program):
                    raise BadJump()
                pc = arg
        elif op == "CALL":
            if arg < 0 or arg > len(program):
                raise BadJump()
            call_stack.append(pc)
            pc = arg
        elif op == "RET":
            if len(call_stack) < 1:
                raise StackUnderflow()
            pc = call_stack.pop()
        elif op == "HALT":
            return operand_stack
        else:
            raise BadOpcode()
