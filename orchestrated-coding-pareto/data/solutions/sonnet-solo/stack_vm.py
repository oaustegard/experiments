"""A small stack-based virtual machine."""


class VMError(Exception):
    """Base class for all VM errors."""


class StackUnderflow(VMError):
    """Popping/peeking an empty operand stack."""


class BadJump(VMError):
    """Jump/call target out of range or into nowhere."""


class BadOpcode(VMError):
    """Unknown instruction name."""


class StepLimitExceeded(VMError):
    """Executed more than max_steps instructions."""


class VMZeroDivision(VMError):
    """DIV or MOD with zero divisor."""


def run(program: list, max_steps: int = 10_000) -> list:
    stack = []
    store = {}
    call_stack = []
    pc = 0
    steps = 0
    n = len(program)

    def pop():
        if not stack:
            raise StackUnderflow()
        return stack.pop()

    def check_target(target):
        if target < 0 or target > n:
            raise BadJump()

    while pc < n:
        steps += 1
        if steps > max_steps:
            raise StepLimitExceeded()

        instr = program[pc]
        op = instr[0]
        arg = instr[1] if len(instr) > 1 else None

        if op == "PUSH":
            stack.append(arg)
            pc += 1
        elif op == "POP":
            pop()
            pc += 1
        elif op == "DUP":
            if not stack:
                raise StackUnderflow()
            stack.append(stack[-1])
            pc += 1
        elif op == "SWAP":
            if len(stack) < 2:
                raise StackUnderflow()
            stack[-1], stack[-2] = stack[-2], stack[-1]
            pc += 1
        elif op == "ADD":
            b = pop()
            a = pop()
            stack.append(a + b)
            pc += 1
        elif op == "SUB":
            b = pop()
            a = pop()
            stack.append(a - b)
            pc += 1
        elif op == "MUL":
            b = pop()
            a = pop()
            stack.append(a * b)
            pc += 1
        elif op == "DIV":
            b = pop()
            a = pop()
            if b == 0:
                raise VMZeroDivision()
            stack.append(a // b)
            pc += 1
        elif op == "MOD":
            b = pop()
            a = pop()
            if b == 0:
                raise VMZeroDivision()
            stack.append(a % b)
            pc += 1
        elif op == "NEG":
            a = pop()
            stack.append(-a)
            pc += 1
        elif op == "EQ":
            b = pop()
            a = pop()
            stack.append(1 if a == b else 0)
            pc += 1
        elif op == "LT":
            b = pop()
            a = pop()
            stack.append(1 if a < b else 0)
            pc += 1
        elif op == "GT":
            b = pop()
            a = pop()
            stack.append(1 if a > b else 0)
            pc += 1
        elif op == "LOAD":
            stack.append(store.get(arg, 0))
            pc += 1
        elif op == "STORE":
            a = pop()
            store[arg] = a
            pc += 1
        elif op == "JMP":
            check_target(arg)
            pc = arg
        elif op == "JZ":
            a = pop()
            if a == 0:
                check_target(arg)
                pc = arg
            else:
                pc += 1
        elif op == "CALL":
            check_target(arg)
            call_stack.append(pc + 1)
            pc = arg
        elif op == "RET":
            if not call_stack:
                raise StackUnderflow()
            pc = call_stack.pop()
        elif op == "HALT":
            return stack
        else:
            raise BadOpcode()

    return stack
