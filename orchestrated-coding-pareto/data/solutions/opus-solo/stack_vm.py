"""A small stack-based virtual machine.

Public API:
    run(program, max_steps=10_000) -> list[int]
    VMError, StackUnderflow, BadJump, BadOpcode, StepLimitExceeded, VMZeroDivision
"""

__all__ = [
    "VMError",
    "StackUnderflow",
    "BadJump",
    "BadOpcode",
    "StepLimitExceeded",
    "VMZeroDivision",
    "run",
]


class VMError(Exception):
    """Base class for all VM errors."""


class StackUnderflow(VMError):
    """Popping or peeking an empty operand stack (or an empty call stack)."""


class BadJump(VMError):
    """Jump/call/return target out of range."""


class BadOpcode(VMError):
    """Unknown instruction name."""


class StepLimitExceeded(VMError):
    """Executed more than max_steps instructions."""


class VMZeroDivision(VMError):
    """DIV or MOD with a zero divisor."""


_NULLARY = frozenset(
    {
        "POP",
        "DUP",
        "SWAP",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MOD",
        "NEG",
        "EQ",
        "LT",
        "GT",
        "RET",
        "HALT",
    }
)

_UNARY = frozenset({"PUSH", "LOAD", "STORE", "JMP", "JZ", "CALL"})

_OPCODES = _NULLARY | _UNARY


def run(program, max_steps=10_000):
    """Execute `program` from index 0 and return the final operand stack."""
    stack = []
    store = {}
    calls = []
    pc = 0
    steps = 0

    n = len(program)

    def pop():
        if not stack:
            raise StackUnderflow("pop from empty operand stack")
        return stack.pop()

    def check_target(target):
        if not isinstance(target, int) or isinstance(target, bool):
            raise BadJump("jump target is not an integer: %r" % (target,))
        if target < 0 or target > n:
            raise BadJump("jump target out of range: %r" % (target,))
        return target

    while pc != n:
        if pc < 0 or pc > n:
            raise BadJump("program counter out of range: %r" % (pc,))

        if steps >= max_steps:
            raise StepLimitExceeded(
                "exceeded max_steps=%r" % (max_steps,)
            )
        steps += 1

        instr = program[pc]
        if not instr:
            raise BadOpcode("empty instruction at %d" % (pc,))
        op = instr[0]
        arg = instr[1] if len(instr) > 1 else None

        if op not in _OPCODES:
            raise BadOpcode("unknown opcode: %r" % (op,))

        if op == "PUSH":
            stack.append(arg)
            pc += 1
        elif op == "POP":
            pop()
            pc += 1
        elif op == "DUP":
            if not stack:
                raise StackUnderflow("DUP on empty operand stack")
            stack.append(stack[-1])
            pc += 1
        elif op == "SWAP":
            if len(stack) < 2:
                raise StackUnderflow("SWAP needs two operands")
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
                raise VMZeroDivision("DIV by zero")
            stack.append(a // b)
            pc += 1
        elif op == "MOD":
            b = pop()
            a = pop()
            if b == 0:
                raise VMZeroDivision("MOD by zero")
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
            pc = check_target(arg)
        elif op == "JZ":
            a = pop()
            if a == 0:
                pc = check_target(arg)
            else:
                pc += 1
        elif op == "CALL":
            target = check_target(arg)
            calls.append(pc + 1)
            pc = target
        elif op == "RET":
            if not calls:
                raise StackUnderflow("RET with empty call stack")
            pc = check_target(calls.pop())
        elif op == "HALT":
            return stack

    return stack
