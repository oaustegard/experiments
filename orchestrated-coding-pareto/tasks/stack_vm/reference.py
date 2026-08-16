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


_KNOWN = {"PUSH", "POP", "DUP", "SWAP", "ADD", "SUB", "MUL", "DIV", "MOD", "NEG",
          "EQ", "LT", "GT", "LOAD", "STORE", "JMP", "JZ", "CALL", "RET", "HALT"}


def run(program, max_steps: int = 10_000):
    stack = []
    store = {}
    calls = []
    pc = 0
    steps = 0
    n = len(program)

    def pop():
        if not stack:
            raise StackUnderflow("operand stack empty")
        return stack.pop()

    def jump_to(target):
        if not (0 <= target <= n):
            raise BadJump(f"target {target} out of range")
        return target

    while pc < n:
        ins = program[pc]
        op = ins[0]
        if op not in _KNOWN:
            raise BadOpcode(op)
        steps += 1
        if steps > max_steps:
            raise StepLimitExceeded(f"exceeded {max_steps} steps")
        nxt = pc + 1
        if op == "PUSH":
            stack.append(ins[1])
        elif op == "POP":
            pop()
        elif op == "DUP":
            a = pop()
            stack.append(a)
            stack.append(a)
        elif op == "SWAP":
            b, a = pop(), pop()
            stack.append(b)
            stack.append(a)
        elif op in ("ADD", "SUB", "MUL", "DIV", "MOD", "EQ", "LT", "GT"):
            b, a = pop(), pop()
            if op == "ADD":
                stack.append(a + b)
            elif op == "SUB":
                stack.append(a - b)
            elif op == "MUL":
                stack.append(a * b)
            elif op == "DIV":
                if b == 0:
                    raise VMZeroDivision("DIV by zero")
                stack.append(a // b)
            elif op == "MOD":
                if b == 0:
                    raise VMZeroDivision("MOD by zero")
                stack.append(a % b)
            elif op == "EQ":
                stack.append(1 if a == b else 0)
            elif op == "LT":
                stack.append(1 if a < b else 0)
            else:
                stack.append(1 if a > b else 0)
        elif op == "NEG":
            stack.append(-pop())
        elif op == "LOAD":
            stack.append(store.get(ins[1], 0))
        elif op == "STORE":
            store[ins[1]] = pop()
        elif op == "JMP":
            nxt = jump_to(ins[1])
        elif op == "JZ":
            a = pop()
            if a == 0:
                nxt = jump_to(ins[1])
        elif op == "CALL":
            calls.append(pc + 1)
            nxt = jump_to(ins[1])
        elif op == "RET":
            if not calls:
                raise StackUnderflow("call stack empty")
            nxt = jump_to(calls.pop())
        elif op == "HALT":
            return stack
        pc = nxt
    return stack
