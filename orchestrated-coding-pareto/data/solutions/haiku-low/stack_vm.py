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

    while pc < len(program):
        steps += 1
        if steps > max_steps:
            raise StepLimitExceeded()

        instruction = program[pc]
        op = instruction[0]

        if op == "PUSH":
            operand_stack.append(instruction[1])
            pc += 1

        elif op == "POP":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            operand_stack.pop()
            pc += 1

        elif op == "DUP":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            operand_stack.append(operand_stack[-1])
            pc += 1

        elif op == "SWAP":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            operand_stack[-1], operand_stack[-2] = operand_stack[-2], operand_stack[-1]
            pc += 1

        elif op == "ADD":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(a + b)
            pc += 1

        elif op == "SUB":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(a - b)
            pc += 1

        elif op == "MUL":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(a * b)
            pc += 1

        elif op == "DIV":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            if b == 0:
                raise VMZeroDivision()
            operand_stack.append(a // b)
            pc += 1

        elif op == "MOD":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            if b == 0:
                raise VMZeroDivision()
            operand_stack.append(a % b)
            pc += 1

        elif op == "NEG":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            operand_stack.append(-operand_stack.pop())
            pc += 1

        elif op == "EQ":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(1 if a == b else 0)
            pc += 1

        elif op == "LT":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(1 if a < b else 0)
            pc += 1

        elif op == "GT":
            if len(operand_stack) < 2:
                raise StackUnderflow()
            b = operand_stack.pop()
            a = operand_stack.pop()
            operand_stack.append(1 if a > b else 0)
            pc += 1

        elif op == "LOAD":
            operand_stack.append(store.get(instruction[1], 0))
            pc += 1

        elif op == "STORE":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            store[instruction[1]] = operand_stack.pop()
            pc += 1

        elif op == "JMP":
            target = instruction[1]
            if target < 0 or target > len(program):
                raise BadJump()
            pc = target

        elif op == "JZ":
            if len(operand_stack) < 1:
                raise StackUnderflow()
            a = operand_stack.pop()
            if a == 0:
                target = instruction[1]
                if target < 0 or target > len(program):
                    raise BadJump()
                pc = target
            else:
                pc += 1

        elif op == "CALL":
            target = instruction[1]
            if target < 0 or target > len(program):
                raise BadJump()
            call_stack.append(pc + 1)
            pc = target

        elif op == "RET":
            if len(call_stack) < 1:
                raise StackUnderflow()
            pc = call_stack.pop()

        elif op == "HALT":
            return operand_stack

        else:
            raise BadOpcode()

    return operand_stack
