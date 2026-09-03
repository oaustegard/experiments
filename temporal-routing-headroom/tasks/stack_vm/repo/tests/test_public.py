import pytest
from solution import (run, VMError, StackUnderflow, BadJump, BadOpcode,
                      StepLimitExceeded, VMZeroDivision)


def test_push_arith():
    assert run([("PUSH", 7), ("PUSH", 3), ("SUB",)]) == [4]
    assert run([("PUSH", 2), ("PUSH", 3), ("ADD",)]) == [5]
    assert run([("PUSH", 4), ("PUSH", 5), ("MUL",)]) == [20]


def test_countdown_loop():
    # store[0]=5; loop: load, jz end, load, push 1, sub, store, jmp loop; end
    prog = [("PUSH", 5), ("STORE", 0),
            ("LOAD", 0), ("JZ", 9),
            ("LOAD", 0), ("PUSH", 1), ("SUB",), ("STORE", 0),
            ("JMP", 2),
            ("LOAD", 0)]
    assert run(prog) == [0]
