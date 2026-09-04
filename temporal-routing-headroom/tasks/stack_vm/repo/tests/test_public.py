import pytest
from solution import (run, VMError, StackUnderflow, BadJump, BadOpcode,
                      StepLimitExceeded, VMZeroDivision)


def test_neg_comparisons():
    assert run([("PUSH", 3), ("NEG",)]) == [-3]
    assert run([("PUSH", 2), ("PUSH", 2), ("EQ",)]) == [1]
    assert run([("PUSH", 1), ("PUSH", 2), ("LT",)]) == [1]
    assert run([("PUSH", 1), ("PUSH", 2), ("GT",)]) == [0]


def test_div_mod_zero():
    with pytest.raises(VMZeroDivision):
        run([("PUSH", 1), ("PUSH", 0), ("DIV",)])
    with pytest.raises(VMZeroDivision):
        run([("PUSH", 1), ("PUSH", 0), ("MOD",)])


def test_countdown_loop():
    # store[0]=5; loop: load, jz end, load, push 1, sub, store, jmp loop; end
    prog = [("PUSH", 5), ("STORE", 0),
            ("LOAD", 0), ("JZ", 9),
            ("LOAD", 0), ("PUSH", 1), ("SUB",), ("STORE", 0),
            ("JMP", 2),
            ("LOAD", 0)]
    assert run(prog) == [0]
