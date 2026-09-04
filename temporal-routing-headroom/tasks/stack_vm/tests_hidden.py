import pytest
from solution import (run, VMError, StackUnderflow, BadJump, BadOpcode,
                      StepLimitExceeded, VMZeroDivision)


def test_push_arith():
    assert run([("PUSH", 7), ("PUSH", 3), ("SUB",)]) == [4]
    assert run([("PUSH", 2), ("PUSH", 3), ("ADD",)]) == [5]
    assert run([("PUSH", 4), ("PUSH", 5), ("MUL",)]) == [20]

def test_operand_order_div_mod():
    assert run([("PUSH", 7), ("PUSH", 2), ("DIV",)]) == [3]
    assert run([("PUSH", -7), ("PUSH", 2), ("DIV",)]) == [-4]  # floor
    assert run([("PUSH", -7), ("PUSH", 3), ("MOD",)]) == [2]   # python sign

def test_stack_manip():
    assert run([("PUSH", 1), ("PUSH", 2), ("SWAP",)]) == [2, 1]
    assert run([("PUSH", 5), ("DUP",), ("ADD",)]) == [10]
    assert run([("PUSH", 1), ("PUSH", 2), ("POP",)]) == [1]

def test_neg_comparisons():
    assert run([("PUSH", 3), ("NEG",)]) == [-3]
    assert run([("PUSH", 2), ("PUSH", 2), ("EQ",)]) == [1]
    assert run([("PUSH", 1), ("PUSH", 2), ("LT",)]) == [1]
    assert run([("PUSH", 1), ("PUSH", 2), ("GT",)]) == [0]

def test_load_store():
    assert run([("PUSH", 42), ("STORE", 0), ("LOAD", 0), ("LOAD", 0), ("ADD",)]) == [84]

def test_load_missing_pushes_zero():
    assert run([("LOAD", 9)]) == [0]

def test_halt_stops():
    assert run([("PUSH", 1), ("HALT",), ("PUSH", 2)]) == [1]

def test_fall_off_end():
    assert run([("PUSH", 1)]) == [1]

def test_empty_program():
    assert run([]) == []

def test_jmp_and_jz():
    # skip pushing 99
    assert run([("PUSH", 1), ("JMP", 3), ("PUSH", 99), ("PUSH", 2)]) == [1, 2]
    # JZ taken on zero
    assert run([("PUSH", 0), ("JZ", 3), ("PUSH", 99), ("PUSH", 5)]) == [5]
    # JZ not taken on nonzero
    assert run([("PUSH", 7), ("JZ", 3), ("PUSH", 99)]) == [99]

def test_jump_to_len_is_normal_end():
    assert run([("PUSH", 1), ("JMP", 2)]) == [1]

def test_untaken_jz_target_not_validated():
    assert run([("PUSH", 1), ("JZ", 999)]) == []

def test_call_ret():
    prog = [("PUSH", 2), ("CALL", 4), ("HALT",),
            ("PUSH", 99),           # never executed
            ("DUP",), ("MUL",), ("RET",)]
    assert run(prog) == [4]

def test_nested_calls():
    # main: CALL f; f: CALL g, RET; g: PUSH 7, RET
    prog = [("CALL", 2), ("HALT",),
            ("CALL", 4), ("RET",),
            ("PUSH", 7), ("RET",)]
    assert run(prog) == [7]

def test_step_limit_counts_halt():
    # 3 instructions incl. HALT at limit 3 is fine; limit 2 raises
    prog = [("PUSH", 1), ("PUSH", 2), ("HALT",)]
    assert run(prog, max_steps=3) == [1, 2]
    with pytest.raises(StepLimitExceeded):
        run(prog, max_steps=2)

def test_infinite_loop_limited():
    with pytest.raises(StepLimitExceeded):
        run([("JMP", 0)], max_steps=10)

def test_underflow_cases():
    for prog in ([("POP",)], [("DUP",)], [("PUSH", 1), ("SWAP",)],
                 [("PUSH", 1), ("ADD",)], [("STORE", 0)], [("JZ", 0)], [("NEG",)]):
        with pytest.raises(StackUnderflow):
            run(prog)

def test_ret_empty_callstack_underflow():
    with pytest.raises(StackUnderflow):
        run([("RET",)])

def test_bad_jump():
    with pytest.raises(BadJump):
        run([("JMP", 5)])
    with pytest.raises(BadJump):
        run([("JMP", -1)])
    with pytest.raises(BadJump):
        run([("PUSH", 0), ("JZ", 7)])
    with pytest.raises(BadJump):
        run([("CALL", 9)])

def test_bad_opcode():
    with pytest.raises(BadOpcode):
        run([("NOPE",)])

def test_div_mod_zero():
    with pytest.raises(VMZeroDivision):
        run([("PUSH", 1), ("PUSH", 0), ("DIV",)])
    with pytest.raises(VMZeroDivision):
        run([("PUSH", 1), ("PUSH", 0), ("MOD",)])

def test_error_hierarchy():
    for cls in (StackUnderflow, BadJump, BadOpcode, StepLimitExceeded, VMZeroDivision):
        assert issubclass(cls, VMError)

def test_countdown_loop():
    # store[0]=5; loop: load, jz end, load, push 1, sub, store, jmp loop; end
    prog = [("PUSH", 5), ("STORE", 0),
            ("LOAD", 0), ("JZ", 9),
            ("LOAD", 0), ("PUSH", 1), ("SUB",), ("STORE", 0),
            ("JMP", 2),
            ("LOAD", 0)]
    assert run(prog) == [0]
