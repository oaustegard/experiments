# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
_____________________________ test_neg_comparisons _____________________________
    def test_neg_comparisons():
        assert run([("PUSH", 3), ("NEG",)]) == [-3]
        assert run([("PUSH", 2), ("PUSH", 2), ("EQ",)]) == [1]
>       assert run([("PUSH", 1), ("PUSH", 2), ("LT",)]) == [1]
E       assert [0] == [1]
E         
E         At index 0 diff: 0 != 1
E         Use -v to get more diff
tests/test_public.py:9: AssertionError
______________________________ test_div_mod_zero _______________________________
    def test_div_mod_zero():
>       with pytest.raises(VMZeroDivision):
E       Failed: DID NOT RAISE VMZeroDivision
tests/test_public.py:14: Failed
_____________________________ test_countdown_loop ______________________________
    def test_countdown_loop():
        # store[0]=5; loop: load, jz end, load, push 1, sub, store, jmp loop; end
        prog = [("PUSH", 5), ("STORE", 0),
                ("LOAD", 0), ("JZ", 9),
                ("LOAD", 0), ("PUSH", 1), ("SUB",), ("STORE", 0),
                ("JMP", 2),
...
FAILED tests/test_public.py::test_div_mod_zero - Failed: DID NOT RAISE VMZero...
FAILED tests/test_public.py::test_countdown_loop - solution.core.StepLimitExc...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `run, VMError, StackUnderflow, BadJump, BadOpcode, StepLimitExceeded, VMZeroDivision`.
