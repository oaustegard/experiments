# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
_______________________________ test_push_arith ________________________________
    def test_push_arith():
>       assert run([("PUSH", 7), ("PUSH", 3), ("SUB",)]) == [4]
E       assert [-4] == [4]
E         
E         At index 0 diff: -4 != 4
E         Use -v to get more diff
tests/test_public.py:7: AssertionError
_____________________________ test_countdown_loop ______________________________
    def test_countdown_loop():
        # store[0]=5; loop: load, jz end, load, push 1, sub, store, jmp loop; end
        prog = [("PUSH", 5), ("STORE", 0),
                ("LOAD", 0), ("JZ", 9),
                ("LOAD", 0), ("PUSH", 1), ("SUB",), ("STORE", 0),
                ("JMP", 2),
                ("LOAD", 0)]
>       assert run(prog) == [0]
               ^^^^^^^^^
tests/test_public.py:19: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
program = [('PUSH', 5), ('STORE', 0), ('LOAD', 0), ('JZ', 9), ('LOAD', 0), ('PUSH', 1), ...]
max_steps = 10000
...
FAILED tests/test_public.py::test_push_arith - assert [-4] == [4]
FAILED tests/test_public.py::test_countdown_loop - solution.core.StepLimitExc...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `run, VMError, StackUnderflow, BadJump, BadOpcode, StepLimitExceeded, VMZeroDivision`.
