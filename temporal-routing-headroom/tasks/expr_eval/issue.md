# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
____________________________ test_power_right_assoc ____________________________
    def test_power_right_assoc():
>       assert evaluate("2**3**2") == 512.0
               ^^^^^^^^^^^^^^^^^^^
tests/test_public.py:6: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    v = p.expr()
        ^^^^^^^^
    v = self.term()
        ^^^^^^^^^^^
    rhs = self.unary()
          ^^^^^^^^^^^^
    return self.power()
           ^^^^^^^^^^^^
    v = self.atom()
        ^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
self = <solution.core._Parser object at 0x...>
    def atom(self):
        t = self.next()
        if t[0] == "num":
            return t[1]
...
FAILED tests/test_public.py::test_unary_after_power_binds_tight - ValueError:...
FAILED tests/test_public.py::test_precedence_mul_vs_power - ValueError: unexp...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `evaluate`.
