# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
______________________ test_unary_minus_looser_than_power ______________________
    def test_unary_minus_looser_than_power():
>       assert evaluate("-2**2") == -4.0
               ^^^^^^^^^^^^^^^^^
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
FAILED tests/test_public.py::test_zero_power_zero - ValueError: unexpected to...
FAILED tests/test_public.py::test_precedence_mul_vs_power - ValueError: unexp...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `evaluate`.
