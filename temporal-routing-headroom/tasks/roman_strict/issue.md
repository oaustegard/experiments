# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
______________________________ test_basic_values _______________________________
    def test_basic_values():
        assert from_roman("I") == 1
        assert from_roman("IV") == 4
        assert from_roman("IX") == 9
        assert from_roman("XIV") == 14
        assert from_roman("XL") == 40
        assert from_roman("XC") == 90
        assert from_roman("CD") == 400
>       assert from_roman("CM") == 900
               ^^^^^^^^^^^^^^^^
tests/test_public.py:16: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
s = 'CM'
    def from_roman(s: str) -> int:
        if not isinstance(s, str) or s not in _TABLE:
>           raise ValueError(f"not a strict Roman numeral: {s!r}")
E           ValueError: not a strict Roman numeral: 'CM'
_________________________________ test_famous __________________________________
    def test_famous():
>       assert from_roman("MCMXCIV") == 1994
               ^^^^^^^^^^^^^^^^^^^^^
...
FAILED tests/test_public.py::test_famous - ValueError: not a strict Roman num...
FAILED tests/test_public.py::test_full_roundtrip_1_to_3999 - NameError: name ...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `from_roman`.
