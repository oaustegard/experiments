# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
_________________________ test_semver_canonical_chain __________________________
    def test_semver_canonical_chain():
        chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta",
                 "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0"]
        for i in range(len(chain) - 1):
>           assert compare(chain[i], chain[i + 1]) == -1, (chain[i], chain[i + 1])
E           AssertionError: ('1.0.0-beta.2', '1.0.0-beta.11')
E           assert 1 == -1
E            +  where 1 = compare('1.0.0-beta.2', '1.0.0-beta.11')
tests/test_public.py:9: AssertionError
____________________ test_numeric_identifiers_numeric_order ____________________
    def test_numeric_identifiers_numeric_order():
>       assert compare("1.0.0-beta.2", "1.0.0-beta.11") == -1
E       AssertionError: assert 1 == -1
E        +  where 1 = compare('1.0.0-beta.2', '1.0.0-beta.11')
tests/test_public.py:14: AssertionError
=========================== short test summary info ============================
FAILED tests/test_public.py::test_semver_canonical_chain - AssertionError: ('...
FAILED tests/test_public.py::test_numeric_identifiers_numeric_order - Asserti...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `compare`.
