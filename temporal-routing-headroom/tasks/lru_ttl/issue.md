# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
=================================== FAILURES ===================================
__________________________ test_expiry_at_exactly_ttl __________________________
    def test_expiry_at_exactly_ttl():
        c = Clock()
        cache = TTLCache(2, 10.0, c)
        cache.put("a", 1)
        c.t = 10.0
>       assert cache.get("a") is None
E       AssertionError: assert 1 is None
E        +  where 1 = get('a')
E        +    where get = <solution.core.TTLCache object at 0x...>.get
tests/test_public.py:14: AssertionError
________________________ test_get_expired_removes_entry ________________________
    def test_get_expired_removes_entry():
        c = Clock()
        cache = TTLCache(2, 10.0, c)
        cache.put("a", 1)
        c.t = 11.0
>       assert cache.get("a") is None
E       AssertionError: assert 1 is None
E        +  where 1 = get('a')
E        +    where get = <solution.core.TTLCache object at 0x...>.get
tests/test_public.py:22: AssertionError
=========================== short test summary info ============================
FAILED tests/test_public.py::test_expiry_at_exactly_ttl - AssertionError: ass...
FAILED tests/test_public.py::test_get_expired_removes_entry - AssertionError:...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `TTLCache`.
