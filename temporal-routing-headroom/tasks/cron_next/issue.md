# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
________________________________ test_dow_only _________________________________
    def test_dow_only():
        # 2026-01-01 is a Thursday; next Sunday (dow 0) is 2026-01-04
>       assert cron_next("0 0 * * 0", DT(2026, 1, 1)) == DT(2026, 1, 4, 0, 0)
E       AssertionError: assert datetime.datetime(2026, 1, 2, 0, 0) == datetime.datetime(2026, 1, 4, 0, 0)
E        +  where datetime.datetime(2026, 1, 2, 0, 0) = cron_next('0 0 * * 0', datetime.datetime(2026, 1, 1, 0, 0))
E        +    where datetime.datetime(2026, 1, 1, 0, 0) = DT(2026, 1, 1)
E        +  and   datetime.datetime(2026, 1, 4, 0, 0) = DT(2026, 1, 4, 0, 0)
tests/test_public.py:9: AssertionError
______________________________ test_dow_saturday _______________________________
    def test_dow_saturday():
>       assert cron_next("0 12 * * 6", DT(2026, 1, 1)) == DT(2026, 1, 3, 12, 0)
E       AssertionError: assert datetime.datetime(2026, 1, 1, 12, 0) == datetime.datetime(2026, 1, 3, 12, 0)
E        +  where datetime.datetime(2026, 1, 1, 12, 0) = cron_next('0 12 * * 6', datetime.datetime(2026, 1, 1, 0, 0))
E        +    where datetime.datetime(2026, 1, 1, 0, 0) = DT(2026, 1, 1)
E        +  and   datetime.datetime(2026, 1, 3, 12, 0) = DT(2026, 1, 3, 12, 0)
tests/test_public.py:13: AssertionError
_________________________ test_dom_restricted_dow_star _________________________
    def test_dom_restricted_dow_star():
>       assert cron_next("0 0 13 * *", DT(2026, 1, 5)) == DT(2026, 1, 13, 0, 0)
E       AssertionError: assert datetime.datetime(2026, 1, 6, 0, 0) == datetime.datetime(2026, 1, 13, 0, 0)
E        +  where datetime.datetime(2026, 1, 6, 0, 0) = cron_next('0 0 13 * *', datetime.datetime(2026, 1, 5, 0, 0))
...
FAILED tests/test_public.py::test_dow_saturday - AssertionError: assert datet...
FAILED tests/test_public.py::test_dom_restricted_dow_star - AssertionError: a...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `cron_next`.
