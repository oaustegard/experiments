# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
___________________________ test_dom_and_dow_or_rule ___________________________
    def test_dom_and_dow_or_rule():
        # 13th OR Friday. From 2026-01-05 (Mon): first Friday is 2026-01-09 (before the 13th)
>       assert cron_next("0 0 13 * 5", DT(2026, 1, 5)) == DT(2026, 1, 9, 0, 0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_public.py:9: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
expr = '0 0 13 * 5', after = datetime.datetime(2026, 1, 5, 0, 0)
    def cron_next(expr: str, after: datetime) -> datetime:
        fields = expr.split()
        if len(fields) != 5:
            raise ValueError(f"expected 5 fields, got {len(fields)}")
        (minutes, _), (hours, _), (doms, dom_r), (months, _), (dows, dow_r) = [
            _parse_field(f, lo, hi) for f, (lo, hi) in zip(fields, _BOUNDS)
        ]
        def day_ok(d: datetime) -> bool:
            dom_hit = d.day in doms
            # python: Monday=0..Sunday=6; cron: Sunday=0..Saturday=6
            dow_hit = ((d.weekday() + 1) % 7) in dows
            if dom_r and dow_r:
                return dom_hit or dow_hit
            if dom_r:
...
FAILED tests/test_public.py::test_dom_restricted_dow_star - ValueError: no ma...
FAILED tests/test_public.py::test_both_star_every_day - ValueError: no match ...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `cron_next`.
