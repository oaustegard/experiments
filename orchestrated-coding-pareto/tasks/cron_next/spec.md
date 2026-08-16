# Task: cron_next

Write a Python module defining exactly one public function:

```python
from datetime import datetime

def cron_next(expr: str, after: datetime) -> datetime
```

Return the earliest datetime **strictly after** `after` that matches a 5-field cron
expression. Do not use any third-party library (no `croniter`); `datetime` /
`calendar` from the standard library are fine.

Field syntax (standard cron):
- Five whitespace-separated fields: `minute hour day-of-month month day-of-week`.
- Ranges: minute 0-59, hour 0-23, day-of-month 1-31, month 1-12, day-of-week 0-6
  with **0 = Sunday**, 6 = Saturday. (7 is NOT accepted for Sunday.)
- Each field is a comma-separated list of items. An item is:
  - `*` — all values
  - a single value `N`
  - a range `A-B` (inclusive, A <= B required)
  - a step over a range or star: `*/S`, `A-B/S` (every S-th value starting at A;
    e.g. `10-40/10` = 10,20,30,40; `*/15` in the minute field = 0,15,30,45).
    A step on a single value (`5/2`) is invalid. S must be >= 1.
- Errors — raise `ValueError`: wrong field count, values out of range, `A-B` with
  A > B, zero/negative step, malformed items, empty fields.

Matching semantics:
- The result has seconds and microseconds equal to 0. A time matches when minute,
  hour and month all match, and the **day rule** matches.
- Day rule (the classic cron quirk): if **both** day-of-month and day-of-week fields
  are restricted (neither is `*`), the day matches when **either** field matches.
  If only one of them is restricted, that field alone must match. If both are `*`,
  every day matches. Note: a field like `*/2` or `1-31` counts as **restricted**
  (anything other than a bare `*`).
- `after` is a naive datetime, possibly with nonzero seconds — the answer is the
  first matching whole minute strictly later than `after` (e.g. `after=12:05:30`
  and every-minute cron -> 12:06:00; `after=12:05:00` exactly on a matching minute
  -> the *next* matching minute).
- Month lengths and leap years follow the real calendar (`0 0 29 2 *` only fires
  on Feb 29 of leap years).
- You may assume a matching time exists within 5 years of `after`; behavior past
  that horizon is not tested (an impossible cron like `0 0 31 2 *` will not appear).

Examples (ISO format):
- `cron_next("*/15 * * * *", 2026-01-01T00:07:00)` -> 2026-01-01T00:15:00
- `cron_next("0 0 13 * 5", ...)` -> fires on every Friday AND every 13th (OR rule)
- `cron_next("30 4 1 1 *", 2026-06-01T00:00:00)` -> 2027-01-01T04:30:00

No I/O. Standard library only. Only `cron_next` is tested.
