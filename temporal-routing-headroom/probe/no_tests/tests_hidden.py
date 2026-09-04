import pytest
from datetime import datetime
from solution import cron_next

DT = datetime


def test_every_minute():
    assert cron_next("* * * * *", DT(2026, 1, 1, 0, 0, 0)) == DT(2026, 1, 1, 0, 1)

def test_strictly_after_with_seconds():
    assert cron_next("* * * * *", DT(2026, 1, 1, 12, 5, 30)) == DT(2026, 1, 1, 12, 6)

def test_exact_match_excluded():
    assert cron_next("5 12 * * *", DT(2026, 1, 1, 12, 5, 0)) == DT(2026, 1, 2, 12, 5)

def test_step_minutes():
    assert cron_next("*/15 * * * *", DT(2026, 1, 1, 0, 7)) == DT(2026, 1, 1, 0, 15)
    assert cron_next("*/15 * * * *", DT(2026, 1, 1, 0, 45)) == DT(2026, 1, 1, 1, 0)

def test_range_with_step():
    # 10,20,30,40
    assert cron_next("10-40/10 * * * *", DT(2026, 1, 1, 0, 35)) == DT(2026, 1, 1, 0, 40)
    assert cron_next("10-40/10 * * * *", DT(2026, 1, 1, 0, 41)) == DT(2026, 1, 1, 1, 10)

def test_lists_and_ranges():
    assert cron_next("0 9-17 * * *", DT(2026, 1, 1, 17, 30)) == DT(2026, 1, 2, 9, 0)
    assert cron_next("0,30 8 * * *", DT(2026, 1, 1, 8, 1)) == DT(2026, 1, 1, 8, 30)

def test_yearly_rollover():
    assert cron_next("30 4 1 1 *", DT(2026, 6, 1)) == DT(2027, 1, 1, 4, 30)
    assert cron_next("59 23 31 12 *", DT(2026, 12, 31, 23, 59)) == DT(2027, 12, 31, 23, 59)

def test_month_length():
    # 31st of a month: April has 30 days -> May 31
    assert cron_next("0 0 31 * *", DT(2026, 4, 1)) == DT(2026, 5, 31, 0, 0)

def test_leap_year():
    # 2026, 2027 not leap; 2028 is
    assert cron_next("0 0 29 2 *", DT(2026, 3, 1)) == DT(2028, 2, 29, 0, 0)

def test_dow_only():
    # 2026-01-01 is a Thursday; next Sunday (dow 0) is 2026-01-04
    assert cron_next("0 0 * * 0", DT(2026, 1, 1)) == DT(2026, 1, 4, 0, 0)

def test_dow_saturday():
    assert cron_next("0 12 * * 6", DT(2026, 1, 1)) == DT(2026, 1, 3, 12, 0)

def test_dom_and_dow_or_rule():
    # 13th OR Friday. From 2026-01-05 (Mon): first Friday is 2026-01-09 (before the 13th)
    assert cron_next("0 0 13 * 5", DT(2026, 1, 5)) == DT(2026, 1, 9, 0, 0)
    # from the 10th: the 13th (Tuesday) comes before Friday the 16th
    assert cron_next("0 0 13 * 5", DT(2026, 1, 10)) == DT(2026, 1, 13, 0, 0)

def test_dom_restricted_dow_star():
    assert cron_next("0 0 13 * *", DT(2026, 1, 5)) == DT(2026, 1, 13, 0, 0)

def test_dow_step_counts_as_restricted():
    # dow */7 == {0}; dom 13 restricted too -> OR rule: 13th or Sunday
    # from 2026-01-05 (Mon): next Sunday 2026-01-11 before the 13th
    assert cron_next("0 0 13 * */7", DT(2026, 1, 5)) == DT(2026, 1, 11, 0, 0)

def test_both_star_every_day():
    assert cron_next("0 0 * * *", DT(2026, 1, 1, 5, 0)) == DT(2026, 1, 2, 0, 0)

@pytest.mark.parametrize("bad", [
    "* * * *", "* * * * * *", "60 * * * *", "* 24 * * *", "* * 0 * *",
    "* * 32 * *", "* * * 13 *", "* * * * 7", "5-3 * * * *", "*/0 * * * *",
    "5/2 * * * *", "a * * * *", "1--2 * * * *", "1,,2 * * * *", "", "* * * * -1",
])
def test_invalid_raises(bad):
    with pytest.raises(ValueError):
        cron_next(bad, DT(2026, 1, 1))
