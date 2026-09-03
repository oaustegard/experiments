import pytest
from datetime import datetime
from solution import cron_next
DT = datetime


def test_dom_and_dow_or_rule():
    # 13th OR Friday. From 2026-01-05 (Mon): first Friday is 2026-01-09 (before the 13th)
    assert cron_next("0 0 13 * 5", DT(2026, 1, 5)) == DT(2026, 1, 9, 0, 0)
    # from the 10th: the 13th (Tuesday) comes before Friday the 16th
    assert cron_next("0 0 13 * 5", DT(2026, 1, 10)) == DT(2026, 1, 13, 0, 0)


def test_dom_restricted_dow_star():
    assert cron_next("0 0 13 * *", DT(2026, 1, 5)) == DT(2026, 1, 13, 0, 0)


def test_both_star_every_day():
    assert cron_next("0 0 * * *", DT(2026, 1, 1, 5, 0)) == DT(2026, 1, 2, 0, 0)
