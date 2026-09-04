import pytest
from datetime import datetime
from solution import cron_next
DT = datetime


def test_dow_only():
    # 2026-01-01 is a Thursday; next Sunday (dow 0) is 2026-01-04
    assert cron_next("0 0 * * 0", DT(2026, 1, 1)) == DT(2026, 1, 4, 0, 0)


def test_dow_saturday():
    assert cron_next("0 12 * * 6", DT(2026, 1, 1)) == DT(2026, 1, 3, 12, 0)


def test_dom_restricted_dow_star():
    assert cron_next("0 0 13 * *", DT(2026, 1, 5)) == DT(2026, 1, 13, 0, 0)
