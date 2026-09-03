from datetime import datetime, timedelta
from .util import _BOUNDS, _parse_field


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
            return dom_hit
        if dow_r:
            return dow_hit
        return True

    # scan day by day; whole-minute candidates need only be strictly after `after`
    day = after.replace(hour=0, minute=0, second=0, microsecond=0)
    horizon = after + timedelta(days=366 * 5 + 5)
    while day <= horizon:
        if day.month in months and day_ok(day):
            for h in sorted(hours):
                for m in sorted(minutes):
                    cand = day.replace(hour=h, minute=m)
                    if cand > after:
                        return cand
        day += timedelta(days=1)
    raise ValueError("no match within horizon")
