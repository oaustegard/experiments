"""Minimal 5-field cron expression evaluator.

Public API::

    cron_next(expr: str, after: datetime) -> datetime

Returns the earliest datetime strictly after ``after`` matching ``expr``.
Standard library only.
"""

from datetime import datetime, timedelta

__all__ = ["cron_next"]


_FIELD_BOUNDS = (
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 6),    # day of week, 0 = Sunday
)


def _parse_int(text):
    """Parse a bare non-negative decimal integer, else raise ValueError."""
    if not text or not text.isdigit():
        raise ValueError("invalid cron number: %r" % (text,))
    return int(text)


def _parse_item(item, lo, hi):
    """Expand one comma-separated cron item into a set of ints."""
    if not item:
        raise ValueError("empty cron item")

    step = 1
    base = item
    if "/" in item:
        parts = item.split("/")
        if len(parts) != 2:
            raise ValueError("malformed step in cron item: %r" % (item,))
        base, step_text = parts
        step = _parse_int(step_text)
        if step < 1:
            raise ValueError("cron step must be >= 1: %r" % (item,))
        if base != "*" and "-" not in base:
            raise ValueError("step on a single value is invalid: %r" % (item,))

    if base == "*":
        start, end = lo, hi
    elif "-" in base:
        parts = base.split("-")
        if len(parts) != 2:
            raise ValueError("malformed range in cron item: %r" % (item,))
        start = _parse_int(parts[0])
        end = _parse_int(parts[1])
        if start > end:
            raise ValueError("range start after end: %r" % (item,))
    else:
        start = end = _parse_int(base)

    if start < lo or start > hi or end < lo or end > hi:
        raise ValueError("cron value out of range [%d,%d]: %r" % (lo, hi, item))

    return set(range(start, end + 1, step))


def _parse_field(field, lo, hi):
    """Expand one whitespace-delimited cron field into a set of ints."""
    if not field:
        raise ValueError("empty cron field")
    values = set()
    for item in field.split(","):
        values |= _parse_item(item.strip(), lo, hi)
    if not values:
        raise ValueError("cron field matches nothing: %r" % (field,))
    return values


def _parse(expr):
    if not isinstance(expr, str):
        raise ValueError("cron expression must be a string")
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError("cron expression must have exactly 5 fields")
    sets = [
        _parse_field(f, lo, hi)
        for f, (lo, hi) in zip(fields, _FIELD_BOUNDS)
    ]
    dom_restricted = fields[2].strip() != "*"
    dow_restricted = fields[4].strip() != "*"
    return sets, dom_restricted, dow_restricted


def _day_matches(moment, doms, dows, dom_restricted, dow_restricted):
    dom_hit = moment.day in doms
    dow_hit = ((moment.weekday() + 1) % 7) in dows
    if dom_restricted and dow_restricted:
        return dom_hit or dow_hit
    if dom_restricted:
        return dom_hit
    if dow_restricted:
        return dow_hit
    return True


def _next_day(moment):
    return (moment + timedelta(days=1)).replace(hour=0, minute=0)


def cron_next(expr, after):
    """Earliest datetime strictly after ``after`` matching the cron ``expr``."""
    (minutes, hours, doms, months, dows), dom_r, dow_r = _parse(expr)

    if not isinstance(after, datetime):
        raise ValueError("after must be a datetime")

    moment = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    horizon = after + timedelta(days=366 * 5 + 2)
    while moment <= horizon:
        if moment.month not in months:
            # Jump to the first minute of the next month.
            if moment.month == 12:
                moment = moment.replace(
                    year=moment.year + 1, month=1, day=1, hour=0, minute=0
                )
            else:
                moment = moment.replace(
                    month=moment.month + 1, day=1, hour=0, minute=0
                )
            continue

        if not _day_matches(moment, doms, dows, dom_r, dow_r):
            moment = _next_day(moment)
            continue

        if moment.hour not in hours:
            nxt = [h for h in hours if h > moment.hour]
            if not nxt:
                moment = _next_day(moment)
            else:
                moment = moment.replace(hour=min(nxt), minute=0)
            continue

        if moment.minute not in minutes:
            nxt = [m for m in minutes if m > moment.minute]
            if not nxt:
                if moment.hour == 23:
                    moment = _next_day(moment)
                else:
                    moment = moment.replace(hour=moment.hour + 1, minute=0)
            else:
                moment = moment.replace(minute=min(nxt))
            continue

        return moment

    raise ValueError("no matching time found within 5 years")
