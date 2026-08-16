"""cron_next: compute the next datetime matching a 5-field cron expression."""

import re
from datetime import datetime, timedelta

_INT_RE = re.compile(r'\d+')


def _parse_int(tok):
    if not _INT_RE.fullmatch(tok):
        raise ValueError(f"invalid integer token: {tok!r}")
    return int(tok)


def _parse_item(item, lo, hi):
    if item == '':
        raise ValueError("empty item in cron field")

    if '/' in item:
        parts = item.split('/')
        if len(parts) != 2:
            raise ValueError(f"malformed step item: {item!r}")
        base, step_s = parts
        if step_s == '':
            raise ValueError(f"malformed step item: {item!r}")
        step = _parse_int(step_s)
        if step < 1:
            raise ValueError(f"step must be >= 1: {item!r}")

        if base == '*':
            start, end = lo, hi
        elif '-' in base:
            b_parts = base.split('-')
            if len(b_parts) != 2:
                raise ValueError(f"malformed range: {base!r}")
            a_s, b_s = b_parts
            a = _parse_int(a_s)
            b = _parse_int(b_s)
            if a > b:
                raise ValueError(f"range start > end: {base!r}")
            start, end = a, b
        else:
            # step applied to a single value is invalid (e.g. "5/2")
            raise ValueError(f"step on single value is invalid: {item!r}")

        if start < lo or start > hi or end < lo or end > hi:
            raise ValueError(f"value out of range: {item!r}")

        return set(range(start, end + 1, step))

    if item == '*':
        return set(range(lo, hi + 1))

    if '-' in item:
        b_parts = item.split('-')
        if len(b_parts) != 2:
            raise ValueError(f"malformed range: {item!r}")
        a_s, b_s = b_parts
        a = _parse_int(a_s)
        b = _parse_int(b_s)
        if a > b:
            raise ValueError(f"range start > end: {item!r}")
        if a < lo or a > hi or b < lo or b > hi:
            raise ValueError(f"value out of range: {item!r}")
        return set(range(a, b + 1))

    v = _parse_int(item)
    if v < lo or v > hi:
        raise ValueError(f"value out of range: {item!r}")
    return {v}


def _parse_field(field_str, lo, hi):
    if field_str == '':
        raise ValueError("empty cron field")
    items = field_str.split(',')
    values = set()
    for it in items:
        if it == '':
            raise ValueError("empty item in cron field")
        values |= _parse_item(it, lo, hi)
    restricted = field_str != '*'
    return values, restricted


def _cron_dow(d):
    # Python's weekday(): Monday=0 ... Sunday=6
    # Cron's day-of-week: Sunday=0 ... Saturday=6
    return (d.weekday() + 1) % 7


def cron_next(expr: str, after: datetime) -> datetime:
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(
            f"expected 5 fields (minute hour day-of-month month day-of-week), got {len(fields)}"
        )
    minute_f, hour_f, dom_f, month_f, dow_f = fields

    minutes, _ = _parse_field(minute_f, 0, 59)
    hours, _ = _parse_field(hour_f, 0, 23)
    doms, dom_restricted = _parse_field(dom_f, 1, 31)
    months, _ = _parse_field(month_f, 1, 12)
    dows, dow_restricted = _parse_field(dow_f, 0, 6)

    sorted_hours = sorted(hours)
    sorted_minutes = sorted(minutes)

    def day_matches(d):
        dom_ok = d.day in doms
        dow_ok = _cron_dow(d) in dows
        if dom_restricted and dow_restricted:
            return dom_ok or dow_ok
        if dom_restricted:
            return dom_ok
        if dow_restricted:
            return dow_ok
        return True

    def next_hour_after(h):
        for hh in sorted_hours:
            if hh > h:
                return hh
        return None

    def next_minute_after(m):
        for mm in sorted_minutes:
            if mm > m:
                return mm
        return None

    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    horizon = candidate + timedelta(days=365 * 5 + 2)

    while candidate <= horizon:
        if candidate.month not in months or not day_matches(candidate):
            nxt = candidate + timedelta(days=1)
            candidate = nxt.replace(hour=0, minute=0)
            continue

        if candidate.hour not in hours:
            nh = next_hour_after(candidate.hour)
            if nh is None:
                nxt = candidate + timedelta(days=1)
                candidate = nxt.replace(hour=0, minute=0)
            else:
                candidate = candidate.replace(hour=nh, minute=0)
            continue

        if candidate.minute not in minutes:
            nm = next_minute_after(candidate.minute)
            if nm is None:
                nh = next_hour_after(candidate.hour)
                if nh is None:
                    nxt = candidate + timedelta(days=1)
                    candidate = nxt.replace(hour=0, minute=0)
                else:
                    candidate = candidate.replace(hour=nh, minute=0)
            else:
                candidate = candidate.replace(minute=nm)
            continue

        return candidate

    raise ValueError("no matching time found within the search horizon")
