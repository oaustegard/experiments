from datetime import datetime, timedelta

_BOUNDS = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]


def _parse_field(field: str, lo: int, hi: int):
    """Return (set_of_values, restricted: bool)."""
    if field == "":
        raise ValueError("empty field")
    if field == "*":
        return set(range(lo, hi + 1)), False
    vals = set()
    for item in field.split(","):
        if item == "":
            raise ValueError("empty item")
        step = 1
        if "/" in item:
            base, s = item.split("/", 1)
            if not s.isdigit() or int(s) < 1:
                raise ValueError(f"bad step: {item!r}")
            step = int(s)
            if base != "*" and "-" not in base:
                raise ValueError(f"step on single value: {item!r}")
        else:
            base = item
        if base == "*":
            a, b = lo, hi
        elif "-" in base:
            parts = base.split("-")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"bad range: {base!r}")
            a, b = int(parts[0]), int(parts[1])
            if a > b:
                raise ValueError(f"reversed range: {base!r}")
        else:
            if not base.isdigit():
                raise ValueError(f"bad value: {base!r}")
            a = b = int(base)
        if a < lo or b > hi:
            raise ValueError(f"out of range: {item!r}")
        vals.update(range(a, b + 1, step))
    return vals, True


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
