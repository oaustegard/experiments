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
        vals.update(range(a, b, step))
    return vals, True
