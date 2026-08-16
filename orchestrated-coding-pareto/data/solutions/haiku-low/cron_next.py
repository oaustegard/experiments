from datetime import datetime, timedelta


def cron_next(expr: str, after: datetime) -> datetime:
    """
    Return the earliest datetime strictly after `after` that matches the cron expression.

    Args:
        expr: 5-field cron expression (minute hour day-of-month month day-of-week)
        after: datetime to start searching after

    Returns:
        datetime with seconds and microseconds set to 0

    Raises:
        ValueError: if the expression is invalid
    """

    fields = expr.split()
    if len(fields) != 5:
        raise ValueError("Expected 5 fields")

    ranges = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day-of-month
        (1, 12),   # month
        (0, 6),    # day-of-week
    ]

    # Parse each field
    parsed_fields = []
    for i, field in enumerate(fields):
        if not field:
            raise ValueError("Empty field")
        min_val, max_val = ranges[i]
        values = _parse_field(field, min_val, max_val)
        parsed_fields.append(values)

    minute_vals, hour_vals, dom_vals, month_vals, dow_vals = parsed_fields

    # Check if dom and dow are restricted (literally not "*")
    dom_restricted = fields[2] != '*'
    dow_restricted = fields[4] != '*'

    # Start searching from the next minute
    current = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Search for up to 5 years
    max_date = after + timedelta(days=5*365)

    while current <= max_date:
        minute = current.minute
        hour = current.hour
        day = current.day
        month = current.month
        weekday = current.weekday()
        # Convert Python weekday (0=Monday) to cron weekday (0=Sunday)
        cron_dow = (weekday + 1) % 7

        # Check if time matches
        if minute not in minute_vals or hour not in hour_vals or month not in month_vals:
            current += timedelta(minutes=1)
            continue

        # Check day rule
        day_matches = False
        if dom_restricted and dow_restricted:
            # Either day-of-month or day-of-week must match
            day_matches = (day in dom_vals) or (cron_dow in dow_vals)
        elif dom_restricted:
            # Only day-of-month is restricted
            day_matches = day in dom_vals
        elif dow_restricted:
            # Only day-of-week is restricted
            day_matches = cron_dow in dow_vals
        else:
            # Both are unrestricted
            day_matches = True

        if day_matches:
            return current

        current += timedelta(minutes=1)

    raise ValueError("No matching time found within 5 years")


def _parse_field(field: str, min_val: int, max_val: int) -> set:
    """Parse a single cron field and return a set of valid values."""
    values = set()
    items = field.split(',')

    for item in items:
        if not item:
            raise ValueError("Empty item in field")

        if '/' in item:
            parts = item.split('/')
            if len(parts) != 2:
                raise ValueError(f"Invalid step syntax: {item}")

            range_part = parts[0]
            try:
                step = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid step value: {parts[1]}")

            if step < 1:
                raise ValueError("Step must be >= 1")

            if range_part == '*':
                range_min, range_max = min_val, max_val
            elif '-' in range_part:
                try:
                    range_min, range_max = map(int, range_part.split('-'))
                except ValueError:
                    raise ValueError(f"Invalid range: {range_part}")
                if range_min < min_val or range_max > max_val:
                    raise ValueError(f"Range out of bounds: {range_part}")
                if range_min > range_max:
                    raise ValueError(f"Invalid range (start > end): {range_part}")
            else:
                try:
                    int(range_part)
                except ValueError:
                    raise ValueError(f"Invalid item: {item}")
                else:
                    raise ValueError(f"Step on single value not allowed: {item}")

            current = range_min
            while current <= range_max:
                values.add(current)
                current += step

        elif '-' in item:
            parts = item.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid range: {item}")

            try:
                range_min, range_max = map(int, parts)
            except ValueError:
                raise ValueError(f"Invalid range: {item}")

            if range_min < min_val or range_max > max_val:
                raise ValueError(f"Range out of bounds: {item}")
            if range_min > range_max:
                raise ValueError(f"Invalid range (start > end): {item}")

            for val in range(range_min, range_max + 1):
                values.add(val)

        elif item == '*':
            for val in range(min_val, max_val + 1):
                values.add(val)

        else:
            try:
                val = int(item)
            except ValueError:
                raise ValueError(f"Invalid item: {item}")

            if val < min_val or val > max_val:
                raise ValueError(f"Value out of range: {val}")

            values.add(val)

    return values
