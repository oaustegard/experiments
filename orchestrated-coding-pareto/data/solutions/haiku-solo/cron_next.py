from datetime import datetime, timedelta
import calendar


def cron_next(expr: str, after: datetime) -> datetime:
    """
    Return the earliest datetime strictly after `after` that matches
    a 5-field cron expression.

    Fields: minute hour day-of-month month day-of-week
    """
    # Parse the cron expression
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields, got {len(fields)}")

    minute_field, hour_field, dom_field, month_field, dow_field = fields

    # Parse each field into sets of allowed values
    minutes = parse_field(minute_field, 0, 59)
    hours = parse_field(hour_field, 0, 23)
    days_of_month = parse_field(dom_field, 1, 31)
    months = parse_field(month_field, 1, 12)
    days_of_week = parse_field(dow_field, 0, 6)

    # Check if day fields are restricted (not just *)
    dom_restricted = dom_field != "*"
    dow_restricted = dow_field != "*"

    # Start from the next minute
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

    # Search for the next matching time (up to 5 years as per spec)
    max_iterations = 5 * 365 * 24 * 60 + 1000

    for _ in range(max_iterations):
        if matches(candidate, minutes, hours, days_of_month, months,
                   days_of_week, dom_restricted, dow_restricted):
            return candidate
        candidate += timedelta(minutes=1)

    raise ValueError("No matching time found")


def parse_field(field, min_val, max_val):
    """
    Parse a single cron field and return a set of allowed values.
    Raises ValueError for malformed input.
    """
    if not field:
        raise ValueError("Empty field")

    # Handle the * wildcard
    if field == "*":
        return set(range(min_val, max_val + 1))

    allowed = set()

    # Process comma-separated items
    for item in field.split(","):
        item = item.strip()

        if not item:
            raise ValueError("Empty item in field")

        # Check for step syntax (/ operator)
        if "/" in item:
            parts = item.split("/")
            if len(parts) != 2:
                raise ValueError(f"Invalid step syntax: {item}")

            range_part, step_str = parts

            try:
                step = int(step_str)
            except ValueError:
                raise ValueError(f"Invalid step value: {step_str}")

            if step < 1:
                raise ValueError(f"Step must be >= 1, got {step}")

            if range_part == "*":
                # */S: every S-th value from min_val to max_val
                for val in range(min_val, max_val + 1, step):
                    allowed.add(val)
            elif "-" in range_part:
                # A-B/S: every S-th value in range [A, B]
                try:
                    a, b = map(int, range_part.split("-"))
                except ValueError:
                    raise ValueError(f"Invalid range in step: {range_part}")

                if a > b:
                    raise ValueError(f"Invalid range {a}-{b}")
                if a < min_val or b > max_val:
                    raise ValueError(f"Range out of bounds: {a}-{b}")

                for val in range(a, b + 1, step):
                    allowed.add(val)
            else:
                # Single value with step is invalid
                raise ValueError(f"Step on single value: {item}")

        # Check for range syntax (- operator, without step)
        elif "-" in item:
            try:
                a, b = map(int, item.split("-"))
            except ValueError:
                raise ValueError(f"Invalid range: {item}")

            if a > b:
                raise ValueError(f"Invalid range {a}-{b}")
            if a < min_val or b > max_val:
                raise ValueError(f"Range out of bounds: {item}")

            for val in range(a, b + 1):
                allowed.add(val)

        # Single numeric value
        else:
            try:
                val = int(item)
            except ValueError:
                raise ValueError(f"Non-numeric value: {item}")

            if val < min_val or val > max_val:
                raise ValueError(f"Value out of range: {val}")

            allowed.add(val)

    return allowed


def matches(dt, minutes, hours, days_of_month, months, days_of_week,
            dom_restricted, dow_restricted):
    """
    Check if a datetime matches the given cron field values.
    """
    # Check minute, hour, and month
    if dt.minute not in minutes or dt.hour not in hours or dt.month not in months:
        return False

    # Get the maximum day in this month
    max_day_in_month = calendar.monthrange(dt.year, dt.month)[1]

    # Apply the day rule
    if dom_restricted and dow_restricted:
        # Both day-of-month and day-of-week are restricted:
        # Match if EITHER field matches
        dom_match = dt.day in days_of_month and dt.day <= max_day_in_month

        # Convert Python weekday (0=Monday) to cron weekday (0=Sunday)
        python_weekday = dt.weekday()
        cron_weekday = (python_weekday + 1) % 7
        dow_match = cron_weekday in days_of_week

        return dom_match or dow_match

    elif dom_restricted:
        # Only day-of-month is restricted
        return dt.day in days_of_month and dt.day <= max_day_in_month

    elif dow_restricted:
        # Only day-of-week is restricted
        python_weekday = dt.weekday()
        cron_weekday = (python_weekday + 1) % 7
        return cron_weekday in days_of_week

    else:
        # Both are wildcards (*)
        return True
