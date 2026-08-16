import re


def compare(a: str, b: str) -> int:
    """Compare two semantic versions according to SemVer 2.0.0 rules.

    Returns -1 if a < b, 0 if equal precedence, 1 if a > b.
    Raises ValueError for invalid version strings.
    """
    ver_a = _parse_version(a)
    ver_b = _parse_version(b)

    # Compare MAJOR.MINOR.PATCH numerically
    if ver_a['major'] != ver_b['major']:
        return -1 if ver_a['major'] < ver_b['major'] else 1
    if ver_a['minor'] != ver_b['minor']:
        return -1 if ver_a['minor'] < ver_b['minor'] else 1
    if ver_a['patch'] != ver_b['patch']:
        return -1 if ver_a['patch'] < ver_b['patch'] else 1

    # Compare prerelease versions
    return _compare_prerelease(ver_a['prerelease'], ver_b['prerelease'])


def _parse_version(version_str: str) -> dict:
    """Parse and validate a semantic version string."""
    if version_str.startswith('v') or version_str.startswith('V'):
        raise ValueError("v prefix not allowed")

    if any(c.isspace() for c in version_str):
        raise ValueError("whitespace not allowed")

    # Split BUILD metadata (ignored for precedence but must be valid)
    if '+' in version_str:
        version_part, build_part = version_str.rsplit('+', 1)
        _validate_build(build_part)
    else:
        version_part = version_str

    # Split PRERELEASE
    if '-' in version_part:
        core_part, prerelease_str = version_part.split('-', 1)
        prerelease = _parse_prerelease(prerelease_str)
    else:
        core_part = version_part
        prerelease = None

    # Parse MAJOR.MINOR.PATCH
    parts = core_part.split('.')
    if len(parts) != 3:
        raise ValueError("must have MAJOR.MINOR.PATCH")

    try:
        major = _parse_numeric_field(parts[0])
        minor = _parse_numeric_field(parts[1])
        patch = _parse_numeric_field(parts[2])
    except ValueError:
        raise ValueError("invalid numeric field")

    return {
        'major': major,
        'minor': minor,
        'patch': patch,
        'prerelease': prerelease
    }


def _parse_numeric_field(s: str) -> int:
    """Parse MAJOR/MINOR/PATCH field (no leading zeros except '0')."""
    if not s or not s.isdigit():
        raise ValueError("not numeric")
    if len(s) > 1 and s[0] == '0':
        raise ValueError("leading zero")
    return int(s)


def _parse_prerelease(prerelease_str: str) -> list:
    """Parse and validate prerelease identifiers."""
    if not prerelease_str:
        raise ValueError("empty prerelease")

    identifiers = prerelease_str.split('.')
    parsed = []

    for identifier in identifiers:
        if not identifier:
            raise ValueError("empty identifier")

        if identifier.isdigit():
            # Numeric identifier: digits only, no leading zeros except '0'
            if len(identifier) > 1 and identifier[0] == '0':
                raise ValueError("leading zero in numeric identifier")
            parsed.append(('numeric', int(identifier)))
        else:
            # Alphanumeric identifier: must contain at least one non-digit
            if not re.match(r'^[a-zA-Z0-9-]+$', identifier):
                raise ValueError("invalid characters in identifier")
            parsed.append(('alphanumeric', identifier))

    return parsed


def _validate_build(build_str: str) -> None:
    """Validate build metadata (ignored for precedence)."""
    if not build_str:
        raise ValueError("empty build")

    identifiers = build_str.split('.')
    for identifier in identifiers:
        if not identifier:
            raise ValueError("empty build identifier")
        if not re.match(r'^[a-zA-Z0-9-]+$', identifier):
            raise ValueError("invalid build metadata")


def _compare_prerelease(pre_a, pre_b) -> int:
    """Compare two prerelease versions.

    Rules:
    - Version without prerelease has higher precedence than with prerelease.
    - Numeric identifiers compare numerically.
    - Alphanumeric identifiers compare as ASCII strings.
    - Numeric identifiers have lower precedence than alphanumeric.
    - Shorter prerelease list has lower precedence if all common identifiers equal.
    """
    if pre_a is None and pre_b is None:
        return 0
    if pre_a is None:
        return 1  # a > b (no prerelease > prerelease)
    if pre_b is None:
        return -1  # a < b (prerelease < no prerelease)

    # Both have prerelease: compare identifier by identifier
    for i in range(min(len(pre_a), len(pre_b))):
        type_a, val_a = pre_a[i]
        type_b, val_b = pre_b[i]

        if type_a == 'numeric' and type_b == 'numeric':
            # Both numeric: compare numerically
            if val_a != val_b:
                return -1 if val_a < val_b else 1
        elif type_a == 'numeric':
            # Numeric < alphanumeric
            return -1
        elif type_b == 'numeric':
            # Alphanumeric > numeric
            return 1
        else:
            # Both alphanumeric: compare as ASCII strings
            if val_a != val_b:
                return -1 if val_a < val_b else 1

    # All compared identifiers equal: shorter list has lower precedence
    if len(pre_a) < len(pre_b):
        return -1
    elif len(pre_a) > len(pre_b):
        return 1
    return 0
