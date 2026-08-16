def compare(a: str, b: str) -> int:
    """Compare two semantic version strings per SemVer 2.0.0.

    Args:
        a: First version string (MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD])
        b: Second version string (MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD])

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b

    Raises:
        ValueError: If either version string is invalid per SemVer 2.0.0
    """

    def parse_version(v):
        """Parse and validate a semantic version string."""
        # Check for invalid prefixes and whitespace
        if not v or v[0] == 'v' or any(c.isspace() for c in v):
            raise ValueError("Invalid version string")

        # Parse structure: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
        # Split off BUILD metadata first (always ignored)
        if '+' in v:
            version_part, build = v.split('+', 1)
            if not build:
                raise ValueError("Invalid version string")
            # Validate build: non-empty dot-separated identifiers of letters/digits/hyphens
            build_ids = build.split('.')
            if not all(build_ids):
                raise ValueError("Invalid version string")
            for bid in build_ids:
                if not all(c.isalnum() or c == '-' for c in bid):
                    raise ValueError("Invalid version string")
        else:
            version_part = v

        # Split off PRERELEASE
        if '-' in version_part:
            version_nums, prerelease = version_part.split('-', 1)
            if not prerelease:
                raise ValueError("Invalid version string")
            prerel_ids = prerelease.split('.')
            if not all(prerel_ids):  # Empty identifier
                raise ValueError("Invalid version string")
            parsed_prerel = []
            for pid in prerel_ids:
                if pid.isdigit():
                    # Numeric: no leading zeros unless exactly "0"
                    if len(pid) > 1 and pid[0] == '0':
                        raise ValueError("Invalid version string")
                    parsed_prerel.append((0, int(pid)))
                elif all(c.isalnum() or c == '-' for c in pid):
                    # Alphanumeric: contains at least one non-digit, letters/digits/hyphens only
                    parsed_prerel.append((1, pid))
                else:
                    raise ValueError("Invalid version string")
        else:
            version_nums = version_part
            parsed_prerel = None

        # Parse MAJOR.MINOR.PATCH
        parts = version_nums.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid version string")

        major, minor, patch = parts

        # Validate each component: non-negative integer without leading zeros
        for component in [major, minor, patch]:
            if not component or not component.isdigit():
                raise ValueError("Invalid version string")
            if len(component) > 1 and component[0] == '0':
                raise ValueError("Invalid version string")

        return (int(major), int(minor), int(patch), parsed_prerel)

    def compare_prerelease(pre_a, pre_b):
        """Compare two prerelease version lists."""
        # Both have no prerelease
        if pre_a is None and pre_b is None:
            return 0
        # Version without prerelease > version with prerelease
        if pre_a is None:
            return 1
        if pre_b is None:
            return -1

        # Compare identifier by identifier
        for id_a, id_b in zip(pre_a, pre_b):
            type_a, val_a = id_a
            type_b, val_b = id_b

            # Numeric vs numeric
            if type_a == 0 and type_b == 0:
                if val_a != val_b:
                    return -1 if val_a < val_b else 1
            # Alphanumeric vs alphanumeric
            elif type_a == 1 and type_b == 1:
                if val_a != val_b:
                    return -1 if val_a < val_b else 1
            # Numeric vs alphanumeric: numeric is lower
            elif type_a == 0:
                return -1
            else:
                return 1

        # If all compared identifiers are equal, shorter list has lower precedence
        if len(pre_a) < len(pre_b):
            return -1
        elif len(pre_a) > len(pre_b):
            return 1
        return 0

    # Parse both versions
    major_a, minor_a, patch_a, prerel_a = parse_version(a)
    major_b, minor_b, patch_b, prerel_b = parse_version(b)

    # Compare MAJOR.MINOR.PATCH numerically
    if major_a != major_b:
        return -1 if major_a < major_b else 1
    if minor_a != minor_b:
        return -1 if minor_a < minor_b else 1
    if patch_a != patch_b:
        return -1 if patch_a < patch_b else 1

    # Compare prerelease (BUILD is ignored)
    return compare_prerelease(prerel_a, prerel_b)
