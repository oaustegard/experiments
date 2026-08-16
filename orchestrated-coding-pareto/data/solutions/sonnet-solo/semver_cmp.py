"""Compare Semantic Versioning 2.0.0 version strings by precedence."""

import re

_VERSION_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)

_NUMERIC_IDENT_RE = re.compile(r"^(0|[1-9]\d*)$")
_ALNUM_IDENT_RE = re.compile(r"^[0-9A-Za-z-]+$")
_BUILD_IDENT_RE = re.compile(r"^[0-9A-Za-z-]+$")


def _parse_prerelease_identifiers(prerelease: str):
    parts = prerelease.split(".")
    identifiers = []
    for part in parts:
        if part == "":
            raise ValueError("empty prerelease identifier")
        if _NUMERIC_IDENT_RE.match(part):
            identifiers.append((True, int(part)))
        elif _ALNUM_IDENT_RE.match(part):
            identifiers.append((False, part))
        else:
            raise ValueError(f"invalid prerelease identifier: {part!r}")
    return identifiers


def _validate_build(build: str):
    parts = build.split(".")
    for part in parts:
        if part == "":
            raise ValueError("empty build identifier")
        if not _BUILD_IDENT_RE.match(part):
            raise ValueError(f"invalid build identifier: {part!r}")


def _parse(version: str):
    if not isinstance(version, str):
        raise ValueError("version must be a string")
    match = _VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"invalid version string: {version!r}")

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    prerelease_raw = match.group("prerelease")
    build_raw = match.group("build")

    prerelease = None
    if prerelease_raw is not None:
        prerelease = _parse_prerelease_identifiers(prerelease_raw)

    if build_raw is not None:
        _validate_build(build_raw)

    return major, minor, patch, prerelease


def _compare_prerelease(a_pre, b_pre) -> int:
    # No prerelease has higher precedence than having one.
    if a_pre is None and b_pre is None:
        return 0
    if a_pre is None:
        return 1
    if b_pre is None:
        return -1

    for a_ident, b_ident in zip(a_pre, b_pre):
        a_is_num, a_val = a_ident
        b_is_num, b_val = b_ident

        if a_is_num and b_is_num:
            if a_val != b_val:
                return -1 if a_val < b_val else 1
        elif not a_is_num and not b_is_num:
            if a_val != b_val:
                return -1 if a_val < b_val else 1
        else:
            # numeric identifiers always have lower precedence than alphanumeric
            return -1 if a_is_num else 1

    if len(a_pre) != len(b_pre):
        return -1 if len(a_pre) < len(b_pre) else 1

    return 0


def compare(a: str, b: str) -> int:
    a_major, a_minor, a_patch, a_pre = _parse(a)
    b_major, b_minor, b_patch, b_pre = _parse(b)

    if a_major != b_major:
        return -1 if a_major < b_major else 1
    if a_minor != b_minor:
        return -1 if a_minor < b_minor else 1
    if a_patch != b_patch:
        return -1 if a_patch < b_patch else 1

    return _compare_prerelease(a_pre, b_pre)
