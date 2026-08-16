"""Semantic Versioning 2.0.0 comparison.

Public API: ``compare(a, b) -> int``.
"""

import re

__all__ = ["compare"]

_NUM_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_ALNUM_ID_RE = re.compile(r"^[0-9A-Za-z-]+$")
_DIGITS_RE = re.compile(r"^[0-9]+$")


def _parse_numeric(field, name, version):
    if not _NUM_RE.match(field):
        raise ValueError(
            "invalid %s version %r in %r" % (name, field, version)
        )
    return int(field)


def _parse_prerelease(pre, version):
    ids = pre.split(".")
    out = []
    for ident in ids:
        if ident == "":
            raise ValueError("empty prerelease identifier in %r" % (version,))
        if not _ALNUM_ID_RE.match(ident):
            raise ValueError(
                "illegal character in prerelease identifier %r in %r"
                % (ident, version)
            )
        if _DIGITS_RE.match(ident):
            if len(ident) > 1 and ident[0] == "0":
                raise ValueError(
                    "leading zero in numeric prerelease identifier %r in %r"
                    % (ident, version)
                )
            out.append((0, int(ident)))
        else:
            out.append((1, ident))
    return out


def _validate_build(build, version):
    for ident in build.split("."):
        if ident == "":
            raise ValueError("empty build identifier in %r" % (version,))
        if not _ALNUM_ID_RE.match(ident):
            raise ValueError(
                "illegal character in build identifier %r in %r"
                % (ident, version)
            )


def _parse(version):
    if not isinstance(version, str):
        raise ValueError("version must be a string, got %r" % (type(version),))
    if version == "":
        raise ValueError("empty version string")
    if version != version.strip() or re.search(r"\s", version):
        raise ValueError("whitespace in version %r" % (version,))

    rest = version

    # Split off build metadata at the first '+'.
    plus = rest.find("+")
    if plus >= 0:
        build = rest[plus + 1:]
        rest = rest[:plus]
        if build == "":
            raise ValueError("empty build metadata in %r" % (version,))
        _validate_build(build, version)

    # Split off prerelease at the first '-' after the version core.
    dash = rest.find("-")
    if dash >= 0:
        pre = rest[dash + 1:]
        rest = rest[:dash]
        if pre == "":
            raise ValueError("empty prerelease in %r" % (version,))
        prerelease = _parse_prerelease(pre, version)
    else:
        prerelease = None

    core = rest.split(".")
    if len(core) != 3:
        raise ValueError(
            "version core must be MAJOR.MINOR.PATCH in %r" % (version,)
        )
    major = _parse_numeric(core[0], "major", version)
    minor = _parse_numeric(core[1], "minor", version)
    patch = _parse_numeric(core[2], "patch", version)

    return (major, minor, patch), prerelease


def _cmp(x, y):
    if x < y:
        return -1
    if x > y:
        return 1
    return 0


def _compare_prerelease(pa, pb):
    # None means "no prerelease", which has higher precedence.
    if pa is None and pb is None:
        return 0
    if pa is None:
        return 1
    if pb is None:
        return -1
    for ia, ib in zip(pa, pb):
        kind_a, val_a = ia
        kind_b, val_b = ib
        if kind_a != kind_b:
            # numeric (0) < alphanumeric (1)
            return -1 if kind_a < kind_b else 1
        c = _cmp(val_a, val_b)
        if c:
            return c
    return _cmp(len(pa), len(pb))


def compare(a: str, b: str) -> int:
    """Compare two SemVer 2.0.0 strings.

    Returns -1 if ``a`` sorts before ``b``, 0 if they have equal precedence,
    and 1 if ``a`` sorts after ``b``. Build metadata is ignored.

    Raises ``ValueError`` if either argument is not a valid version string.
    """
    core_a, pre_a = _parse(a)
    core_b, pre_b = _parse(b)

    c = _cmp(core_a, core_b)
    if c:
        return c
    return _compare_prerelease(pre_a, pre_b)
