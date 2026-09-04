import re


_NUM = re.compile(r"^(0|[1-9]\d*)$")


_ALNUM_ID = re.compile(r"^[0-9A-Za-z-]+$")


_BUILD_ID = re.compile(r"^[0-9A-Za-z-]+$")


def _parse(v: str):
    if not isinstance(v, str):
        raise ValueError("not a string")
    build = None
    if "+" in v:
        v, build = v.split("+", 1)
        if not build:
            raise ValueError("empty build")
        for ident in build.split("."):
            if not ident or not _BUILD_ID.match(ident):
                raise ValueError(f"bad build identifier: {ident!r}")
    pre = None
    if "-" in v:
        core, pre = v.split("-", 1)
    else:
        core = v
    parts = core.split(".")
    if len(parts) != 3:
        raise ValueError(f"core must be MAJOR.MINOR.PATCH: {core!r}")
    nums = []
    for p in parts:
        if not _NUM.match(p):
            raise ValueError(f"bad numeric field: {p!r}")
        nums.append(int(p))
    pre_ids = None
    if pre is not None:
        if pre == "":
            raise ValueError("empty prerelease")
        pre_ids = []
        for ident in pre.split("."):
            if not ident or not _ALNUM_ID.match(ident):
                raise ValueError(f"bad prerelease identifier: {ident!r}")
            if ident.isdigit():
                if not _NUM.match(ident):
                    raise ValueError(f"leading zero in numeric identifier: {ident!r}")
                pre_ids.append((1, ident))
            else:
                pre_ids.append((1, ident))
    return nums, pre_ids
