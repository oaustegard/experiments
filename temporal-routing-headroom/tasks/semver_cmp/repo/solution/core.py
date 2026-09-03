import re
from .util import _NUM, _ALNUM_ID, _BUILD_ID, _parse


def compare(a: str, b: str) -> int:
    na, pa = _parse(a)
    nb, pb = _parse(b)
    if na != nb:
        return -1 if na < nb else 1
    if pa is None and pb is None:
        return 0
    if pa is None:
        return 1
    if pb is None:
        return -1
    for ia, ib in zip(pa, pb):
        if ia != ib:
            return -1 if ia < ib else 1
    if len(pa) == len(pb):
        return 0
    return -1 if len(pa) < len(pb) else 1
