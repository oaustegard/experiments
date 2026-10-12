"""Peephole rewrites on the emitted line buffer (see context.CodeGen.emit).

Fuses a comparison result that is immediately re-tested for a branch:

    cmp %edi, %eax / setCC %al / movzbl %al, %eax / cmp $0, %eax / jne L
        -> cmp %edi, %eax / jCC L

The flags of `cmp $0, %eax` equal those of the original comparison, so the
branch can test them directly; `je` inverts the condition.
"""

from __future__ import annotations

_CC = {"sete": "e", "setne": "ne", "setl": "l", "setge": "ge", "setle": "le",
       "setg": "g", "setb": "b", "setae": "ae", "setbe": "be", "seta": "a"}
_NEG = {"e": "ne", "ne": "e", "l": "ge", "ge": "l", "le": "g", "g": "le",
        "b": "ae", "ae": "b", "be": "a", "a": "be"}


def fuse_setcc_branch(out: list, line: str) -> bool:
    """If `line` is a je/jne following the setcc pattern at the end of `out`,
    rewrite the buffer in place and return True (caller must not append)."""
    if len(out) < 3:
        return False
    if out[-1] != "  cmp $0, %eax" or out[-2] != "  movzbl %al, %eax":
        return False
    setline = out[-3].strip()
    parts = setline.split()
    if len(parts) != 2 or parts[1] != "%al" or parts[0] not in _CC:
        return False
    jparts = line.split()
    if len(jparts) != 2 or jparts[0] not in ("je", "jne"):
        return False
    cc = _CC[parts[0]]
    if jparts[0] == "je":
        cc = _NEG[cc]
    del out[-3:]
    out.append(f"  j{cc} {jparts[1]}")
    return True
