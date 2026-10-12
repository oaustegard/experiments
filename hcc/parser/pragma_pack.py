"""#pragma pack(...) state, applied to struct definitions that follow it.

The preprocessor forwards each `#pragma pack` as a marker token whose value
is the argument texts (commas kept). `apply_pack_pragma(parser, texts)` updates
`parser.pack_align` (0 = default natural alignment) and `parser.pack_stack`
for push/pop, following gcc's forms: pack(), pack(N), pack(push[,id][,N]),
pack(pop[,id]).
"""
from __future__ import annotations

from hcc.preprocess.directives import PRAGMA_PACK_MARK


def apply_pack_pragma(parser, texts: list[str]) -> None:
    args = [a for a in texts if a != ","]
    if not args:
        parser.pack_align = 0
        return
    head = args[0]
    if head == "push":
        parser.pack_stack.append(parser.pack_align)
        rest = args[1:]
    elif head == "pop":
        prev = parser.pack_stack.pop() if parser.pack_stack else 0
        parser.pack_align = prev
        return
    else:
        rest = args
    # optional identifier, then optional numeric alignment
    for a in rest:
        if a.isdigit():
            parser.pack_align = int(a)
            break
    else:
        if head == "push":
            return          # pack(push) keeps the current alignment
        if head.isdigit():
            parser.pack_align = int(head)


def at_pack_marker(parser) -> bool:
    return parser.at(PRAGMA_PACK_MARK)


def consume_pack_marker(parser) -> None:
    tok = parser.next()
    apply_pack_pragma(parser, tok.value or [])
