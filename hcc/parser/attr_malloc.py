"""`__attribute__((malloc(deallocator[, index])))` argument checks.

GCC requires the optional deallocator to name a function declared earlier,
whose first parameter is a pointer (the object it releases). hcc records
nothing for malloc itself; it only validates the argument.
"""
from __future__ import annotations

from hcc import typesys as T
from hcc.tokens import TokenKind


def check_malloc_attr(parser, args_start: int, args: list) -> None:
    """Reject malloc deallocator arguments that GCC rejects."""
    if not args:
        return
    where = parser.toks[args_start - 1] if args_start > 0 else None
    first = args[0]
    if first.kind != TokenKind.IDENT or (len(args) > 1 and args[1].text != ","):
        parser.error("'malloc' attribute argument 1 does not name a function", where)
        return
    vs = parser.find_var(first.text)
    fty = vs.var.ty if vs is not None and vs.var is not None else None
    if fty is None or fty.kind != T.TypeKind.FUNC:
        parser.error("'malloc' attribute argument 1 does not name a function", where)
        return
    if not fty.params or fty.params[0].kind != T.TypeKind.PTR:
        parser.error("'malloc' attribute argument 1 must take a pointer type "
                     "as its first argument", where)
