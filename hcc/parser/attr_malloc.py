"""`__attribute__((malloc(deallocator[, index])))` argument checks.

GCC requires the optional deallocator to name a function declared earlier,
whose first parameter is a pointer (the object it releases). hcc records
nothing for malloc itself; it only validates the argument.
"""
from __future__ import annotations

from hcc import typesys as T
from hcc.tokens import TokenKind


# Library allocation functions that GCC also exposes as implicit
# `__builtin_<name>` functions; value: whether the first parameter is a pointer.
_BUILTIN_LIB_FUNCS = {
    "free": True,
    "realloc": True,
    "malloc": False,
    "calloc": False,
    "aligned_alloc": False,
}


def _builtin_deallocator_ok(name: str):
    """Return (known, first_param_is_pointer) for an implicit `__builtin_` name."""
    if not name.startswith("__builtin_"):
        return None
    lib = name[len("__builtin_"):]
    if lib not in _BUILTIN_LIB_FUNCS:
        return None
    return _BUILTIN_LIB_FUNCS[lib]


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
    if vs is None or vs.var is None:
        ptr_first = _builtin_deallocator_ok(first.text)
        if ptr_first is None:
            parser.error("'malloc' attribute argument 1 does not name a function", where)
        elif not ptr_first:
            parser.error("'malloc' attribute argument 1 must take a pointer type "
                         "as its first argument", where)
        return
    fty = vs.var.ty
    if fty is None or fty.kind != T.TypeKind.FUNC:
        parser.error("'malloc' attribute argument 1 does not name a function", where)
        return
    if not fty.params or fty.params[0].kind != T.TypeKind.PTR:
        parser.error("'malloc' attribute argument 1 must take a pointer type "
                     "as its first argument", where)
