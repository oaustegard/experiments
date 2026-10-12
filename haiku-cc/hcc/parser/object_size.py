"""__builtin_object_size(ptr, type): compile-time size of the object `ptr`
points into, minus the offset of `ptr` in it.

Only objects that are visible statically are sized: the object named by
`&obj`, `&obj.member...` or an array lvalue used as a pointer (following
pointer casts). Anything else (parameters, dereferenced pointers, arithmetic
on them) is unknown: the result is (size_t)-1 for types 0/1 and 0 for types
2/3, as GCC does when the size cannot be determined.
A trailing flexible array member is unknown unless its object has a static
initializer, whose byte length then bounds it.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.constexpr import try_eval_const

UNKNOWN_MAX = (1 << 64) - 1


def _object_of(e):
    """(object_size, offset_of_e_in_object) for the lvalue `e`, or None."""
    if isinstance(e, N.VarRef):
        var = e.var
        ty = var.ty
        if ty.kind in (T.TypeKind.FUNC,) or not ty.is_complete:
            return None
        size = ty.size
        init = getattr(var, "init_data", None)
        if init is not None and ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
            size = max(size, len(init.data))
        return size, 0
    if isinstance(e, N.Member_):
        base = _object_of(e.operand)
        if base is None:
            return None
        size, off = base
        m = e.member
        if m.ty.kind == T.TypeKind.ARRAY and m.ty.array_len in (0, -1) \
                and not _has_init(e.operand):
            return None     # trailing flexible array without an initializer
        return size, off + m.offset
    return None


def _has_init(e):
    while isinstance(e, N.Member_):
        e = e.operand
    return isinstance(e, N.VarRef) and getattr(e.var, "init_data", None) is not None


def _strip_casts(e):
    while isinstance(e, N.Cast) and T.is_pointer_like(e.operand.ty):
        e = e.operand
    return e


def _static_object(ptr):
    """(object_size, offset) the pointer expression `ptr` refers to, or None."""
    ptr = _strip_casts(ptr)
    if isinstance(ptr, N.AddrOf):
        return _object_of(ptr.operand)
    if ptr.ty is not None and ptr.ty.kind == T.TypeKind.ARRAY:
        return _object_of(ptr)
    return None


def lower_object_size(parser, tok):
    """Return the IntLit (type size_t) that __builtin_object_size evaluates to.
    Called with the '(' already consumed; the caller consumes ')'."""
    args = parser._args(2)
    mode = try_eval_const(args[1])
    if mode is None or not 0 <= mode <= 3:
        parser.error("__builtin_object_size: second argument must be a constant 0..3")
    known = _static_object(args[0])
    if known is None or known[1] > known[0]:
        value = 0 if mode & 2 else UNKNOWN_MAX
    else:
        value = known[0] - known[1]
    return N.IntLit(tok.loc, value, ty=T.TY_ULONG)
