"""__builtin_object_size(ptr, type): compile-time size of the object `ptr`
points into, minus the offset of `ptr` in it.

The pointer is resolved statically to (whole object size, byte offset,
closest subobject size or None) by `_ptr`:
  * `&lv` and array lvalues used as pointers (through pointer casts),
  * `p + k` / `p - k` with a constant integer k (scaled by the element size),
  * `&a[i][j]...` subscripts (they are `*(a + i)` after parsing).
Anything else (parameters, dereferenced pointers, non-constant arithmetic)
is unknown: the result is (size_t)-1 for types 0/1 and 0 for types 2/3, as
GCC does when the size cannot be determined.

Types: bit 1 (value 2) selects the minimum estimate (unknown -> 0); bit 0
(value 1) selects the closest enclosing subobject instead of the whole
object. As at -O0 in GCC, a nonzero offset from the closest subobject is
unknown (-1 for type 1, 0 for type 3).

A trailing flexible array member is unknown unless its object has a static
initializer, whose byte length then bounds it. An incomplete non-extern
array (`char a[];`) is assumed to have one element.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.constexpr import try_eval_const

UNKNOWN_MAX = (1 << 64) - 1
_NON_SCALAR = (T.TypeKind.ARRAY, T.TypeKind.STRUCT, T.TypeKind.UNION,
               T.TypeKind.FUNC, T.TypeKind.VLA)


def _is_scalar(t):
    return t is not None and t.kind not in _NON_SCALAR


def _has_init(e):
    while isinstance(e, N.Member_):
        e = e.operand
    return isinstance(e, N.VarRef) and getattr(e.var, "init_data", None) is not None


def _lvalue(e):
    """(whole_size, offset, closest_size_or_None) of the object lvalue `e`."""
    if isinstance(e, N.VarRef):
        var = e.var
        ty = var.ty
        if ty.kind in (T.TypeKind.FUNC, T.TypeKind.VLA):
            return None
        if not ty.is_complete:
            # `extern char a[];` is unknown; a tentative `char a[];` has one element.
            if var.is_extern or ty.kind != T.TypeKind.ARRAY or ty.base is None:
                return None
            size = ty.base.size
        else:
            size = ty.size
        init = getattr(var, "init_data", None)
        if init is not None and ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
            size = max(size, len(init.data))
        return size, 0, (size if _is_scalar(ty) else None)
    if isinstance(e, N.Member_):
        base = _lvalue(e.operand)
        if base is None:
            return None
        whole, off, _ = base
        m = e.member
        mt = m.ty
        off += m.offset
        if mt.kind == T.TypeKind.ARRAY:
            if mt.array_len in (0, -1) and _has_init(e.operand):
                return whole, off, None
            # An array member is itself the object (flexible/zero-length: 0).
            return (mt.size if mt.array_len not in (0, -1) else 0), 0, None
        if mt.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
            return whole, off, None
        return whole, off, (mt.size if _is_scalar(mt) else None)
    if isinstance(e, N.Deref):
        res = _ptr(e.operand)
        if res is None:
            return None
        whole, off, closest = res
        return whole, off, (closest if _is_scalar(e.ty) else None)
    return None


def _strip(e):
    """Look through pointer/array-preserving casts."""
    while isinstance(e, N.Cast) and T.is_pointer_like(e.ty) \
            and T.is_pointer_like(e.operand.ty):
        e = e.operand
    return e


def _ptr(e):
    """(whole_size, offset, closest_or_None) for the pointer value `e`, or None."""
    e = _strip(e)
    if e.ty is None:
        return None
    if e.ty.kind == T.TypeKind.ARRAY:
        res = _lvalue(e)            # array decays to a pointer to its first element
        if res is None:
            return None
        return res[0], res[1], None
    if isinstance(e, N.AddrOf):
        return _lvalue(e.operand)
    if isinstance(e, N.Binary) and e.op in ("+", "-"):
        return _ptr_arith(e)
    return None


def _ptr_arith(e):
    lhs, rhs = _strip(e.lhs), _strip(e.rhs)
    if T.is_pointer_like(lhs.ty) and not T.is_pointer_like(rhs.ty):
        ptr, idx = lhs, rhs
    elif e.op == "+" and T.is_pointer_like(rhs.ty) and not T.is_pointer_like(lhs.ty):
        ptr, idx = rhs, lhs
    else:
        return None
    k = try_eval_const(idx)
    if k is None:
        return None
    if ptr.ty.kind == T.TypeKind.ARRAY:
        res = _lvalue(ptr)
        if res is None:
            return None
        whole, off, _ = res
        # Indexing an array of scalars: the indexed array is the closest subobject.
        closest = ptr.ty.size if _is_scalar(ptr.ty.base) else None
    else:
        res = _ptr(ptr)
        if res is None:
            return None
        whole, off, closest = res
    # The parser already scaled the integer operand to bytes.
    delta = k if e.op == "+" else -k
    return whole, off + delta, closest


def _static_object(ptr):
    return _ptr(ptr)


def lower_object_size(parser, tok):
    """Return the IntLit (type size_t) that __builtin_object_size evaluates to.
    Called with the '(' already consumed; the caller consumes ')'."""
    args = parser._args(2)
    mode = try_eval_const(args[1])
    if mode is None or not 0 <= mode <= 3:
        parser.error("__builtin_object_size: second argument must be a constant 0..3")
    known = _static_object(args[0])
    if known is None or known[1] < 0:
        value = 0 if mode & 2 else UNKNOWN_MAX
    else:
        whole, off, closest = known
        if not mode & 1:
            value = whole - off if off <= whole else 0
        elif off != 0:
            value = 0 if mode & 2 else UNKNOWN_MAX
        else:
            value = closest if closest is not None else whole
    return N.IntLit(tok.loc, value, ty=T.TY_ULONG)
