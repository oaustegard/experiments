"""Initializer lowering.

    lower_local_init(init, var, typed) -> Expr
        Comma(MemZero(var), assignments...) for a local (or compound literal).
    serialize_global_init(init, ty) -> GlobalData
        bytes + relocations for static storage; errors if not constant.
"""
from __future__ import annotations

import struct

from hcc import diag
from hcc import nodes as N
from hcc.sema.arith import new_add
from hcc.sema.constexpr import NotConstant, eval_float, eval_reloc
from hcc import typesys as T


def _has_value(init) -> bool:
    if init.expr is not None:
        return True
    return bool(init.children) and any(_has_value(c) for c in init.children)


def lower_local_init(init, var, typed) -> N.Expr:
    loc = var.loc

    def var_ref():
        return typed(N.VarRef(loc, var))
    exprs: list[N.Expr] = []
    _lower(init, var.ty, var_ref, exprs, typed, loc)
    node = typed(N.MemZero(loc, var))
    for e in exprs:
        node = typed(N.Comma(loc, node, e))
    return node


def _lower(init, ty, lhs, out, typed, loc) -> None:
    if init.expr is not None:
        out.append(typed(N.Assign(loc, lhs(), init.expr)))
        return
    if ty.kind == T.TypeKind.ARRAY and init.children:
        for i, child in enumerate(init.children):
            if not _has_value(child):
                continue
            def elem(i=i):
                return typed(N.Deref(loc, new_add(lhs(), N.IntLit(loc, i, ty=T.TY_LONG), loc)))
            _lower(child, ty.base, elem, out, typed, loc)
        return
    if ty.kind == T.TypeKind.STRUCT and init.children:
        for m, child in zip(ty.members, init.children):
            if not _has_value(child):
                continue
            def mem(m=m):
                return N.Member_(loc, lhs(), m, ty=m.ty)
            _lower(child, m.ty, mem, out, typed, loc)
        return
    if ty.kind == T.TypeKind.UNION and init.children:
        m = init.mem or (ty.members[0] if ty.members else None)
        if m is None:
            return
        child = init.children[m.idx]
        if _has_value(child):
            _lower(child, m.ty, lambda: N.Member_(loc, lhs(), m, ty=m.ty), out, typed, loc)


def serialize_global_init(init, ty) -> N.GlobalData:
    gd = N.GlobalData(bytearray(ty.size))
    _write(init, ty, gd, 0)
    return gd


def _write(init, ty, gd, offset) -> None:
    if ty.kind == T.TypeKind.ARRAY and init.expr is None:
        if init.children:
            sz = ty.base.size
            for i, child in enumerate(init.children):
                _write(child, ty.base, gd, offset + sz * i)
        return
    if ty.kind == T.TypeKind.STRUCT and init.expr is None:
        if init.children:
            for m, child in zip(ty.members, init.children):
                if m.is_bitfield:
                    if child.expr is None:
                        continue
                    v = _const_int(child.expr)
                    _write_bits(gd, offset + m.offset, m, v)
                else:
                    _write(child, m.ty, gd, offset + m.offset)
        return
    if ty.kind == T.TypeKind.UNION and init.expr is None:
        if init.children and ty.members:
            m = init.mem or ty.members[0]
            _write(init.children[m.idx], m.ty, gd, offset)
        return
    if init.expr is None:
        return
    expr = init.expr
    if T.is_struct_like(ty):
        src = _static_compound_literal(expr)
        if src is None:
            diag.error(expr, "initializer element is not constant", "sema")
        gd.data[offset:offset + ty.size] = src.data[:ty.size]
        gd.relocs.extend(N.Reloc(offset + r.offset, r.label, r.addend, r.size, r.minus)
                         for r in src.relocs)
        return
    diff = _label_difference(expr)
    if diff is not None:
        a, b, addend = diff
        gd.relocs.append(N.Reloc(offset, a, addend, ty.size, b))
        return
    if T.is_flonum(ty):
        try:
            v = eval_float(expr)
        except NotConstant:
            diag.error(expr, "initializer element is not constant", "sema")
        if ty.kind == T.TypeKind.FLOAT:
            gd.data[offset:offset + 4] = struct.pack("<f", v)
        elif ty.kind == T.TypeKind.DOUBLE:
            gd.data[offset:offset + 8] = struct.pack("<d", v)
        else:
            gd.data[offset:offset + 10] = float_to_x87(v)
        return
    try:
        label, v = eval_reloc(expr)
    except NotConstant:
        diag.error(expr, "initializer element is not constant", "sema")
    if label is not None:
        gd.relocs.append(N.Reloc(offset, label, v))
        return
    size = ty.size
    gd.data[offset:offset + size] = (v & ((1 << (size * 8)) - 1)).to_bytes(size, "little")


def _static_compound_literal(expr):
    """File-scope compound literal used as a static initializer: its GlobalData, else None."""
    if isinstance(expr, N.VarRef) and not getattr(expr.var, "is_local", False):
        src = getattr(expr.var, "init_data", None)
        if src is not None:
            return src
    return None


def _label_difference(expr):
    """`&&a - &&b` (possibly cast) -> (a, b, addend) for an assembler-resolved reloc, else None."""
    while isinstance(expr, N.Cast):
        expr = expr.operand
    # pointer difference is lowered to (a - b) / elemsize; only elemsize 1 is a plain byte difference
    if isinstance(expr, N.Binary) and expr.op == "/" and isinstance(expr.rhs, N.IntLit) and expr.rhs.value == 1:
        expr = expr.lhs
    if not (isinstance(expr, N.Binary) and expr.op == "-"):
        return None
    try:
        la, a = eval_reloc(expr.lhs)
        lb, b = eval_reloc(expr.rhs)
    except NotConstant:
        return None
    if la is None or lb is None:
        return None
    return la, lb, a - b


def _const_int(expr) -> int:
    try:
        label, v = eval_reloc(expr)
    except NotConstant:
        diag.error(expr, "initializer element is not constant", "sema")
    return v


def _write_bits(gd, off, m, v) -> None:
    size = m.ty.size
    cur = int.from_bytes(gd.data[off:off + size], "little")
    mask = ((1 << m.bit_width) - 1) << m.bit_offset
    cur = (cur & ~mask) | ((v << m.bit_offset) & mask)
    gd.data[off:off + size] = cur.to_bytes(size, "little")


def float_to_x87(v: float) -> bytes:
    """Encode a Python float as an 80-bit x87 extended value (10 bytes)."""
    import math
    if v == 0:
        return (b"\0" * 9) + (b"\x80" if math.copysign(1, v) < 0 else b"\0")
    if math.isinf(v) or math.isnan(v):
        sign = 0x8000 if v < 0 else 0
        mant = (1 << 63) | (1 << 62 if math.isnan(v) else 0)
        return mant.to_bytes(8, "little") + (sign | 0x7FFF).to_bytes(2, "little")
    sign = 0x8000 if v < 0 else 0
    m, e = math.frexp(abs(v))          # v = m * 2**e, 0.5 <= m < 1
    mant = int(m * (1 << 64))
    exp = e - 1 + 16383
    return mant.to_bytes(8, "little") + (sign | exp).to_bytes(2, "little")
