"""Initializer lowering.

    lower_local_init(init, var, typed) -> Expr
        Comma(MemZero(var), assignments...) for a local (or compound literal).
    serialize_global_init(init, ty) -> GlobalData
        bytes + relocations for static storage; errors if not constant.
"""
from __future__ import annotations

import dataclasses

from hcc.bidbits import bid_from_int
from hcc import diag
from hcc import nodes as N
from hcc.sema.arith import new_add
from hcc.sema.constexpr import NotConstant, eval_float, eval_reloc, float_to_int
from hcc import typesys as T
from hcc.floatbits import pack_float32, pack_float64
from hcc.halfbits import half_bits
from hcc.f128_lit import float128_bits
from hcc.sema.x87_widen import widened_nonfinite_x87


def _has_value(init) -> bool:
    if init.expr is not None:
        return True
    return bool(init.children) and any(_has_value(c) for _, c in N.present_children(init.children))


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
        if ty.kind == T.TypeKind.ARRAY and not ty.is_vector:
            diag.error(init.expr, "array initialized from non-constant array expression", "sema")
        out.append(typed(N.Assign(loc, lhs(), init.expr)))
        return
    if ty.kind == T.TypeKind.ARRAY and init.children:
        for i, child in N.present_children(init.children):
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
    gd = N.GlobalData(bytearray(max(ty.size, _flex_extent(init, ty))))
    _write(init, ty, gd, 0)
    return gd


def _flex_extent(init, ty) -> int:
    """Bytes needed by a struct's trailing flexible array member (`char b[]`)
    when it is initialized with elements past the struct's own size, e.g.
    `struct T t = { 1, .b[2] = 'a' };` needs 4 + 3 bytes of storage."""
    if ty.kind != T.TypeKind.STRUCT or not ty.members or init.expr is not None:
        return 0
    last = ty.members[-1]
    if not getattr(last, "open_array", False) or not isinstance(init.children, list):
        return 0
    child = init.children[-1]
    if child is None or child.expr is not None or not isinstance(child.children, list):
        return 0
    return last.offset + len(child.children) * last.ty.base.size


def _write(init, ty, gd, offset) -> None:
    if ty.is_reverse_sso and init.expr is None:
        # scalar_storage_order struct: no relocation can be byte-swapped, so any
        # address constant among its members is rejected (as gcc does).
        nrel = len(gd.relocs)
        _write_le(init, ty, gd, offset)
        if len(gd.relocs) != nrel:
            diag.error(_first_expr(init), "initializer element is not constant", "sema")
        return
    if not ty.is_rev_order or init.expr is None:
        return _write_le(init, ty, gd, offset)
    # scalar_storage_order("big-endian"): write the little-endian image, then
    # reverse its bytes in place.
    nrel = len(gd.relocs)
    _write_le(init, ty, gd, offset)
    if len(gd.relocs) != nrel:
        diag.error(init.expr, "address constant in a reverse-order struct "
                   "is not supported", "sema")
    gd.data[offset:offset + ty.size] = gd.data[offset:offset + ty.size][::-1]


def _first_expr(init):
    """The first expression found in an initializer tree (for diagnostics)."""
    if init.expr is not None:
        return init.expr
    for c in init.children or ():
        if c is not None:
            e = _first_expr(c)
            if e is not None:
                return e
    return init


def _write_le(init, ty, gd, offset) -> None:
    if ty.kind == T.TypeKind.ARRAY and init.expr is None:
        if init.children:
            sz = ty.base.size
            for i, child in N.present_children(init.children):
                _write(child, ty.base, gd, offset + sz * i)
        return
    if ty.kind == T.TypeKind.STRUCT and init.expr is None:
        if init.children:
            for m, child in zip(ty.members, init.children):
                if m.is_bitfield:
                    if child.expr is None:
                        continue
                    v = _const_int(child.expr, m.ty)
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
    if T.is_struct_like(ty) or ty.kind == T.TypeKind.ARRAY:
        src = _static_compound_literal(expr)
        if src is None:
            diag.error(expr, "initializer element is not constant", "sema")
        gd.data[offset:offset + ty.size] = src.data[:ty.size]
        gd.relocs.extend(dataclasses.replace(r, offset=offset + r.offset) for r in src.relocs)
        return
    diff = _label_difference(expr)
    if diff is not None:
        if ty.size > 8:     # a label difference is a 32/64-bit assembler reloc
            diag.error(expr, "initializer element is not computable at load time", "sema")
        label, minus, extra, addend = diff
        gd.relocs.append(N.Reloc(offset, label, addend, ty.size, minus, extra))
        return
    inner = _same_size_operand(expr, ty)
    src = _static_compound_literal(inner) if _is_anon_compound(inner, ty) else None
    if src is not None:                  # scalar file-scope compound literal, e.g. (int){1}
        gd.data[offset:offset + ty.size] = src.data[:ty.size]
        gd.relocs.extend(dataclasses.replace(r, offset=offset + r.offset) for r in src.relocs)
        return
    if T.is_flonum(ty):
        try:
            v = eval_float(expr)
        except NotConstant:
            diag.error(expr, "initializer element is not constant", "sema")
        if ty.kind == T.TypeKind.FLOAT16:
            gd.data[offset:offset + 2] = half_bits(v).to_bytes(2, "little")
        elif ty.kind in (T.TypeKind.FLOAT, T.TypeKind.DOUBLE):
            size = 4 if ty.kind == T.TypeKind.FLOAT else 8
            exact = _exact_float_bits(expr, ty)
            if exact is not None:        # keeps signaling NaN payloads (no float round trip)
                gd.data[offset:offset + size] = exact.to_bytes(size, "little")
            elif size == 4:
                gd.data[offset:offset + 4] = pack_float32(v)
            else:
                gd.data[offset:offset + 8] = pack_float64(v)
        elif ty.kind == T.TypeKind.FLOAT128:
            exact = _exact_bits(expr, 127, T.TypeKind.FLOAT128)
            if exact is None:
                exact = float128_bits(v)
            gd.data[offset:offset + 16] = exact.to_bytes(16, "little")
        else:
            exact = _exact_bits(expr, 79, T.TypeKind.LDOUBLE)
            if exact is None:            # a double/float inf or NaN widened to long double
                wide = widened_nonfinite_x87(expr)
                if wide is not None:
                    gd.data[offset:offset + 10] = wide
                    return
            gd.data[offset:offset + 10] = (exact.to_bytes(10, "little") if exact is not None
                                           else float_to_x87(v))
        return
    if T.is_decimal(ty):
        gd.data[offset:offset + ty.size] = _decimal_static_bits(expr, ty)
        return
    try:
        label, v = _static_scalar(expr, ty)
    except NotConstant:
        diag.error(expr, "initializer element is not constant", "sema")
    if label is not None:
        gd.relocs.append(N.Reloc(offset, label, v))
        return
    size = ty.size
    gd.data[offset:offset + size] = (v & ((1 << (size * 8)) - 1)).to_bytes(size, "little")


def _static_compound_literal(expr):
    """File-scope compound literal used as a static initializer: its GlobalData, else None."""
    if isinstance(expr, N.Comma) and expr.lvalue and isinstance(expr.rhs, N.VarRef):
        expr = expr.rhs                     # block-scope compound literal (its temp object)
    if isinstance(expr, N.VarRef):
        if not getattr(expr.var, "is_local", False):
            return getattr(expr.var, "init_data", None)
        return getattr(expr.var, "const_data", None)    # constexpr block-scope aggregate
    return None


def _is_anon_compound(expr, ty) -> bool:
    """True for a VarRef to the anonymous global the parser makes for a file-scope scalar
    compound literal of the target's size. String literals are anonymous too (and decay
    to arrays), so arrays never qualify."""
    return (isinstance(expr, N.VarRef) and expr.var.name.startswith(".L..anon.")
            and expr.var.ty.kind != T.TypeKind.ARRAY and expr.var.ty.size == ty.size)


def _same_size_operand(expr, ty):
    """Strip casts that keep the byte representation (e.g. `(int){1}` cast to int)."""
    while isinstance(expr, N.Cast) and expr.operand.ty is not None and expr.operand.ty.size == ty.size \
            and not T.is_flonum(expr.operand.ty) and not T.is_flonum(ty):
        expr = expr.operand
    return expr


def _label_form(expr):
    """Linear form of an address expression: (terms, const) where terms is a list of
    (sign, label) and const an integer, or None when `expr` is not made of address
    constants, integer constants, casts and +/- (and unary -)."""
    while isinstance(expr, N.Cast):
        expr = expr.operand
    # pointer difference is lowered to (a - b) / elemsize; only elemsize 1 is a plain byte difference
    if isinstance(expr, N.Binary) and expr.op == "/" and isinstance(expr.rhs, N.IntLit) and expr.rhs.value == 1:
        expr = expr.lhs
    if isinstance(expr, N.Binary) and expr.op in ("+", "-"):
        lhs = _label_form(expr.lhs)
        rhs = _label_form(expr.rhs)
        if lhs is None or rhs is None:
            return None
        sg = 1 if expr.op == "+" else -1
        return lhs[0] + [(sg * s, lab) for s, lab in rhs[0]], lhs[1] + sg * rhs[1]
    if isinstance(expr, N.Unary) and expr.op == "-":
        inner = _label_form(expr.operand)
        if inner is None:
            return None
        return [(-s, lab) for s, lab in inner[0]], -inner[1]
    try:
        label, v = eval_reloc(expr)
    except NotConstant:
        return None
    return ([] if label is None else [(1, label)]), v


def _label_difference(expr):
    """An address expression with at least one label subtracted, e.g. `&&a - &&b`,
    `(&&a - &&b) - 1` or `(&&a - &&b) + (&&c - &&d)`, as (label, minus, extra, addend)
    for an assembler-resolved reloc: value = label - minus + sum(extra) + addend.
    None when the expression has no subtracted label or is not a valid linear form."""
    form = _label_form(expr)
    if form is None or not any(s < 0 for s, _ in form[0]):
        return None
    terms, addend = form
    if sum(s for s, _ in terms) not in (0, 1):
        return None
    pos = [lab for s, lab in terms if s > 0]
    neg = [lab for s, lab in terms if s < 0]
    label = pos[0] if pos else None
    minus = neg[0] if neg else None
    if minus is None or label is None:
        return None
    rest = [(1, lab) for lab in pos[1:]] + [(-1, lab) for lab in neg[1:]]
    return label, minus, tuple(rest), addend


def _static_scalar(expr, ty):
    """(label, value) of a static scalar initializer converted to `ty`.
    A floating constant stored into an integer object is converted like a cast
    (the implicit conversion may not have been materialised as an ast.Cast)."""
    if T.is_integer(ty) and T.is_flonum(getattr(expr, "ty", None) or T.TY_VOID):
        return None, float_to_int(eval_float(expr), ty)
    return eval_reloc(expr)


def _decimal_static_bits(expr, ty) -> bytes:
    """BID bits of a static _Decimal object initialised from an integer constant.
    Other constants (floating, decimal-to-decimal) are not folded yet."""
    inner = expr
    while type(inner) is N.Cast and T.is_integer(inner.operand.ty):
        inner = inner.operand
    if not T.is_integer(getattr(inner, "ty", None) or T.TY_VOID):
        diag.error(expr, f"unsupported static initializer for '{ty}'", "sema")
    try:
        label, v = _static_scalar(inner, inner.ty)
    except NotConstant:
        diag.error(expr, "initializer element is not constant", "sema")
    if label is not None:
        diag.error(expr, "unsupported static initializer for decimal type", "sema")
    return bid_from_int(v, ty.size)


def _const_int(expr, ty=T.TY_INT) -> int:
    try:
        label, v = _static_scalar(expr, ty)
    except NotConstant:
        diag.error(expr, "initializer element is not constant", "sema")
    return v


def _write_bits(gd, off, m, v) -> None:
    size = m.rev_bytes or m.ty.size
    # A packed bitfield may straddle its declared type's storage unit.
    size = max(size, (m.bit_offset + m.bit_width + 7) // 8)
    order = "big" if m.rev_bytes else "little"     # scalar_storage_order("big-endian")
    cur = int.from_bytes(gd.data[off:off + size], order)
    mask = ((1 << m.bit_width) - 1) << m.bit_offset
    cur = (cur & ~mask) | ((v << m.bit_offset) & mask)
    gd.data[off:off + size] = cur.to_bytes(size, order)


def _exact_bits(node, sign_bit: int, kind=None) -> int | None:
    """Exact IEEE/x87 encoding of a floating literal (float, double or long double
    width; `sign_bit` is its sign position), optionally negated: -1e-4940L would
    underflow to -0.0 through a Python float, and a signaling NaN is quieted by one.
    None when the expression is not a literal."""
    if isinstance(node, N.FloatLit):
        if kind is not None and (node.ty is None or node.ty.kind != kind):
            return None                  # e.g. a double NaN literal: encode its value instead
        return node.bits
    if isinstance(node, N.Unary) and node.op == "-":
        bits = _exact_bits(node.operand, sign_bit, kind)
        return None if bits is None else bits ^ (1 << sign_bit)
    return None


def _exact_float_bits(node, ty) -> int | None:
    """Exact float/double encoding of a literal of the same type (optionally negated);
    None when the Python float value is the only source (not a literal)."""
    if isinstance(node, N.FloatLit):
        if node.bits is None or node.ty is None or node.ty.kind != ty.kind:
            return None
        return node.bits
    if isinstance(node, N.Unary) and node.op == "-":
        bits = _exact_float_bits(node.operand, ty)
        return None if bits is None else bits ^ (1 << (ty.size * 8 - 1))
    return None


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
