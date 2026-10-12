"""Constant expression evaluation.

    eval_const(node) -> int                  integer constant expression (error if not)
    try_eval_const(node) -> int | None
    eval_reloc(node) -> (label | None, int)  address constant: &global + offset
    is_const_expr(node) -> bool
Results are wrapped to the node's type width/signedness.
"""
from __future__ import annotations

import struct

from hcc import diag
from hcc import nodes as N
from hcc.floatbits import pack_float32
from hcc.typesys import TypeKind, is_flonum, is_integer, round_to_half
from hcc.sema.constexpr_math import fold_int_call, fold_math_call
from hcc.sema.constexpr_rel import fold_rel_equal
from hcc.sema.constexpr_cx import is_cx_init as _is_cx_init


class NotConstant(Exception):
    pass


# __builtin_bswapN (parser/builtins.py) -> byte width
_BSWAP_BYTES = {"bswap16": 2, "bswap32": 4, "bswap64": 8}
# operand sizes that x86-64 performs lock-free (matches codegen/atomics.py _SUF)
_ALWAYS_LOCK_FREE_SIZES = frozenset([1, 2, 4, 8])


def wrap(v: int, ty) -> int:
    if ty is None or not is_integer(ty) and ty.kind != TypeKind.PTR:
        return v
    if ty.kind == TypeKind.BOOL:
        return int(v != 0)
    bits = ty.size * 8
    v &= (1 << bits) - 1
    if not ty.is_unsigned and v >= 1 << (bits - 1):
        v -= 1 << bits
    return v


def eval_const(node) -> int:
    try:
        label, v = _eval(node)
    except NotConstant:
        diag.error(node, "expression is not an integer constant", "sema")
    if label is not None:
        diag.error(node, "expression is not an integer constant", "sema")
    return v


def try_eval_const(node):
    try:
        label, v = _eval(node)
    except NotConstant:
        return None
    return None if label is not None else v


def eval_reloc(node):
    """Return (label, addend); raises NotConstant if not an address constant."""
    return _eval(node)


def is_const_expr(node) -> bool:
    return try_eval_const(node) is not None


def float_to_int(f: float, ty) -> int:
    """Fold a floating -> integer conversion the way GCC does: out-of-range
    values saturate to the target range and NaN becomes 0."""
    if ty.kind == TypeKind.BOOL:
        return int(f != 0)
    if f != f:
        return 0
    bits = ty.size * 8
    if ty.is_unsigned:
        lo, hi = 0, (1 << bits) - 1
    else:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if f >= float(hi):
        return hi
    if f <= float(lo):
        return lo
    return int(f)


def _ieee_div(a: float, b: float) -> float:
    """IEEE 754 division: x/0 is a signed infinity, 0/0 and NaN/0 are NaN."""
    import math
    if b != 0:
        return a / b
    if a == 0 or math.isnan(a):
        return math.nan
    return math.copysign(math.inf, a) * math.copysign(1.0, b)


def _round_to_flonum(v: float, ty) -> float:
    """Value of the float constant `v` once it is stored in `ty` (_Float16 and
    float round; double and wider are kept as Python floats)."""
    if ty.kind == TypeKind.FLOAT16:
        return round_to_half(v)
    if ty.kind == TypeKind.FLOAT:
        return struct.unpack("<f", pack_float32(v))[0]
    return v


def eval_float(node) -> float:
    if isinstance(node, N.FloatLit):
        return _round_to_flonum(node.value, node.ty)
    if isinstance(node, N.Cast) and is_flonum(node.operand.ty):
        return _round_to_flonum(eval_float(node.operand), node.ty)
    if isinstance(node, N.Unary) and node.op == "-":
        return -eval_float(node.operand)
    if isinstance(node, N.Binary) and node.op in "+-*/":
        a, b = eval_float(node.lhs), eval_float(node.rhs)
        if node.op == "/":
            return _ieee_div(a, b)
        return {"+": a + b, "-": a - b, "*": a * b}[node.op]
    if isinstance(node, N.Cond):
        return eval_float(node.then) if _num(node.cond) else eval_float(node.els)
    if isinstance(node, N.Call):
        v = fold_math_call(node, eval_float)
        if v is None:
            raise NotConstant
        return v
    return float(_num(node))


def _num(node) -> int:
    label, v = _eval(node)
    if label is not None:
        raise NotConstant
    return v


def _eval(node):
    t = type(node)
    if t is N.IntLit:
        return None, node.value
    if t is N.FloatLit:
        # A float value has no integer-constant meaning on its own (e.g. 0.1 is
        # true, not 0). Float->int casts fold via eval_float; truth tests on
        # floats fall back to the runtime path.
        raise NotConstant
    if t is N.Binary:
        op = node.op
        if op == "-":
            try:
                return _eval_sub(node)
            except NotConstant:
                # GCC folds `p - (p + k)` for a pointer whose value is unknown: both
                # operands share one base, so only the byte offsets matter.
                d = _same_base_diff(node.lhs, node.rhs)
                if d is None:
                    raise
                return None, wrap(d, node.ty)
        if op in ("+", "-"):
            la, a = _eval(node.lhs)
            lb, b = _eval(node.rhs)
            if lb is not None:
                if op == "-":
                    if la == lb:             # &x.f - &x: same object, plain integer difference
                        return None, wrap(a - b, node.ty)
                    raise NotConstant
                if la is not None:
                    raise NotConstant
                return lb, wrap(a + b, node.ty)
            return la, wrap(a + b if op == "+" else a - b, node.ty) if la is None else (a + b if op == "+" else a - b)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            folded = _fold_address_compare(node)
            if folded is not None:
                return None, folded
        if op in ("==", "!=", "<", "<=", ">", ">=") and \
                (is_flonum(node.lhs.ty) or is_flonum(node.rhs.ty)):
            fa, fb = eval_float(node.lhs), eval_float(node.rhs)     # float comparison folds too
            return None, int({"==": fa == fb, "!=": fa != fb, "<": fa < fb, "<=": fa <= fb,
                              ">": fa > fb, ">=": fa >= fb}[op])
        if op in ("==", "!=") and type(node.lhs) is N.Binary and type(node.rhs) is N.Binary:
            r = fold_rel_equal(node, try_eval_const)     # e.g. (a < b) != (b > a)
            if r is not None:
                return None, r
        a, b = _num(node.lhs), _num(node.rhs)
        u = node.lhs.ty.is_unsigned if node.lhs.ty is not None else False
        if op in ("/", "%"):
            if b == 0:
                raise NotConstant
            if u:
                a, b = wrap(a, node.lhs.ty) & ((1 << 64) - 1), b & ((1 << 64) - 1)
                a &= (1 << (node.lhs.ty.size * 8)) - 1
                b &= (1 << (node.lhs.ty.size * 8)) - 1
            q = abs(a) // abs(b) * (1 if (a >= 0) == (b >= 0) else -1)
            return None, wrap(q if op == "/" else a - q * b, node.ty)
        if op == ">>":
            if u:
                a &= (1 << (node.lhs.ty.size * 8)) - 1
            return None, wrap(a >> b, node.ty)
        if op in ("<", "<=", ">", ">=") and u:
            mask = (1 << (node.lhs.ty.size * 8)) - 1
            a, b = a & mask, b & mask
        r = {"*": lambda: a * b, "&": lambda: a & b, "|": lambda: a | b, "^": lambda: a ^ b,
             "<<": lambda: a << b, "==": lambda: int(a == b), "!=": lambda: int(a != b),
             "<": lambda: int(a < b), "<=": lambda: int(a <= b), ">": lambda: int(a > b),
             ">=": lambda: int(a >= b)}[op]()
        return None, wrap(r, node.ty)
    if t is N.Unary:
        v = _num(node.operand)
        if node.op == "-":
            return None, wrap(-v, node.ty)
        if node.op == "~":
            return None, wrap(~v, node.ty)
        return None, int(not v)
    if t is N.Logical:
        a = _num(node.lhs)
        if node.op == "&&":
            return None, int(bool(a) and bool(_num(node.rhs)))
        return None, int(bool(a) or bool(_num(node.rhs)))
    if t is N.Cond:
        return _eval(node.then) if _num(node.cond) else _eval(node.els)
    if t is N.Comma:
        if not _is_cx_init(node.lhs):
            _eval(node.lhs)      # must itself be constant, else folding would drop its side effects
        return _eval(node.rhs)
    if t is N.Cast:
        if is_flonum(node.operand.ty) and is_integer(node.ty):
            return None, float_to_int(eval_float(node.operand), node.ty)
        label, v = _eval(node.operand)
        if label is not None:
            if node.ty.size != 8:
                raise NotConstant
            return label, v
        return None, wrap(v, node.ty)
    if t is N.AddrOf:
        return _eval_addr(node.operand)
    if t is N.VarRef:
        if node.var.ty.kind in (TypeKind.ARRAY, TypeKind.FUNC) and _has_link_address(node.var):
            return node.var.symbol, 0
        return None, _const_var_value(node.var)
    if t is N.StrLit:
        raise NotConstant
    if t is N.LabelAddr:                 # GNU &&label: same symbol codegen emits (context.user_label)
        return f".L.{node.func}.{node.label}", 0
    if t is N.Member_ and node.ty.kind == TypeKind.ARRAY:
        return _eval_addr(node)
    if t is N.Member_ and is_integer(node.ty):
        return None, _const_member_value(node)
    if t is N.Deref and node.ty.kind in (TypeKind.ARRAY, TypeKind.FUNC):
        return _eval(node.operand)
    if t is N.Deref and is_integer(node.ty):
        try:
            return None, _string_char(node)
        except NotConstant:
            return None, _const_object_read(node)
    if t is N.BuiltinCall and node.name == "__builtin_constant_p":
        return None, builtin_constant_p(node.args[0])
    if t is N.BuiltinCall and node.name == "atomic_always_lock_free":
        size = _num(node.args[0])     # size is constant; ptr only feeds alignment (ignored)
        return None, int(size in _ALWAYS_LOCK_FREE_SIZES)
    if t is N.BuiltinCall and node.name in _BSWAP_BYTES:
        nbytes = _BSWAP_BYTES[node.name]
        v = _num(node.args[0]) & ((1 << (8 * nbytes)) - 1)
        return None, wrap(int.from_bytes(v.to_bytes(nbytes, "little"), "big"), node.ty)
    if t is N.Call and is_integer(node.ty):          # integer-valued libm call, e.g. lround
        v = fold_int_call(node, eval_float)
        if v is not None:
            return None, wrap(v, node.ty)
    raise NotConstant


def builtin_constant_p(node) -> int:
    """Value of __builtin_constant_p(node): 1 for constant expressions and for
    string literals (possibly array-decayed), which gcc folds to 1."""
    inner = node
    while isinstance(inner, N.Cast):
        inner = inner.operand
    if type(inner) is N.StrLit:
        return 1
    if type(inner) is N.VarRef and getattr(inner.var, "is_string_literal", False):
        return 1
    return int(try_eval_const(node) is not None)


_SWAPPED_REL = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _fold_address_compare(node):
    """Fold a comparison involving address constants: two addresses of the same
    object compare by offset; an object's address compared with null is known
    (objects are never null). Returns 0/1, or None if not foldable here."""
    try:
        la, a = _eval(node.lhs)
        lb, b = _eval(node.rhs)
    except NotConstant:
        return None
    op = node.op
    if la is not None and lb is not None:
        if la != lb:
            return None
        return int(_compare(op, a, b))
    if la is None and lb is None:
        return None
    # exactly one side is an address: it must be compared against a null constant
    if la is None:                       # 0 OP &x: swap so the address is on the left
        op, b = _SWAPPED_REL[op], a
    if b != 0:
        return None
    if op == "==":
        return 0
    if op == "!=":
        return 1
    return int(op in (">", ">="))


def _compare(op, a, b) -> bool:
    return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]


def _has_link_address(var) -> bool:
    """True when `var` has a link-time symbol usable in a static initializer. Block-scope
    objects, thread-local objects (their address is per-thread) and GNU nested functions
    (which need the enclosing frame) do not."""
    return not (getattr(var, "is_local", False) or getattr(var, "is_tls", False)
                or getattr(var, "outer", None) is not None)


def _eval_addr(node):
    t = type(node)
    if t is N.VarRef:
        if not _has_link_address(node.var):
            raise NotConstant
        return node.var.symbol, 0
    if t is N.Deref:
        return _eval(node.operand)
    if t is N.Member_:
        label, v = _eval_addr(node.operand)
        return label, v + node.member.offset
    if t is N.Cast:
        return _eval_addr(node.operand)
    raise NotConstant


_CMP_OPS = ("==", "!=", "<", "<=", ">", ">=")


def _const_var_value(var) -> int:
    """GCC extension: a const-qualified, non-volatile integer global (or a scalar compound
    literal in a static initializer) whose initializer is a plain number folds to that
    number when read (e.g. `static const int le = 1;`). C23 `constexpr` objects (also
    block-scope ones, and null pointers) are constants too."""
    ty = var.ty
    cx = getattr(var, "is_constexpr", False)
    if getattr(var, "is_local", False) and getattr(var, "const_value", None) is not None:
        return wrap(var.const_value, ty)            # block-scope const, see parser/initializers.py
    data = getattr(var, "init_data", None)
    if data is None or data.relocs:
        raise NotConstant
    if ty.kind == TypeKind.PTR and cx and not ty.is_volatile:
        return int.from_bytes(bytes(data.data[:8]), "little")
    if not is_integer(ty) or ty.is_volatile:
        raise NotConstant
    if not (ty.is_const or cx or getattr(var, "is_compound_literal", False)):
        raise NotConstant
    raw = int.from_bytes(bytes(data.data[:ty.size]), "little")
    return wrap(raw, ty)


def _const_member_value(node) -> int:
    """GCC extension: `v.m` (or `v.a.b`) where `v` is a const-qualified, non-volatile
    global with a plain-number initializer folds to the member's value. Bitfields and
    anything else raise NotConstant."""
    off, base = 0, node
    while True:
        if type(base) is N.Comma and _is_cx_init(base.lhs):   # (constexpr T){...}.m
            base = base.rhs
        elif type(base) is N.Member_:
            if base.member.is_bitfield or base.operand.ty is None:
                raise NotConstant
            off += base.member.offset
            base = base.operand
        else:
            break
    if type(base) is not N.VarRef:
        raise NotConstant
    var = base.var
    data = getattr(var, "init_data", None)
    if data is None or data.relocs or var.ty.is_volatile or not (var.ty.is_const or var.is_constexpr):
        raise NotConstant
    ty = node.ty
    if off < 0 or off + ty.size > len(data.data):
        raise NotConstant
    raw = int.from_bytes(bytes(data.data[off:off + ty.size]), "little")
    return wrap(raw, ty)


def _label_cmp(op, la, a, lb, b) -> int:
    """Comparison where at least one side is an address constant. Addresses of
    objects are non-null, so `&obj` compared with 0 folds; same-symbol addresses
    compare by offset. Anything else is left to the run-time path."""
    if la is not None and lb is not None:
        if la != lb:
            raise NotConstant
        return int(_cmp(op, a, b))
    label, off, other = (la, a, b) if la is not None else (lb, b, a)
    if other != 0:
        raise NotConstant
    if op in ("==", "<", "<="):
        return 0
    if op in ("!=", ">", ">="):
        return 1
    raise NotConstant


def _cmp(op, a, b) -> bool:
    return {"==": a == b, "!=": a != b, "<": a < b, "<=": a <= b,
            ">": a > b, ">=": a >= b}[op]


def _is_string_var(node) -> bool:
    while type(node) is N.Cast:
        node = node.operand
    return type(node) is N.VarRef and getattr(node.var, "is_string_literal", False)


def _string_char(node) -> int:
    """Value of `"literal"[k]` (also `k["literal"]`, `*("literal" + k)`): GCC folds
    a read of a string literal at a constant index. NotConstant otherwise."""
    ptr = node.operand
    while type(ptr) is N.Cast:
        ptr = ptr.operand
    k = 0
    if type(ptr) is N.Binary and ptr.op == "+":
        if _is_string_var(ptr.lhs):
            ptr, k = ptr.lhs, _num(ptr.rhs)
        elif _is_string_var(ptr.rhs):
            ptr, k = ptr.rhs, _num(ptr.lhs)
    if not _is_string_var(ptr):
        raise NotConstant
    while type(ptr) is N.Cast:
        ptr = ptr.operand
    data = ptr.var.init_data.data
    if not 0 <= k < len(data):
        raise NotConstant
    return wrap(data[k], node.ty)


def _const_object_read(node) -> int:
    """GCC extension: an integer read at a constant offset inside a const-qualified,
    non-volatile static object with a plain-byte initializer, e.g. `a[27]` for
    `static const char a[] = "..."` or `e[1].c[4]` for a const member. Raises
    NotConstant otherwise."""
    var, off, is_const = _ptr_obj(node.operand)
    data = getattr(var, "init_data", None)
    ty = node.ty
    if data is None or data.relocs or ty.is_volatile:
        raise NotConstant
    if not (is_const or ty.is_const):
        raise NotConstant
    if off < 0 or off + ty.size > len(data.data):
        raise NotConstant
    raw = int.from_bytes(bytes(data.data[off:off + ty.size]), "little")
    return wrap(raw, ty)


def _decayed_obj(node) -> tuple:
    """_ptr_obj for a pointer, or _obj_addr for an array lvalue (which decays)."""
    if node.ty is not None and node.ty.kind == TypeKind.ARRAY:
        return _obj_addr(node)
    return _ptr_obj(node)


def _elem_const(ty) -> tuple:
    """(is_const, is_volatile) of an object type: `const char a[]` qualifies the element."""
    while ty.kind == TypeKind.ARRAY:
        ty = ty.base
    return ty.is_const, ty.is_volatile


def _ptr_obj(node) -> tuple:
    """(static VarRef var, byte offset, const-qualified path) of a pointer-valued
    expression pointing into a non-local object. Raises NotConstant otherwise."""
    t = type(node)
    if t is N.Cast and node.operand.ty is not None and \
            node.operand.ty.kind in (TypeKind.PTR, TypeKind.ARRAY):
        return _ptr_obj(node.operand)
    if t is N.VarRef and node.var.ty.kind == TypeKind.ARRAY:
        if getattr(node.var, "is_local", False):
            raise NotConstant
        c, v = _elem_const(node.var.ty)
        if v:
            raise NotConstant
        return node.var, 0, c
    if t is N.AddrOf:
        return _obj_addr(node.operand)
    if t is N.Deref and node.ty is not None and node.ty.kind == TypeKind.ARRAY:
        return _obj_addr(node)
    if t is N.Binary and node.op in ("+", "-") and node.ty.kind == TypeKind.PTR:
        if node.lhs.ty is not None and node.lhs.ty.kind in (TypeKind.PTR, TypeKind.ARRAY):
            var, off, c = _decayed_obj(node.lhs)
            k = _num(node.rhs)
        elif node.op == "+":
            var, off, c = _decayed_obj(node.rhs)
            k = _num(node.lhs)
        else:
            raise NotConstant
        return var, off + (k if node.op == "+" else -k), c
    raise NotConstant


def _obj_addr(node) -> tuple:
    """(static VarRef var, byte offset, const-qualified path) of an array/struct lvalue."""
    t = type(node)
    if t is N.VarRef:
        if getattr(node.var, "is_local", False):
            raise NotConstant
        c, v = _elem_const(node.var.ty)
        if v:
            raise NotConstant
        return node.var, 0, c
    if t is N.Deref:
        return _ptr_obj(node.operand)
    if t is N.Member_ and not node.member.is_bitfield:
        var, off, c = _obj_addr(node.operand)
        mc, mv = _elem_const(node.member.ty)
        if mv:
            raise NotConstant
        return var, off + node.member.offset, c or mc
    raise NotConstant


def _eval_sub(node) -> tuple:
    la, a = _eval(node.lhs)
    lb, b = _eval(node.rhs)
    if lb is not None:
        if la is None or la != lb:
            raise NotConstant
        return None, wrap(a - b, node.ty)
    return la, wrap(a - b, node.ty) if la is None else (a - b)


def _same_base_diff(lhs, rhs):
    """Byte difference `lhs - rhs` when both are pointer values (possibly cast to
    an integer) derived from the same object/pointer variable; else None."""
    lhs, rhs = _strip_int_cast(lhs), _strip_int_cast(rhs)
    try:
        ka, oa = _ptr_val(lhs)
        kb, ob = _ptr_val(rhs)
    except NotConstant:
        return None
    return oa - ob if ka == kb else None


def _strip_int_cast(node):
    while type(node) is N.Cast and node.operand.ty is not None and \
            node.operand.ty.kind in (TypeKind.PTR, TypeKind.ARRAY):
        node = node.operand
    return node


def _ptr_val(node) -> tuple:
    """(key, byte offset) of a pointer-valued expression. Keys identify the object
    (`addr`) or the unknown value of a pointer variable (`val`)."""
    t = type(node)
    if t is N.Cast and node.operand.ty is not None and \
            node.operand.ty.kind in (TypeKind.PTR, TypeKind.ARRAY):
        return _ptr_val(node.operand)
    if t is N.VarRef and not node.var.ty.is_volatile:
        if node.var.ty.kind == TypeKind.ARRAY:
            return ("addr", id(node.var)), 0
        if node.var.ty.kind == TypeKind.PTR:
            return ("val", id(node.var)), 0
    if t is N.AddrOf:
        return _ptr_addr(node.operand)
    if t is N.Deref and node.ty is not None and node.ty.kind == TypeKind.ARRAY:
        return _ptr_addr(node)          # an array lvalue decays to its own address
    if t is N.Binary and node.op in ("+", "-") and node.ty.kind == TypeKind.PTR:
        key, off = _ptr_val(node.lhs)
        k = _num(node.rhs)
        return key, off + (k if node.op == "+" else -k)
    raise NotConstant


def _ptr_addr(node) -> tuple:
    """(key, byte offset) of the address of an lvalue."""
    t = type(node)
    if t is N.VarRef and not node.var.ty.is_volatile:
        return ("addr", id(node.var)), 0
    if t is N.Deref:
        return _ptr_val(node.operand)
    if t is N.Member_:
        key, off = _ptr_addr(node.operand)
        return key, off + node.member.offset
    raise NotConstant
