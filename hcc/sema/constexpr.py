"""Constant expression evaluation.

    eval_const(node) -> int                  integer constant expression (error if not)
    try_eval_const(node) -> int | None
    eval_reloc(node) -> (label | None, int)  address constant: &global + offset
    is_const_expr(node) -> bool
Results are wrapped to the node's type width/signedness.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.typesys import TypeKind, is_flonum, is_integer


class NotConstant(Exception):
    pass


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


def eval_float(node) -> float:
    if isinstance(node, N.FloatLit):
        return node.value
    if isinstance(node, N.Cast) and is_flonum(node.operand.ty):
        return eval_float(node.operand)
    if isinstance(node, N.Unary) and node.op == "-":
        return -eval_float(node.operand)
    if isinstance(node, N.Binary) and node.op in "+-*/":
        a, b = eval_float(node.lhs), eval_float(node.rhs)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else float("inf")}[node.op]
    if isinstance(node, N.Cond):
        return eval_float(node.then) if _num(node.cond) else eval_float(node.els)
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
        if op in ("+", "-"):
            la, a = _eval(node.lhs)
            lb, b = _eval(node.rhs)
            if lb is not None:
                if op == "-" or la is not None:
                    raise NotConstant
                return lb, wrap(a + b, node.ty)
            return la, wrap(a + b if op == "+" else a - b, node.ty) if la is None else (a + b if op == "+" else a - b)
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
        _eval(node.lhs)          # must itself be constant, else folding would drop its side effects
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
        if node.var.ty.kind in (TypeKind.ARRAY, TypeKind.FUNC) and not getattr(node.var, "is_local", False):
            return node.var.symbol, 0
        raise NotConstant
    if t is N.StrLit:
        raise NotConstant
    if t is N.LabelAddr:                 # GNU &&label: same symbol codegen emits (context.user_label)
        return f".L.{node.func}.{node.label}", 0
    if t is N.Member_ and node.ty.kind == TypeKind.ARRAY:
        return _eval_addr(node)
    if t is N.Deref and node.ty.kind in (TypeKind.ARRAY, TypeKind.FUNC):
        return _eval(node.operand)
    if t is N.BuiltinCall and node.name == "__builtin_constant_p":
        return None, builtin_constant_p(node.args[0])
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


def _eval_addr(node):
    t = type(node)
    if t is N.VarRef:
        if getattr(node.var, "is_local", False):
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
