"""`++z` / `--z` / `z++` / `z--` on a _Complex operand (C23 allows it).

The step is the complex value `delta + 0i`: it is applied through
Parser.complex_compound (`z = z + delta`), so the complex arithmetic and
conversions stay in complex.py.  The post form saves the old value in a
temporary; the lvalue address is evaluated only once (as for scalars)."""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.arith import _addressed_place
from hcc.typesys import TY_INT


def _side_effect_free(node) -> bool:
    if isinstance(node, (N.VarRef, N.IntLit, N.FloatLit)):
        return True
    if isinstance(node, (N.Member_, N.Deref)):
        return _side_effect_free(node.operand)
    return False


def complex_inc_dec(p, operand, delta: int, is_post: bool, loc):
    if _side_effect_free(operand):
        pre, place = [], (lambda: operand)
    else:
        pre, place = _addressed_place(operand, loc, p.new_temp)
    one = N.IntLit(loc, delta, ty=TY_INT)
    if not is_post:
        result = p.complex_compound("+", place(), one, loc)
    else:
        old = p.new_temp(operand.ty)
        save = p.typed(N.Assign(loc, p.typed(N.VarRef(loc, old)), p.typed(place())))
        update = p.complex_compound("+", place(), one, loc)
        result = p.typed(N.Comma(loc, save, p.typed(N.Comma(loc, update, p.typed(N.VarRef(loc, old))))))
    for e in reversed(pre):
        result = p.typed(N.Comma(loc, e, result))
    return result
