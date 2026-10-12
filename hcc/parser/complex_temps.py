"""Result temporaries for complex-valued expressions (C99 _Complex).

Complex rvalues are addresses (like struct values). Every complex-valued
Binary/Unary/Cast/ComplexLit needs an anonymous local to hold its two parts,
and every complex-valued Call a `ret_buffer`. Implicit casts are created during
sema, so the parser attaches the buffers in one walk per function body after
the body is complete (`attach_complex_temps`).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T

_BUF_NODES = (N.Binary, N.Unary, N.Cast, N.ComplexLit)


def attach_complex_temps(parser, body) -> None:
    seen: set[int] = set()
    stack = [body]
    while stack:
        n = stack.pop()
        if isinstance(n, (list, tuple)):
            stack.extend(n)
            continue
        if not isinstance(n, N.Node) or id(n) in seen:
            continue
        seen.add(id(n))
        ty = getattr(n, "ty", None)
        if ty is not None and T.is_complex(ty):
            if isinstance(n, _BUF_NODES) and n.result_buf is None:
                n.result_buf = parser.new_temp(ty, n.loc)
            elif isinstance(n, N.Call) and n.ret_buffer is None:
                n.ret_buffer = parser.new_temp(ty, n.loc)
        for f in getattr(n, "__dataclass_fields__", {}):
            if f not in _SKIP_FIELDS:
                v = getattr(n, f, None)
                if isinstance(v, (N.Node, list, tuple)):
                    stack.append(v)


_SKIP_FIELDS = frozenset({"ty", "var", "member", "loc", "scope", "func_ty", "result_buf",
                          "ret_buffer"})
