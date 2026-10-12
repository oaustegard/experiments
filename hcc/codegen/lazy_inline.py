"""On-demand emission of inline-only functions.

Parser/toplevel.py leaves `extern inline` definitions that have gnu89 / gnu_inline
semantics with a body but is_definition False: they provide no external
definition (GCC inlines them, e.g. the always_inline SSE intrinsics of
<emmintrin.h>). hcc does not inline, so such a function is emitted as a weak
symbol only when some emitted code or static data refers to it by name. This
also keeps unreferenced header bodies (which may use builtins hcc does not lower)
out of the output.
"""
from __future__ import annotations

import dataclasses

from hcc import nodes as N


def lazy_functions(tu) -> list:
    """Function objects with a body that are emitted only on demand."""
    return [o for o in tu.objs
            if isinstance(o, N.Function) and not o.is_definition and o.body is not None
            and not o.alias_of and not o.ifunc_of]


def _walk_refs(root, out: set) -> None:
    """Add to `out` the symbols of every function named by a VarRef or a
    relocation under `root` (a Node, list or tuple)."""
    stack = [root]
    while stack:
        x = stack.pop()
        if isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, N.VarRef):
            if isinstance(x.var, N.Function):
                out.add(x.var.symbol)
        elif isinstance(x, N.Node):
            for f in dataclasses.fields(x):
                v = getattr(x, f.name)
                if isinstance(v, (N.Node, list, tuple)):
                    stack.append(v)


def _data_refs(var, out: set) -> None:
    if var.init_data is not None:
        for r in var.init_data.relocs:
            out.add(r.label)


def needed_lazy(tu, lazy: list) -> set:
    """ids of the lazy functions reachable from the eagerly emitted objects."""
    lazy_ids = {id(f) for f in lazy}
    by_sym = {f.symbol: f for f in lazy}
    refs: set = set()
    for obj in tu.objs:
        if id(obj) in lazy_ids:
            continue
        if isinstance(obj, N.Function) and obj.is_definition:
            _walk_refs(obj.body, refs)
            for v in obj.locals:
                _data_refs(v, refs)
        elif isinstance(obj, N.Var):
            _data_refs(obj, refs)
    needed = set()
    todo = [s for s in refs if s in by_sym]
    while todo:
        sym = todo.pop()
        fn = by_sym[sym]
        if id(fn) in needed:
            continue
        needed.add(id(fn))
        more: set = set()
        _walk_refs(fn.body, more)
        for v in fn.locals:
            _data_refs(v, more)
        todo.extend(s for s in more if s in by_sym and id(by_sym[s]) not in needed)
    return needed
