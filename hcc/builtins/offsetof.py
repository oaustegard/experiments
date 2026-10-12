"""__builtin_offsetof(type, member-designator): a constant, or a run-time sum when
an array subscript in the designator is not constant (GCC)."""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.builtins import builtin_parser
from hcc.parser.dyn_members import in_dyn_region
from hcc.parser.structs import complete_struct, find_member
from hcc.sema.constexpr import try_eval_const


@builtin_parser("__builtin_offsetof")
def _parse_offsetof(self, tok):
    ty = self.typename()
    self.expect(",")
    off = 0
    dyn = []      # run-time terms: array subscripts that are not integer constants
    cur = ty
    first = True
    while True:
        if not first and self.accept("->"):
            # GCC extension: `arr->m` in offsetof means `arr[0].m`
            if cur.kind == T.TypeKind.ARRAY:
                cur = cur.base
            first = True
        if first or self.accept("."):
            first = False
            name = self.expect_ident()
            path = find_member(cur, name.text)
            if path is None:
                self.error(f"'{cur}' has no member named '{name.text}'", name)
            for m in path:
                cs = complete_struct(cur)
                if T.is_dyn_struct(cs) and in_dyn_region(cs, cs.members.index(m)):
                    self.error("unsupported: offsetof a member after a variably modified member", name)
                off += m.offset
                cur = m.ty
        elif self.accept("["):
            idx = self.conditional()
            self.expect("]")
            k = try_eval_const(idx)
            if k is not None:
                off += k * cur.base.size
            else:       # GCC: a non-constant designator is evaluated at run time
                term = N.Binary(tok.loc, "*", N.Cast(tok.loc, idx, ty=T.TY_LONG),
                                N.IntLit(tok.loc, cur.base.size, ty=T.TY_LONG))
                dyn.append(self.typed(term))
            cur = cur.base
        else:
            break
    if not dyn:
        return N.IntLit(tok.loc, off, ty=T.TY_ULONG)
    node = N.IntLit(tok.loc, off, ty=T.TY_LONG)
    for term in dyn:
        node = self.typed(N.Binary(tok.loc, "+", node, term))
    return self.typed(N.Cast(tok.loc, node, ty=T.TY_ULONG))
