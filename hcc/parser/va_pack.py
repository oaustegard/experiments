"""__builtin_va_arg_pack() / __builtin_va_arg_pack_len() specialisation.

gcc only defines these inside an always_inline variadic function: their value is
the extra arguments of the call that the inliner substitutes the body into. hcc has
no inliner, so the parser leaves placeholder BuiltinCall nodes ("va_arg_pack" and
"va_arg_pack_len") in such functions and this pass, run once after parsing, does
the equivalent of inlining at the call-site level:

* a function whose body holds placeholders is a "pack user" and is dropped from
  the translation unit;
* every call to a pack user is redirected to a fresh clone whose trailing
  parameters are the call's extra arguments (the pack);
* in each clone, `__builtin_va_arg_pack()` becomes the pack parameters and
  `__builtin_va_arg_pack_len()` the constant pack length.

Clones are processed with the same worklist, so pack forwarding through several
wrappers (f -> g -> h) resolves naturally. Static locals inside a pack user are
shared by its clones (they are not duplicated); this is a known limitation.
"""

from __future__ import annotations

import copy
from dataclasses import fields

from hcc import nodes as N
from hcc import typesys as T
from hcc import diag

PACK = "va_arg_pack"
PACK_LEN = "va_arg_pack_len"
PLACEHOLDERS = (PACK, PACK_LEN)
MAX_CLONES = 4096


def is_pack_placeholder(node) -> bool:
    """True for the __builtin_va_arg_pack() placeholder node (not the _len form)."""
    return type(node) is N.BuiltinCall and node.name == PACK


def _uses_pack(node) -> bool:
    """Iterative scan: label/case chains can nest tens of thousands of levels deep.
    Nodes are visited once (Switch.cases and Case.body share sub-chains)."""
    todo = [node]
    seen: set[int] = set()
    while todo:
        cur = todo.pop()
        if type(cur) is N.BuiltinCall and cur.name in PLACEHOLDERS:
            return True
        if isinstance(cur, N.Node):
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            for f in fields(cur):
                val = getattr(cur, f.name, None)
                if isinstance(val, (N.Node, list)):
                    todo.append(val)
        elif isinstance(cur, list):
            todo.extend(x for x in cur if isinstance(x, (N.Node, list)))
    return False


def _clone_tree(node, memo=None):
    """Structural copy of a node tree. Types and Obj references are shared (so
    VarRef identity and singleton types keep working); the Node objects and
    lists are fresh so that the clone can be rewritten independently. Nodes
    shared in the source (see _uses_pack) stay shared in the copy."""
    if memo is None:
        memo = {}
    if isinstance(node, N.Node):
        if id(node) in memo:
            return memo[id(node)]
        new = copy.copy(node)
        memo[id(node)] = new
        for f in fields(node):
            val = getattr(node, f.name, None)
            if isinstance(val, (N.Node, list)):
                setattr(new, f.name, _clone_tree(val, memo))
        return new
    if isinstance(node, list):
        return [_clone_tree(x, memo) for x in node]
    return node


class _PackSpecialiser:
    def __init__(self, tu):
        self.tu = tu
        self.clone_count = 0
        self.user_ids: set[int] = set()
        self.worklist: list[N.Function] = []

    def run(self) -> None:
        funcs = [o for o in self.tu.objs if isinstance(o, N.Function) and o.body is not None]
        for fn in funcs:
            if _uses_pack(fn.body):
                self.user_ids.add(id(fn))
        if not self.user_ids:
            return
        self.tu.objs = [o for o in self.tu.objs if id(o) not in self.user_ids]
        self.worklist = [fn for fn in funcs if id(fn) not in self.user_ids]
        while self.worklist:
            fn = self.worklist.pop()
            pack_vars = getattr(fn, "pack_vars", [])
            fn.body = self.rewrite(fn.body, pack_vars)

    # -- tree rewrite -------------------------------------------------------

    def rewrite(self, node, pack_vars, done=None):
        # `done` maps already-rewritten nodes (the AST is a DAG, see _uses_pack)
        # so each node is processed once and shared nodes stay shared.
        if done is None:
            done = {}
        if isinstance(node, list):
            out = []
            for x in node:
                if type(x) is N.BuiltinCall and x.name == PACK:
                    out.extend(N.VarRef(x.loc, v, ty=v.ty) for v in pack_vars)
                else:
                    out.append(self.rewrite(x, pack_vars, done))
            return out
        if not isinstance(node, N.Node):
            return node
        if id(node) in done:
            return done[id(node)]
        done[id(node)] = node
        if type(node) is N.BuiltinCall and node.name == PACK_LEN:
            res = N.IntLit(node.loc, len(pack_vars), ty=T.TY_INT)
            done[id(node)] = res
            return res
        for f in fields(node):
            val = getattr(node, f.name, None)
            if isinstance(val, (N.Node, list)):
                setattr(node, f.name, self.rewrite(val, pack_vars, done))
        if type(node) is N.Call:
            self.redirect_call(node)
        return node

    def redirect_call(self, call) -> None:
        fn = call.func
        if type(fn) is not N.VarRef or id(fn.var) not in self.user_ids:
            return
        target = fn.var
        nfixed = len(target.ty.params)
        if len(call.args) < nfixed:
            return
        extras = call.args[nfixed:]
        clone = self.make_clone(target, extras)
        call.func = N.VarRef(fn.loc, clone, ty=clone.ty)
        call.func_ty = clone.ty

    def make_clone(self, target, extras):
        self.clone_count += 1
        if self.clone_count > MAX_CLONES:
            diag.error(target.loc, f"too many __builtin_va_arg_pack specialisations of {target.name}")
        clone = copy.copy(target)
        clone.name = f"{target.name}__pack{self.clone_count}"
        clone.asm_name = None
        clone.is_static = True
        clone.params = [p for p in target.params]
        clone.locals = list(target.locals)
        clone.body = _clone_tree(target.body)
        clone.va_area = None
        clone.va_tmp = None
        pack_vars = []
        for i, arg in enumerate(extras):
            pty = arg.ty
            var = N.Var(name=f"__pack{i}", ty=pty, loc=arg.loc, is_local=True, is_param=True)
            pack_vars.append(var)
            clone.params.append(var)
            clone.locals.append(var)
        ty = target.ty.copy()
        ty.params = list(target.ty.params) + [v.ty for v in pack_vars]
        ty.is_variadic = False
        ty.has_proto = True
        clone.ty = ty
        clone.pack_vars = pack_vars
        clone.is_definition = True
        self.tu.objs.append(clone)
        self.worklist.append(clone)
        return clone


def specialise_pack_calls(tu) -> None:
    """Entry point: rewrite pack placeholders in a parsed translation unit."""
    _PackSpecialiser(tu).run()
