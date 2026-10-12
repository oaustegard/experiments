"""Functions returning a struct with a variably modified member (run-time size).

The caller's result block is allocated by its Call node (`Call.dyn_alloc`, a VLAAlloc
built by parser/dyn_struct_ret.py). Its address goes in the hidden first argument
(SysV MEMORY class, see abi.is_memory_struct), so the callee copies the run-time
number of bytes into it and returns that pointer.

DynStructRetMixin:
  hoist_dyn_allocs(node)   emits the allocations of the calls that appear directly in
                           a statement, before the statement itself (their %rsp
                           moves must not happen while operands are pushed)
  return_dyn_struct(node)  `return e;` with Return.size set: copies size bytes
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N


def _collect(obj, out: list) -> None:
    """Appends the dyn_alloc of every call in the expression(s) `obj`, in evaluation
    order. Statements (nested blocks, StmtExpr bodies) are not entered: their own
    statements hoist their allocations."""
    if isinstance(obj, N.Stmt):
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            _collect(x, out)
        return
    if not isinstance(obj, N.Expr):
        return
    for v in vars(obj).values():
        _collect(v, out)
    if isinstance(obj, N.Call) and obj.dyn_alloc is not None:
        out.append(obj.dyn_alloc)


class DynStructRetMixin:

    def hoist_dyn_allocs(self, node) -> None:
        allocs: list = []
        for v in vars(node).values():
            _collect(v, allocs)
        if not allocs:
            return
        if self.depth:
            diag.error(node, "unsupported: struct with a variably modified member returned "
                       "inside an expression with pending operands", "codegen")
        # The statement's allocations run unconditionally, in order: each one is carved
        # below the previous, so live results never overlap. The chain starts at the
        # block's VLA tail (parser's saved slot), never at another statement's temp.
        prev = None
        for a in allocs:
            saved = a.saved if prev is None else prev.var
            self.stmt_VLAAlloc(N.VLAAlloc(a.loc, a.var, saved, a.size))
            prev = a

    def return_dyn_struct(self, node) -> None:
        """%rax = address of the value (a struct with run-time size); copy it to the
        hidden result pointer and return that pointer in %rax."""
        self.gen_expr(node.expr)
        self.push()
        self.gen_expr(node.size)
        self.emit("mov %rax, %rcx")
        self.pop("%rsi")
        self.emit(f"mov {self.ret_ptr_offset}(%rbp), %rdi")
        self.emit("rep movsb")
        self.emit(f"mov {self.ret_ptr_offset}(%rbp), %rax")
