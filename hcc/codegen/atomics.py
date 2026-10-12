"""Lowering of the __atomic_* builtins (parsed in parser/atomic_builtins.py).

AtomicMixin provides builtin_atomic_<name> for every builtin in
ATOMIC_BUILTINS. The operand size (1/2/4/8) comes from the pointee type of the
first argument. Read-modify-write operations are a `lock cmpxchg` retry loop,
exchanges are `xchg` (implicitly locked), plain loads/stores are aligned movs.
Memory-order arguments are accepted and ignored: x86 is strongly ordered and
every RMW is already a full barrier.
"""
from __future__ import annotations

from hcc import diag
from hcc.sema.constexpr import try_eval_const

_SUF = {1: "b", 2: "w", 4: "l", 8: "q"}
_REG = {
    "a": ("%al", "%ax", "%eax", "%rax"),
    "c": ("%cl", "%cx", "%ecx", "%rcx"),
    "d": ("%dl", "%dx", "%edx", "%rdx"),
    "si": ("%sil", "%si", "%esi", "%rsi"),
}
_IDX = {1: 0, 2: 1, 4: 2, 8: 3}


def _r(name: str, sz: int) -> str:
    return _REG[name][_IDX[sz]]


class AtomicMixin:

    def _atomic_size(self, node):
        pt = node.args[0].ty.base
        sz = pt.size
        if sz not in _SUF:
            diag.error(node, f"unsupported __atomic operand size {sz}", "codegen")
        return sz, not pt.is_unsigned

    def _gen_args(self, exprs, regs) -> None:
        """Evaluate exprs left to right; exprs[i] ends up in the 64-bit regs[i].
        Clobbers %rax and the registers in regs."""
        for e in exprs[:-1]:
            self.gen_expr(e)
            self.push()
        self.gen_expr(exprs[-1])
        self.emit(f"mov %rax, {regs[-1]}")
        for reg in reversed(regs[:-1]):
            self.pop(reg)

    def _extend_rax(self, sz: int, signed: bool) -> None:
        """Widen the low `sz` bytes of %rax to the full value of the pointee type."""
        if sz == 1:
            self.emit("movsbq %al, %rax" if signed else "movzbq %al, %rax")
        elif sz == 2:
            self.emit("movswq %ax, %rax" if signed else "movzwq %ax, %rax")
        elif sz == 4:
            self.emit("movslq %eax, %rax" if signed else "movl %eax, %eax")

    def _rmw(self, node, op: str, fetch_old: bool) -> None:
        sz, signed = self._atomic_size(node)
        s, a, c, si = _SUF[sz], _r("a", sz), _r("c", sz), _r("si", sz)
        self._gen_args(node.args[:2], ["%rdi", "%rsi"])
        self.emit(f"mov{s} (%rdi), {a}")
        loop = self.new_label()
        self.label(loop)
        self.emit("mov %rax, %rcx")
        if op == "nand":
            self.emit(f"and{s} {si}, {c}")
            self.emit(f"not{s} {c}")
        else:
            self.emit(f"{op}{s} {si}, {c}")
        self.emit(f"lock cmpxchg{s} {c}, (%rdi)")
        self.emit(f"jne {loop}")
        if not fetch_old:
            self.emit("mov %rcx, %rax")  # new value; for fetch_* %rax is the old one
        self._extend_rax(sz, signed)

    def _load_n(self, node) -> None:
        self._atomic_size(node)
        self.gen_expr(node.args[0])
        self.load(node.ty)

    def _store_n(self, node) -> None:
        sz, _ = self._atomic_size(node)
        self._gen_args(node.args[:2], ["%rdi", "%rsi"])
        self.emit(f"xchg{_SUF[sz]} {_r('si', sz)}, (%rdi)")

    def _exchange_n(self, node) -> None:
        sz, signed = self._atomic_size(node)
        self._gen_args(node.args[:2], ["%rdi", "%rsi"])
        self.emit(f"xchg{_SUF[sz]} {_r('si', sz)}, (%rdi)")
        self.emit("mov %rsi, %rax")
        self._extend_rax(sz, signed)

    def _cmpxchg(self, node, generic: bool) -> None:
        """compare_exchange_n(p, expected*, desired) or the generic form where
        desired is a pointer too. Result (bool) in %eax; *expected is updated."""
        sz, _ = self._atomic_size(node)
        s, a, d = _SUF[sz], _r("a", sz), _r("d", sz)
        self._gen_args(node.args[:3], ["%rdi", "%rsi", "%rdx"])
        if generic:
            self.emit(f"mov{s} (%rdx), {d}")
        self.emit(f"mov{s} (%rsi), {a}")
        self.emit(f"lock cmpxchg{s} {d}, (%rdi)")
        self.emit("sete %cl")
        self.emit(f"mov{s} {a}, (%rsi)")  # on success %rax == *expected already
        self.emit("movzbl %cl, %eax")

    def _generic_load(self, node) -> None:
        sz, _ = self._atomic_size(node)
        s, c = _SUF[sz], _r("c", sz)
        self._gen_args(node.args[:2], ["%rdi", "%rsi"])
        self.emit(f"mov{s} (%rdi), {c}")
        self.emit(f"mov{s} {c}, (%rsi)")

    def _generic_store(self, node) -> None:
        sz, _ = self._atomic_size(node)
        s, c = _SUF[sz], _r("c", sz)
        self._gen_args(node.args[:2], ["%rdi", "%rsi"])
        self.emit(f"mov{s} (%rsi), {c}")
        self.emit(f"xchg{s} {c}, (%rdi)")

    def _generic_exchange(self, node) -> None:
        sz, _ = self._atomic_size(node)
        s, c = _SUF[sz], _r("c", sz)
        self._gen_args(node.args[:3], ["%rdi", "%rsi", "%rdx"])
        self.emit(f"mov{s} (%rsi), {c}")
        self.emit(f"xchg{s} {c}, (%rdi)")
        self.emit(f"mov{s} {c}, (%rdx)")

    def _sync_cas(self, node, want_value: bool) -> None:
        """__sync_(bool|val)_compare_and_swap(p, old, new): old is a value, not a pointer."""
        sz, signed = self._atomic_size(node)
        s, a, d = _SUF[sz], _r("a", sz), _r("d", sz)
        self._gen_args(node.args[:3], ["%rdi", "%rsi", "%rdx"])
        self.emit(f"mov{s} {_r('si', sz)}, {a}")
        self.emit(f"lock cmpxchg{s} {d}, (%rdi)")
        if want_value:
            self._extend_rax(sz, signed)
        else:
            self.emit("sete %al")
            self.emit("movzbl %al, %eax")

    def _sync_release(self, node) -> None:
        sz, _ = self._atomic_size(node)
        self.gen_expr(node.args[0])
        self.emit(f"mov{_SUF[sz]} $0, (%rax)")

    # ---- dispatch targets: builtin_<node.name> ---------------------------
    def builtin_atomic_fetch_add(self, node) -> None: self._rmw(node, "add", True)
    def builtin_atomic_fetch_sub(self, node) -> None: self._rmw(node, "sub", True)
    def builtin_atomic_fetch_and(self, node) -> None: self._rmw(node, "and", True)
    def builtin_atomic_fetch_or(self, node) -> None: self._rmw(node, "or", True)
    def builtin_atomic_fetch_xor(self, node) -> None: self._rmw(node, "xor", True)
    def builtin_atomic_fetch_nand(self, node) -> None: self._rmw(node, "nand", True)
    def builtin_atomic_add_fetch(self, node) -> None: self._rmw(node, "add", False)
    def builtin_atomic_sub_fetch(self, node) -> None: self._rmw(node, "sub", False)
    def builtin_atomic_and_fetch(self, node) -> None: self._rmw(node, "and", False)
    def builtin_atomic_or_fetch(self, node) -> None: self._rmw(node, "or", False)
    def builtin_atomic_xor_fetch(self, node) -> None: self._rmw(node, "xor", False)
    def builtin_atomic_nand_fetch(self, node) -> None: self._rmw(node, "nand", False)
    def builtin_atomic_load_n(self, node) -> None: self._load_n(node)
    def builtin_atomic_store_n(self, node) -> None: self._store_n(node)
    def builtin_atomic_exchange_n(self, node) -> None: self._exchange_n(node)
    def builtin_atomic_compare_exchange_n(self, node) -> None: self._cmpxchg(node, False)
    def builtin_atomic_compare_exchange(self, node) -> None: self._cmpxchg(node, True)
    def builtin_atomic_load(self, node) -> None: self._generic_load(node)
    def builtin_atomic_store(self, node) -> None: self._generic_store(node)
    def builtin_atomic_exchange(self, node) -> None: self._generic_exchange(node)

    def builtin_atomic_test_and_set(self, node) -> None:
        # Set the flag byte with an implicitly locked xchg; result is (old != 0).
        self._gen_args(node.args[:1], ["%rdi"])
        self.emit("mov $1, %eax")
        self.emit("xchgb %al, (%rdi)")
        self.emit("test %al, %al")
        self.emit("setne %al")
        self.emit("movzbl %al, %eax")

    def builtin_atomic_clear(self, node) -> None:
        # Clear the flag byte with a locked xchg (full barrier, like store_n).
        self._gen_args(node.args[:1], ["%rdi"])
        self.emit("xor %eax, %eax")
        self.emit("xchgb %al, (%rdi)")

    def builtin_atomic_is_lock_free(self, node) -> None:
        # Operand size is a compile-time constant; the object pointer is only evaluated.
        size = try_eval_const(node.args[0])
        if size is None:
            diag.error(node, "__atomic_is_lock_free: size must be a constant", "codegen")
        self.gen_expr(node.args[1])
        self.emit(f"mov ${int(size in _SUF)}, %eax")

    def builtin_atomic_thread_fence(self, node) -> None:
        self.emit("mfence")

    def builtin_atomic_signal_fence(self, node) -> None:
        pass

    # __sync_* (legacy gcc intrinsics)
    def builtin_sync_synchronize(self, node) -> None:
        self.emit("mfence")

    def builtin_sync_fetch_and_add(self, node) -> None: self._rmw(node, "add", True)
    def builtin_sync_fetch_and_sub(self, node) -> None: self._rmw(node, "sub", True)
    def builtin_sync_fetch_and_and(self, node) -> None: self._rmw(node, "and", True)
    def builtin_sync_fetch_and_or(self, node) -> None: self._rmw(node, "or", True)
    def builtin_sync_fetch_and_xor(self, node) -> None: self._rmw(node, "xor", True)
    def builtin_sync_fetch_and_nand(self, node) -> None: self._rmw(node, "nand", True)
    def builtin_sync_add_and_fetch(self, node) -> None: self._rmw(node, "add", False)
    def builtin_sync_sub_and_fetch(self, node) -> None: self._rmw(node, "sub", False)
    def builtin_sync_and_and_fetch(self, node) -> None: self._rmw(node, "and", False)
    def builtin_sync_or_and_fetch(self, node) -> None: self._rmw(node, "or", False)
    def builtin_sync_xor_and_fetch(self, node) -> None: self._rmw(node, "xor", False)
    def builtin_sync_nand_and_fetch(self, node) -> None: self._rmw(node, "nand", False)
    def builtin_sync_bool_compare_and_swap(self, node) -> None: self._sync_cas(node, False)
    def builtin_sync_val_compare_and_swap(self, node) -> None: self._sync_cas(node, True)
    def builtin_sync_lock_test_and_set(self, node) -> None: self._exchange_n(node)
    def builtin_sync_lock_release(self, node) -> None: self._sync_release(node)
