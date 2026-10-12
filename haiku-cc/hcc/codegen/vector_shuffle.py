"""__builtin_shuffle lowering (CONTRACT section 5, vector values).

A vector value is represented by its address. The source vector(s) and the
integer mask are evaluated to addresses; each result element is copied from
the selected source element (index taken modulo N for the 2-operand form,
modulo 2N for the 3-operand form, where indices >= N select the second
vector). The result goes to the node's `result_buf`; %rax = its address.
"""
from __future__ import annotations

_STORE_ELEM = {1: ("movb (%rcx), %al", "movb %al, {off}(%rdi)"),
               2: ("movw (%rcx), %ax", "movw %ax, {off}(%rdi)"),
               4: ("movl (%rcx), %eax", "movl %eax, {off}(%rdi)"),
               8: ("movq (%rcx), %rax", "movq %rax, {off}(%rdi)")}


def shuffle(gen, node) -> None:
    two = len(node.args) == 3
    a = node.args[0]
    mask = node.args[-1]
    es = a.ty.base.size
    n = a.ty.size // es
    ms = mask.ty.base.size

    gen.gen_expr(a)
    gen.push()
    if two:
        gen.gen_expr(node.args[1])
        gen.push()
    gen.gen_expr(mask)
    gen.emit("mov %rax, %r8")                 # r8 = mask address
    if two:
        gen.pop("%r9")                        # r9 = second source
        gen.pop("%r10")                       # r10 = first source
    else:
        gen.pop("%r10")
        gen.emit("mov %r10, %r9")
    gen.frame_lea(node.result_buf.offset, "%rdi")
    index_mask = 2 * n - 1 if two else n - 1
    load, store = _STORE_ELEM[es]
    for i in range(n):
        gen._vec_load(ms, False, i * ms, "%r8", "%rcx", "%ecx")
        gen.emit(f"and ${index_mask}, %rcx")
        if two:
            # Branch-free select: idx >= n picks the second vector, idx -= n.
            gen.emit(f"cmp ${n}, %rcx")       # sets CF = (idx < n)
            gen.emit(f"lea {-n}(%rcx), %rdx")  # lea leaves the flags alone
            gen.emit("mov %r10, %rsi")
            gen.emit("cmovae %r9, %rsi")
            gen.emit("cmovae %rdx, %rcx")
        else:
            gen.emit("mov %r10, %rsi")
        gen.emit(f"imul ${es}, %rcx, %rcx")
        gen.emit("add %rsi, %rcx")            # rcx = address of selected element
        gen.emit(load)
        gen.emit(store.format(off=i * es))
    gen.emit("mov %rdi, %rax")
