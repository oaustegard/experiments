"""__builtin_setjmp / __builtin_longjmp (gcc's 5-word buffer: rbp, resume address, rsp)."""
from __future__ import annotations

from hcc import typesys as T
from hcc.builtins import builtin_codegen, builtin_parser
from hcc.sema.conversions import new_cast


@builtin_parser("__builtin_setjmp")
def _parse_setjmp(self, tok):
    """__builtin_setjmp(buf): buf is a void *[5]-style record; returns 0."""
    args = [new_cast(a, T.pointer_to(T.TY_VOID)) for a in self._args(1)]
    return self._call(tok, "setjmp", args, T.TY_INT)


@builtin_parser("__builtin_longjmp")
def _parse_longjmp(self, tok):
    """__builtin_longjmp(buf, 1): resumes the matching __builtin_setjmp."""
    args = self._args(2)
    args = [new_cast(args[0], T.pointer_to(T.TY_VOID)), new_cast(args[1], T.TY_INT)]
    return self._call(tok, "longjmp", args, T.TY_VOID)


@builtin_codegen("setjmp")
def _gen_setjmp(self, node) -> None:
    """Save {rbp, resume address, rsp} into buf[0], buf[1], buf[2] (gcc
    layout) and return 0; a __builtin_longjmp to buf lands on `resume`
    with %eax = the longjmp value (0 is replaced by 1, as in C)."""
    self.gen_expr(node.args[0])
    resume, end = self.new_label(), self.new_label()
    self.emit("mov %rbp, (%rax)")
    self.emit(f"lea {resume}(%rip), %rcx")
    self.emit("mov %rcx, 8(%rax)")
    self.emit("mov %rsp, 16(%rax)")
    self.emit("xor %eax, %eax")
    self.emit(f"jmp {end}")
    self.label(resume)
    nonzero = self.new_label()
    self.emit("test %eax, %eax")
    self.emit(f"jnz {nonzero}")
    self.emit("mov $1, %eax")
    self.label(nonzero)
    self.label(end)


@builtin_codegen("longjmp")
def _gen_longjmp(self, node) -> None:
    # The value is pushed while buf is evaluated, then read from the top
    # of the stack before rsp is restored to the setjmp-time value.
    self.gen_expr(node.args[1])
    self.push()
    self.gen_expr(node.args[0])
    self.emit("mov (%rsp), %rdx")        # longjmp value (int)
    self.emit("mov 8(%rax), %rcx")
    self.emit("mov 16(%rax), %rsp")
    self.emit("mov (%rax), %rbp")
    self.emit("mov %rdx, %rax")
    self.emit("jmp *%rcx")
    self.depth -= 1                      # never falls through; no pop emitted
