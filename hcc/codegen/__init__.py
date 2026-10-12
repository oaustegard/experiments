"""Code generation: nodes.TranslationUnit -> AT&T assembly text.

    generate(tu, opts=None) -> str

CodeGen is assembled from one mixin per concern (CONTRACT section 2/7).
Stack-machine design: expressions leave their value in %rax / %xmm0 / st(0).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.codegen.arith import ArithMixin
from hcc.codegen.asm import AsmMixin
from hcc.codegen.builtins import BuiltinMixin
from hcc.codegen.calls import CallMixin
from hcc.codegen.casts import CastMixin
from hcc.codegen.context import CodeGenBase
from hcc.codegen.data import DataMixin
from hcc.codegen.expressions import ExpressionMixin
from hcc.codegen.floats import FloatMixin
from hcc.codegen.functions import FunctionMixin
from hcc.codegen.memory import MemoryMixin
from hcc.codegen.overflow import OverflowMixin
from hcc.codegen.statements import StatementMixin
from hcc.codegen.varargs import VarargsMixin


class CodeGen(FunctionMixin, StatementMixin, ExpressionMixin, ArithMixin, FloatMixin,
              CastMixin, CallMixin, VarargsMixin, BuiltinMixin, OverflowMixin, AsmMixin, MemoryMixin,
              DataMixin, CodeGenBase):

    def run(self) -> str:
        self.directive(f'.file "{getattr(self.opts, "source_name", "") or "hcc"}"')
        for obj in self.tu.objs:
            if isinstance(obj, N.Asm):
                self.emit_toplevel_asm(obj)
            elif isinstance(obj, N.Function):
                if obj.is_definition:
                    self.emit_function(obj)
                elif obj.alias_of:
                    self.emit_alias(obj)
            elif isinstance(obj, N.Var):
                if obj.is_definition and not obj.is_extern:
                    self.emit_data(obj)
        self.directive('.section .note.GNU-stack,"",@progbits')
        return "\n".join(self.out) + "\n"

    def emit_alias(self, fn) -> None:
        if not fn.is_static:
            self.directive(f".globl {fn.symbol}")
        if fn.is_weak:
            self.directive(f".weak {fn.symbol}")
        self.directive(f".set {fn.symbol}, {fn.alias_of}")


def generate(tu, opts=None) -> str:
    return CodeGen(tu, opts).run()
