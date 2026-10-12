"""Code generation: nodes.TranslationUnit -> AT&T assembly text.

    generate(tu, opts=None) -> str

CodeGen is assembled from one mixin per concern (CONTRACT section 2/7).
Stack-machine design: expressions leave their value in %rax / %xmm0 / st(0).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.codegen.arith import ArithMixin
from hcc.codegen.atomics import AtomicMixin
from hcc.codegen.asm import AsmMixin
from hcc.codegen.builtins import BuiltinMixin
from hcc.codegen.calls import CallMixin
from hcc.codegen.casts import CastMixin
from hcc.codegen.context import CodeGenBase
from hcc.codegen.data import DataMixin
from hcc.codegen.decimal import DecimalMixin
from hcc.codegen.decimal_vector import DecimalVectorMixin
from hcc.codegen.expressions import ExpressionMixin
from hcc.codegen.floats import FloatMixin
from hcc.codegen.fpclassify import FpClassifyMixin
from hcc.codegen.functions import FunctionMixin
from hcc.codegen.initfini import InitFiniMixin
from hcc.codegen.int128 import Int128Mixin
from hcc.codegen.nested import NestedMixin
from hcc.codegen.memory import MemoryMixin
from hcc.codegen.overflow import OverflowMixin
from hcc.codegen.overflow128 import Overflow128Mixin
from hcc.codegen.statements import StatementMixin
from hcc.codegen.varargs import VarargsMixin
from hcc.codegen.vector_ops import VectorOpsMixin
from hcc.codegen.vector_wide import VectorWideMixin


class CodeGen(FunctionMixin, NestedMixin, StatementMixin, ExpressionMixin, ArithMixin, AtomicMixin, FloatMixin,
              CastMixin, CallMixin, VarargsMixin, BuiltinMixin, FpClassifyMixin, OverflowMixin, Overflow128Mixin, AsmMixin, MemoryMixin,
              DataMixin, VectorOpsMixin, VectorWideMixin, InitFiniMixin, Int128Mixin, DecimalMixin, DecimalVectorMixin, CodeGenBase):

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
                elif obj.ifunc_of:
                    self.emit_ifunc(obj)
                elif obj.is_weak:
                    # Weak declaration without a definition here: the symbol
                    # is an undefined weak reference (resolves to 0 if absent).
                    self.directive(f".weak {obj.symbol}")
            elif isinstance(obj, N.Var):
                if obj.is_definition and not obj.is_extern:
                    self.emit_data(obj)
                elif obj.alias_of and not obj.is_definition:
                    self.emit_alias(obj)
                elif obj.is_weak and not obj.is_definition:
                    self.directive(f".weak {obj.symbol}")
        self.emit_init_fini()
        self.directive('.section .note.GNU-stack,"",@progbits')
        return "\n".join(self.out) + "\n"

    def emit_alias(self, fn) -> None:
        if not fn.is_static:
            self.directive(f".globl {fn.symbol}")
        if fn.is_weak:
            self.directive(f".weak {fn.symbol}")
        self.directive(f".set {fn.symbol}, {fn.alias_of}")

    def emit_ifunc(self, fn) -> None:
        """GNU indirect function: the symbol's value is the resolver's address."""
        if not fn.is_static:
            self.directive(f".globl {fn.symbol}")
        if fn.visibility in ("hidden", "internal", "protected"):
            self.directive(f".{fn.visibility} {fn.symbol}")
        self.directive(f".type {fn.symbol}, @gnu_indirect_function")
        self.directive(f".set {fn.symbol}, {fn.ifunc_of}")


def generate(tu, opts=None) -> str:
    return CodeGen(tu, opts).run()
