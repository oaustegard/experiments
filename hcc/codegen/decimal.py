"""_Decimal32/_Decimal64/_Decimal128 (TR 24732) code generation.

A decimal value lives in %xmm0 as its BID bit pattern (low 4/8 bytes for
decimal32/64, all 16 bytes for decimal128), exactly like float/double, so
loads, stores and the call ABI reuse the SSE paths. Arithmetic, comparisons
and conversions call libgcc's `__bid_*` routines (the gcc driver links libgcc).

    decimal_binary(node)     + - * / and == != < <= > >= on decimal operands
    decimal_unary(node)      unary minus (sign-bit flip) and !
    decimal_cast(frm, to)    decimal <-> integer and decimal <-> decimal
    decimal_cmp_zero(ty)     flags for "value != 0" (like cmp_zero)
"""
from __future__ import annotations

from hcc import diag
from hcc.typesys import TY_INT, TypeKind, is_decimal, is_integer

DEC_LOAD = {TypeKind.DEC32: "movss", TypeKind.DEC64: "movsd", TypeKind.DEC128: "movups"}
DEC_STORE = DEC_LOAD

_SFX = {TypeKind.DEC32: "sd", TypeKind.DEC64: "dd", TypeKind.DEC128: "td"}
_ARITH = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
_CMP = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
# libgcc's __bid_<rel>xx2 return <0 / 0 / >0 like the soft-float comparisons,
# so the relation is tested against zero on the returned int.
# binary float <-> decimal conversions (libgcc names are not regular, so a table)
_BIN_TO_DEC = {(TypeKind.FLOAT, TypeKind.DEC32): "extendsfsd", (TypeKind.FLOAT, TypeKind.DEC64): "extendsfdd",
               (TypeKind.FLOAT, TypeKind.DEC128): "extendsftd", (TypeKind.DOUBLE, TypeKind.DEC32): "truncdfsd",
               (TypeKind.DOUBLE, TypeKind.DEC64): "extenddfdd", (TypeKind.DOUBLE, TypeKind.DEC128): "extenddftd"}
_DEC_TO_BIN = {(TypeKind.DEC32, TypeKind.FLOAT): "truncsdsf", (TypeKind.DEC64, TypeKind.FLOAT): "truncddsf",
               (TypeKind.DEC128, TypeKind.FLOAT): "trunctdsf", (TypeKind.DEC32, TypeKind.DOUBLE): "extendsddf",
               (TypeKind.DEC64, TypeKind.DOUBLE): "truncdddf", (TypeKind.DEC128, TypeKind.DOUBLE): "trunctddf"}
_CMP_SETCC = {"==": "sete", "!=": "setne", "<": "setl", "<=": "setle", ">": "setg", ">=": "setge"}


def _dec_size_sfx(ty) -> str:
    return _SFX[ty.kind]


class DecimalMixin:

    def _dec_push(self, ty) -> None:
        """Push the xmm0 bit pattern of a decimal value (16 bytes for decimal128)."""
        if ty.kind == TypeKind.DEC128:
            self.emit("sub $16, %rsp")
            self.emit("movups %xmm0, (%rsp)")
            self.depth += 2
        else:
            self.pushf()

    def _dec_pop(self, ty, n: int) -> None:
        if ty.kind == TypeKind.DEC128:
            self.emit(f"movups (%rsp), %xmm{n}")
            self.emit("add $16, %rsp")
            self.depth -= 2
        else:
            self.popf(n)

    def _dec_call(self, name: str) -> None:
        """Call a libgcc routine, keeping %rsp 16-byte aligned at the call."""
        pad = self.depth % 2 == 1
        if pad:
            self.emit("sub $8, %rsp")
            self.depth += 1
        self.emit(f"call {name}@PLT")
        if pad:
            self.emit("add $8, %rsp")
            self.depth -= 1

    def decimal_binary(self, node) -> None:
        ty = node.lhs.ty
        if node.rhs.ty.kind != ty.kind:
            diag.error(node, f"unsupported mixed decimal operand types '{ty}' and '{node.rhs.ty}'",
                       "codegen")
        self.gen_expr(node.rhs)
        self._dec_push(ty)
        self.gen_expr(node.lhs)
        self._dec_pop(ty, 1)                 # %xmm0 = lhs, %xmm1 = rhs
        self.decimal_lane_op(node.op, ty, node)

    def decimal_lane_op(self, op, ty, node=None) -> None:
        """%xmm0 = %xmm0 op %xmm1 (arithmetic) or the relation's 0/1 in %eax-based %al."""
        sfx = _dec_size_sfx(ty)
        if op in _ARITH:
            self._dec_call(f"__bid_{_ARITH[op]}{sfx}3")
            return
        if op not in _CMP:
            diag.error(node, f"unsupported decimal operator '{op}'", "codegen")
        self._dec_call(f"__bid_{_CMP[op]}{sfx}2")
        self.emit("cmp $0, %eax")
        self.emit(f"{_CMP_SETCC[op]} %al")
        self.emit("movzbl %al, %eax")

    def decimal_unary(self, node) -> None:
        ty = node.operand.ty
        self.gen_expr(node.operand)
        if node.op == "-":
            # flip the sign bit: the top bit of the last byte of the BID encoding
            self.emit("sub $16, %rsp")
            self.emit("movups %xmm0, (%rsp)")
            self.emit(f"xorb $0x80, {ty.size - 1}(%rsp)")
            self.emit("movups (%rsp), %xmm0")
            self.emit("add $16, %rsp")
            return
        if node.op == "!":
            self.decimal_cmp_zero(ty)
            self.emit("sete %al")
            self.emit("movzbl %al, %eax")
            return
        diag.error(node, f"unsupported decimal unary operator '{node.op}'", "codegen")

    def decimal_cmp_zero(self, ty) -> None:
        """Flags as for an integer `cmp $0`: ZF set iff the value is zero."""
        self.emit("xorps %xmm1, %xmm1")
        self._dec_call(f"__bid_ne{_dec_size_sfx(ty)}2")
        self.emit("test %eax, %eax")

    def decimal_cast(self, frm, to) -> None:
        """Value in %xmm0 (decimal) or %rax (integer) converted from frm to to."""
        if is_decimal(frm) and is_decimal(to):
            if frm.kind == to.kind:
                return
            a, b = _SFX[frm.kind], _SFX[to.kind]
            wide = frm.size < to.size
            self._dec_call(f"__bid_{'extend' if wide else 'trunc'}{a}{b}2")
            return
        if (frm.kind, to.kind) in _BIN_TO_DEC:
            return self._dec_call(f"__bid_{_BIN_TO_DEC[(frm.kind, to.kind)]}")
        if (frm.kind, to.kind) in _DEC_TO_BIN:
            return self._dec_call(f"__bid_{_DEC_TO_BIN[(frm.kind, to.kind)]}")
        if is_decimal(to):
            return self._int_to_decimal(frm, to)
        if to.kind == TypeKind.BOOL:
            self.decimal_cmp_zero(frm)
            self.emit("setne %al")
            self.emit("movzbl %al, %eax")
            return
        if is_integer(to) and to.size <= 8 and to.kind != TypeKind.INT128:
            return self._decimal_to_int(frm, to)
        diag.error(None, f"unsupported conversion from '{frm}' to '{to}'", "codegen")

    def _int_to_decimal(self, frm, to) -> None:
        if not is_integer(frm) or frm.kind == TypeKind.INT128:
            diag.error(None, f"unsupported conversion from '{frm}' to '{to}'", "codegen")
        if frm.size < 4:
            self.cast(frm, TY_INT)           # promote small integers first
            frm = TY_INT
        uns = "uns" if frm.is_unsigned else ""
        if frm.size == 8:
            self.emit("mov %rax, %rdi")
            width = "di"
        else:
            self.emit("mov %eax, %edi")
            width = "si"
        self._dec_call(f"__bid_float{uns}{width}{_SFX[to.kind]}")

    def _decimal_to_int(self, frm, to) -> None:
        # libgcc converts to 32- or 64-bit integers; narrower targets go through int
        wide = to.size == 8
        uns = "uns" if to.is_unsigned and to.size >= 4 else ""
        self._dec_call(f"__bid_fix{uns}{_SFX[frm.kind]}{'di' if wide else 'si'}")
        if to.size < 4:
            self.cast(TY_INT, to)
