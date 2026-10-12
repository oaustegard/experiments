"""Expression parsing from comma down to unary (precedence climbing).

ExpressionMixin provides:
    expr()          comma expression          assign()       assignment
    conditional()   ?: (incl. GNU `a ?: b`)    const_expr()   -> int
    binary(level)   || && | ^ & ==/!= relational shift additive multiplicative
    cast_expr()     `(type) x` and compound literals
    unary()         + - ! ~ * & ++ -- sizeof _Alignof && (label address)
Postfix and primary expressions live in postfix.py / primary.py.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.arith import _addressed_place, compound_assign, inc_dec, new_add, new_binop, new_sub
from hcc.sema.bitfield_width import narrow_wide_bitfield_unary
from hcc.sema.constexpr import eval_const
from hcc.sema.strict_overflow import simplify_mul_div
from hcc.parser.vector_broadcast import promote_vector_operands
from hcc.parser.vector_cast import lower_scalar_to_vector
from hcc.sema.conversions import new_cast
from hcc.tokens import TokenKind
from hcc.typesys import TY_INT, TY_ULONG, TypeKind, is_dyn_struct, is_scalar, pointer_to
from hcc.parser.plan9_ptr import plan9_pointer_conv
from hcc.parser.addr_operand import is_rvalue_operand
from hcc.parser import complex as CPX
from hcc.parser.dyn_struct_ret import dyn_struct_assign

ASSIGN_OPS = {"=": None, "+=": "+", "-=": "-", "*=": "*", "/=": "/", "%=": "%", "&=": "&",
              "|=": "|", "^=": "^", "<<=": "<<", ">>=": ">>"}

BINARY_LEVELS = [("|",), ("^",), ("&",), ("==", "!="), ("<", "<=", ">", ">="),
                 ("<<", ">>"), ("+", "-"), ("*", "/", "%")]


def _comma_result_type(t):
    """Type of `a, b`: a function or array designator decays to a pointer."""
    if t.kind == TypeKind.FUNC:
        return pointer_to(t)
    if t.kind == TypeKind.ARRAY and not t.is_vector:
        return pointer_to(t.base)
    return t


def _side_effect_free(node) -> bool:
    if isinstance(node, (N.VarRef, N.IntLit, N.FloatLit)):
        return True
    if isinstance(node, (N.Member_, N.Deref)):
        return _side_effect_free(node.operand)
    return False


class ExpressionMixin:

    def expr(self):
        node = self.assign()
        while self.at(","):
            loc = self.next().loc
            rhs = self.assign()
            node = self.typed(N.Comma(loc, node, rhs))
            node.ty = _comma_result_type(rhs.ty)   # the operator's own result: decayed
        return node

    def const_expr(self) -> int:
        return eval_const(self.conditional())

    def const_expr_typed(self):
        """Constant expression plus its (C) type: returns (value, Type)."""
        node = self.conditional()
        return eval_const(node), (getattr(node, "ty", None) or TY_INT)

    def assign(self):
        node = self.conditional()
        t = self.tok
        if t.kind == TokenKind.PUNCT and t.text in ASSIGN_OPS:
            self.next()
            rhs = self.assign()
            op = ASSIGN_OPS[t.text]
            self.check_assignable(node, t)
            if op is None:
                if is_dyn_struct(self.typed(node).ty):
                    return dyn_struct_assign(self, node, self.typed(rhs), t.loc)
                rhs = plan9_pointer_conv(self, rhs, self.typed(node).ty)
                if node.ty.is_complex or self.typed(rhs).ty.is_complex:
                    rhs = CPX.convert(self, rhs, node.ty, t.loc)
                return self.typed(N.Assign(t.loc, node, rhs))
            if self.typed(node).ty.is_complex or self.typed(rhs).ty.is_complex:
                return self.complex_compound(op, node, rhs, t.loc)
            node, rhs = promote_vector_operands(self, t.loc, op, node, rhs)
            return compound_assign(op, node, rhs, t.loc, self.new_temp)
        return node

    def check_assignable(self, node, tok) -> None:
        if isinstance(node, (N.VarRef, N.Deref, N.Member_)):
            return
        if isinstance(node, N.Comma) and node.lvalue:     # compound literal
            return
        if isinstance(node, N.Cast) and isinstance(node.operand, (N.VarRef, N.Deref, N.Member_)) \
                and node.ty.size == node.operand.ty.size:
            return      # GNU lvalue casts in old code; tolerated
        self.error("lvalue required as left operand of assignment", tok)

    def conditional(self):
        cond = self.logor()
        if not self.at("?"):
            return cond
        cond = CPX.truth(self, cond)
        loc = self.next().loc
        if self.at(":"):                  # GNU a ?: b
            self.next()
            tmp = self.new_temp(_comma_result_type(cond.ty))
            lhs = self.typed(N.Assign(loc, self.typed(N.VarRef(loc, tmp)), cond))
            rhs = self.typed(N.Cond(loc, CPX.truth(self, self.typed(N.VarRef(loc, tmp))),
                                    self.typed(N.VarRef(loc, tmp)), self.conditional()))
            return self.typed(N.Comma(loc, lhs, rhs))
        then = self.expr()
        self.expect(":")
        els = self.conditional()
        return self.typed(N.Cond(loc, cond, then, els))

    def logor(self):
        node = self.logand()
        while self.at("||"):
            loc = self.next().loc
            node = self.typed(N.Logical(loc, "||", CPX.truth(self, node), CPX.truth(self, self.logand())))
        return node

    def logand(self):
        node = self.binary(0)
        while self.at("&&"):
            loc = self.next().loc
            node = self.typed(N.Logical(loc, "&&", CPX.truth(self, node), CPX.truth(self, self.binary(0))))
        return node

    def binary(self, level: int):
        if level == len(BINARY_LEVELS):
            return self.cast_expr()
        node = self.binary(level + 1)
        ops = BINARY_LEVELS[level]
        while True:
            t = self.tok
            if t.kind != TokenKind.PUNCT or t.text not in ops:
                return node
            self.next()
            rhs = self.binary(level + 1)
            if self.typed(node).ty.is_complex or self.typed(rhs).ty.is_complex:
                node = CPX.binary(self, t.text, node, rhs, t.loc)
                continue
            node, rhs = promote_vector_operands(self, t.loc, t.text, node, rhs)
            node = new_binop(t.text, node, rhs, t.loc)
            if t.text == "/" and not getattr(self.opts, "wrapv", False):
                node = simplify_mul_div(node, t.loc)  # signed overflow is UB (-fno-wrapv)
            if isinstance(node, N.Binary) and node.ty.is_vector:
                node.result_buf = self.new_temp(node.ty, t.loc)

    def cast_expr(self):
        while self.at("__extension__"):     # `__extension__ (T)x`: the marker precedes the cast
            self.next()
        if self.at("(") and self.is_typename(self.peek()):
            start = self.pos
            allow = self.compound_lit_follows()
            loc = self.next().loc
            outer_vm = self.vm_typeof
            self.vm_typeof = []
            ty = self.typename(allow_storage=allow)
            effects = self.vm_typeof              # `(typeof(vm-expr))x` evaluates vm-expr
            self.vm_typeof = outer_vm
            self.expect(")")
            if self.at("{"):                      # compound literal
                self.pos = start
                return self.unary()
            _check_cast_target(self, loc, ty)
            ty, pre = self.vla_bind(ty, loc)      # `(int (*)[++i])p` evaluates ++i
            pre = [N.ExprStmt(loc, e) for e in effects] + pre
            operand = self.cast_expr()
            if ty.kind == TypeKind.VOID:
                return self.vla_prefix(pre, N.Cast(loc, operand, ty=ty), loc)
            if ty.is_complex or self.typed(operand).ty.is_complex:
                return self.vla_prefix(pre, CPX.convert(self, operand, ty.unqual(), loc), loc)
            lowered = lower_scalar_to_vector(self, loc, self.typed(operand), ty.unqual())
            if lowered is not None:
                return self.vla_prefix(pre, lowered, loc)
            src_kind = self.typed(operand).ty.kind
            if src_kind in (TypeKind.STRUCT, TypeKind.UNION) and is_scalar(ty):
                self.error(f"used {src_kind.value} type value where scalar is required", loc)
            node = N.Cast(loc, operand, ty=ty.unqual())
            self.typed(operand)
            return self.vla_prefix(pre, node, loc)
        return self.unary()

    def unary(self):
        while self.at("__extension__"):     # GNU: `__extension__ expr` is a no-op prefix
            self.next()
        t = self.tok
        s = t.text
        if t.kind == TokenKind.PUNCT:
            if s == "+":
                self.next()
                return self.cast_expr()
            if s == "-":
                self.next()
                operand = self.cast_expr()
                if self.typed(operand).ty.is_complex:
                    return CPX.negate(self, operand, t.loc)
                buf = self.new_temp(operand.ty, t.loc) if operand.ty.is_vector else None
                return narrow_wide_bitfield_unary(self.typed(N.Unary(t.loc, "-", operand, result_buf=buf)))
            if s == "!":
                self.next()
                return self.typed(N.Unary(t.loc, "!", CPX.truth(self, self.typed(self.cast_expr()))))
            if s == "~":
                self.next()
                operand = self.cast_expr()
                if self.typed(operand).ty.is_complex:
                    return CPX.conjugate(self, operand, t.loc)
                buf = self.new_temp(operand.ty, t.loc) if operand.ty.is_vector else None
                return narrow_wide_bitfield_unary(self.typed(N.Unary(t.loc, "~", operand, result_buf=buf)))
            if s == "&":
                self.next()
                operand = self.cast_expr()
                if is_rvalue_operand(operand):
                    self.error("lvalue required as unary '&' operand", t)
                if isinstance(operand, N.Member_) and operand.member.is_bitfield:
                    self.error("cannot take address of bit-field", t)
                if operand.ty is not None and operand.ty.is_rev_order:
                    self.error("cannot take address of scalar with reverse storage order", t)
                if isinstance(operand, N.VarRef) and getattr(operand.var, "is_register", False):
                    self.error(f"address of register variable '{operand.var.name}' requested", t)
                return self.typed(N.AddrOf(t.loc, operand))
            if s == "*":
                self.next()
                operand = self.cast_expr()
                self.typed(operand)
                if operand.ty.kind == TypeKind.FUNC:
                    return operand
                return self.typed(N.Deref(t.loc, operand))
            if s in ("++", "--"):
                self.next()
                operand = self.unary()
                return inc_dec(operand, 1 if s == "++" else -1, False, t.loc, self.new_temp)
            if s == "&&":                        # GNU label address
                self.next()
                name = self.expect_ident()
                fname = self.fn.name if self.fn is not None else ""
                return self.typed(N.LabelAddr(t.loc, name.text, fname))
        elif t.kind == TokenKind.KEYWORD:
            if s == "sizeof":
                return self.sizeof_expr()
            if s == "_Alignof":
                self.next()
                ty = None
                if self.at("(") and self.is_typename(self.peek()):
                    start = self.pos
                    allow = self.compound_lit_follows()
                    self.next()
                    ty = self.scoped_typename(allow_storage=allow)
                    self.expect(")")
                    if self.at("{"):            # `_Alignof (T){...}`: compound literal operand
                        self.pos = start
                        ty = None
                if ty is None:
                    node = self.unary()
                    ty = node.ty
                    if isinstance(node, N.VarRef) and isinstance(node.var, N.Function) and node.var.align:
                        return N.IntLit(t.loc, node.var.align, ty=TY_ULONG)
                    if isinstance(node, N.Member_) and node.member.align:
                        return N.IntLit(t.loc, node.member.align, ty=TY_ULONG)   # packed/aligned member
                return N.IntLit(t.loc, ty.align, ty=TY_ULONG)
            if s == "__extension__":
                self.next()
                return self.cast_expr()
            if s in ("__real__", "__imag__"):
                self.next()
                operand = self.cast_expr()
                return CPX.part(self, self.typed(operand), 0 if s == "__real__" else 1, t.loc)
        return self.postfix()

    def complex_compound(self, op, node, rhs, loc):
        """`a op= b` on a complex operand: `a = a op b`. When `a` has side effects its
        address is evaluated once into a temporary (as for scalar compound assignment)."""
        if _side_effect_free(node):
            pre, place = [], (lambda: node)
        else:
            pre, place = _addressed_place(node, loc, self.new_temp)
        value = CPX.binary(self, op, place(), rhs, loc)
        result = self.typed(N.Assign(loc, place(), CPX.convert(self, value, node.ty, loc)))
        for e in reversed(pre):
            result = self.typed(N.Comma(loc, e, result))
        return result

    def sizeof_expr(self):
        t = self.next()
        node = None
        effects = []
        if self.at("(") and self.is_typename(self.peek()):
            start = self.pos
            allow = self.compound_lit_follows()
            self.next()
            outer_vm = self.vm_typeof
            self.vm_typeof = []
            ty = self.scoped_typename(allow_storage=allow)
            effects = self.vm_typeof        # `sizeof(typeof(vm-expr))` evaluates vm-expr
            self.vm_typeof = outer_vm
            self.expect(")")
            if self.at("{"):
                self.pos = start
                effects = []
                node = self.unary()
                ty = node.ty
        else:
            node = self.unary()
            self.typed(node)
            ty = node.ty
            if ty.kind == TypeKind.VLA:
                # sizeof(expr) of a variably modified type still evaluates
                # expr (its VLA lengths may have side effects).
                return self.typed(N.Comma(t.loc, node, self.vla_size_expr(ty, t.loc)))
        if ty.kind == TypeKind.VLA or is_dyn_struct(ty):
            size = self.vla_size_expr(ty, t.loc)
            if effects:
                size = self.vla_prefix([N.ExprStmt(e.loc, e) for e in effects], size, t.loc)
            if node is not None:
                # `sizeof expr` on a VLA still evaluates expr (its side effects).
                size = self.typed(N.Comma(t.loc, node, size))
            return size
        if not ty.is_complete and ty.kind != TypeKind.VOID and ty.kind != TypeKind.FUNC:
            self.error(f"invalid application of 'sizeof' to incomplete type '{ty}'", t)
        return N.IntLit(t.loc, ty.size, ty=TY_ULONG)


__all__ = ["ExpressionMixin", "new_add", "new_sub", "new_cast", "inc_dec", "TY_INT"]


def _check_cast_target(parser, loc, ty) -> None:
    """C99 6.5.4: a cast's type name must not be an array or function type."""
    if ty.kind in (TypeKind.ARRAY, TypeKind.VLA) and not ty.is_vector:   # GCC vectors are casts
        parser.error("cast specifies array type", loc)
    if ty.kind == TypeKind.FUNC:
        parser.error("cast specifies function type", loc)
