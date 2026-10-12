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
from hcc.sema.arith import compound_assign, inc_dec, new_add, new_binop, new_sub
from hcc.sema.constexpr import eval_const
from hcc.parser.vector_broadcast import promote_vector_operands
from hcc.parser.vector_cast import lower_scalar_to_vector
from hcc.sema.conversions import new_cast
from hcc.tokens import TokenKind
from hcc.typesys import TY_INT, TY_ULONG, TypeKind, pointer_to

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

    def assign(self):
        node = self.conditional()
        t = self.tok
        if t.kind == TokenKind.PUNCT and t.text in ASSIGN_OPS:
            self.next()
            rhs = self.assign()
            op = ASSIGN_OPS[t.text]
            self.check_assignable(node, t)
            if op is None:
                return self.typed(N.Assign(t.loc, node, rhs))
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
        loc = self.next().loc
        if self.at(":"):                  # GNU a ?: b
            self.next()
            tmp = self.new_temp(cond.ty)
            lhs = self.typed(N.Assign(loc, self.typed(N.VarRef(loc, tmp)), cond))
            rhs = self.typed(N.Cond(loc, self.typed(N.VarRef(loc, tmp)),
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
            node = self.typed(N.Logical(loc, "||", node, self.logand()))
        return node

    def logand(self):
        node = self.binary(0)
        while self.at("&&"):
            loc = self.next().loc
            node = self.typed(N.Logical(loc, "&&", node, self.binary(0)))
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
            node, rhs = promote_vector_operands(self, t.loc, t.text, node, rhs)
            node = new_binop(t.text, node, rhs, t.loc)
            if isinstance(node, N.Binary) and node.ty.is_vector:
                node.result_buf = self.new_temp(node.ty, t.loc)

    def cast_expr(self):
        if self.at("(") and self.is_typename(self.peek()):
            start = self.pos
            loc = self.next().loc
            ty = self.typename()
            self.expect(")")
            if self.at("{"):                      # compound literal
                self.pos = start
                return self.unary()
            ty, pre = self.vla_bind(ty, loc)      # `(int (*)[++i])p` evaluates ++i
            operand = self.cast_expr()
            if ty.kind == TypeKind.VOID:
                return self.vla_prefix(pre, N.Cast(loc, operand, ty=ty), loc)
            lowered = lower_scalar_to_vector(self, loc, self.typed(operand), ty.unqual())
            if lowered is not None:
                return self.vla_prefix(pre, lowered, loc)
            node = N.Cast(loc, operand, ty=ty.unqual())
            self.typed(operand)
            return self.vla_prefix(pre, node, loc)
        return self.unary()

    def unary(self):
        t = self.tok
        s = t.text
        if t.kind == TokenKind.PUNCT:
            if s == "+":
                self.next()
                return self.cast_expr()
            if s == "-":
                self.next()
                operand = self.cast_expr()
                buf = self.new_temp(operand.ty, t.loc) if operand.ty.is_vector else None
                return self.typed(N.Unary(t.loc, "-", operand, result_buf=buf))
            if s == "!":
                self.next()
                return self.typed(N.Unary(t.loc, "!", self.cast_expr()))
            if s == "~":
                self.next()
                operand = self.cast_expr()
                buf = self.new_temp(operand.ty, t.loc) if operand.ty.is_vector else None
                return self.typed(N.Unary(t.loc, "~", operand, result_buf=buf))
            if s == "&":
                self.next()
                operand = self.cast_expr()
                if isinstance(operand, N.Member_) and operand.member.is_bitfield:
                    self.error("cannot take address of bit-field", t)
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
                if self.at("(") and self.is_typename(self.peek()):
                    self.next()
                    ty = self.typename()
                    self.expect(")")
                else:
                    node = self.unary()
                    ty = node.ty
                    if isinstance(node, N.VarRef) and isinstance(node.var, N.Function) and node.var.align:
                        return N.IntLit(t.loc, node.var.align, ty=TY_ULONG)
                return N.IntLit(t.loc, ty.align, ty=TY_ULONG)
            if s == "__extension__":
                self.next()
                return self.cast_expr()
            if s in ("__real__", "__imag__"):
                self.next()
                return self.cast_expr()       # TODO: _Complex
        return self.postfix()

    def sizeof_expr(self):
        t = self.next()
        if self.at("(") and self.is_typename(self.peek()):
            start = self.pos
            self.next()
            ty = self.typename()
            self.expect(")")
            if self.at("{"):
                self.pos = start
                ty = self.unary().ty
        else:
            node = self.unary()
            self.typed(node)
            ty = node.ty
        if ty.kind == TypeKind.VLA:
            return self.vla_size_expr(ty, t.loc)
        if not ty.is_complete and ty.kind != TypeKind.VOID and ty.kind != TypeKind.FUNC:
            self.error(f"invalid application of 'sizeof' to incomplete type '{ty}'", t)
        return N.IntLit(t.loc, ty.size, ty=TY_ULONG)


__all__ = ["ExpressionMixin", "new_add", "new_sub", "new_cast", "inc_dec", "TY_INT"]
