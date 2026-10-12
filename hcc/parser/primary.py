"""Primary expressions.

PrimaryMixin provides primary(): parenthesized exprs, GNU statement
expressions `({...})`, identifiers (variables, enum constants, implicit
function declarations), number/char/string literals, `_Generic`,
__func__/__FUNCTION__, and dispatch of __builtin_* to builtins.py.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.parser.builtins import SPECIAL_BUILTINS, floatn_libc_name
from hcc.parser.libc_builtins import POWI_HELPERS, libc_return_type
from hcc.parser.scope import VarScope
from hcc.sema.compat import is_compatible
from hcc.tokens import TokenKind
from hcc import typesys as T


class PrimaryMixin:

    def primary(self):
        t = self.tok
        k = t.kind
        if k == TokenKind.PUNCT and t.text == "(":
            if self.peek().text == "{" and self.peek().kind == TokenKind.PUNCT:
                self.next()
                self.next()
                outer_base = self.stmt_expr_base
                if outer_base is None:
                    self.stmt_expr_base = len(self.scope_stack)
                body = self.compound_stmt_body(t.loc)
                self.stmt_expr_base = outer_base
                self.expect(")")
                return self.typed(N.StmtExpr(t.loc, body))
            self.next()
            node = self.expr()
            self.expect(")")
            return node
        if k == TokenKind.NUMBER:
            self.next()
            if T.is_flonum(t.ty):
                return N.FloatLit(t.loc, t.value, ty=t.ty, bits=getattr(t.value, "x87", None))
            return N.IntLit(t.loc, t.value, ty=t.ty)
        if k == TokenKind.CHAR:
            self.next()
            return N.IntLit(t.loc, t.value, ty=t.ty)
        if k == TokenKind.STRING:
            self.next()
            return self.typed(N.VarRef(t.loc, self.string_var(t)))
        if t.text in SPECIAL_BUILTINS and k in (TokenKind.IDENT, TokenKind.KEYWORD):
            return self.parse_builtin(self.next())
        if k == TokenKind.KEYWORD and t.text == "_Generic":
            return self.generic_selection()
        if k == TokenKind.IDENT:
            return self.identifier()
        diag.error(t, f"expected expression before '{t.text or 'end of file'}'")

    def identifier(self):
        t = self.next()
        name = t.text
        vs = self.find_var(name)
        if vs is not None:
            if vs.var is not None:
                return self.typed(N.VarRef(t.loc, vs.var))
            if vs.enum_ty is not None:
                v = vs.enum_val
                ty = vs.enum_cty or (T.TY_INT if -(1 << 31) <= v < (1 << 31) else T.TY_LONG)
                return N.IntLit(t.loc, v, ty=ty)
            self.error(f"unexpected type name '{name}': expected expression", t)
        if name in ("__func__", "__FUNCTION__", "__PRETTY_FUNCTION__") and self.fn is not None:
            return self.typed(N.VarRef(t.loc, self.func_name_var()))
        if name in ("__func__", "__FUNCTION__", "__PRETTY_FUNCTION__") and self.fn is None:
            # Outside a function GCC yields an empty string (warning for __func__ only).
            if name == "__func__":
                diag.warn(t, "'__func__' is not defined outside of function scope")
            var = self.new_anon_gvar(T.array_of(T.TY_CHAR, 1), t.loc)
            var.init_data = N.GlobalData(bytearray(b"\0"))
            return self.typed(N.VarRef(t.loc, var))
        if name == "nullptr" and self.c23_mode():
            # C23 nullptr constant: a null pointer constant (approximated as void *).
            return N.Cast(t.loc, N.IntLit(t.loc, 0, ty=T.TY_INT), ty=T.pointer_to(T.TY_VOID))
        is_call = self.at("(")
        if name.startswith("__builtin_"):
            bname = name[len("__builtin_"):]
            if is_call and bname in POWI_HELPERS:
                return self.typed(N.VarRef(t.loc, self.powi_helper(t, bname)))
            if is_call or libc_return_type(bname) is not None:
                # &__builtin_abs and __builtin_abs(...) both name the library function.
                name = floatn_libc_name(bname)
                vs = self.find_var(name)
                if vs is not None and vs.var is not None:
                    return self.typed(N.VarRef(t.loc, vs.var))
                is_call = True
        if is_call:
            rty = libc_return_type(name)
            if rty is None:
                diag.warn(t, f"implicit declaration of function '{name}'")
                rty = T.TY_INT
            return self.typed(N.VarRef(t.loc, self.implicit_function(name, t, rty)))
        self.error(f"'{name}' undeclared", t)

    def c23_mode(self):
        std = (getattr(self.opts, "std", None) or "gnu11").lower()
        return std in ("c2x", "gnu2x", "c23", "gnu23")

    def string_var(self, tok):
        var = self.new_anon_gvar(tok.ty, tok.loc)
        var.init_data = N.GlobalData(bytearray(tok.value))
        var.is_string_literal = True     # __builtin_constant_p folds these to 1
        return var

    def func_name_var(self):
        if getattr(self, "_func_name_var", None) is None or self._func_name_var[0] is not self.fn:
            data = self.fn.name.encode() + b"\0"
            var = self.new_anon_gvar(T.array_of(T.TY_CHAR, len(data)))
            var.init_data = N.GlobalData(bytearray(data))
            self._func_name_var = (self.fn, var)
        return self._func_name_var[1]

    def implicit_function(self, name, tok, ret_ty=None):
        fty = T.func_type(ret_ty or T.TY_INT)
        fty.has_proto = False
        fn = N.Function(name, fty, loc=tok.loc)
        self.scopes[0].vars.setdefault(name, VarScope()).var = fn
        self.objs.append(fn)
        return fn

    def powi_helper(self, tok, bname):
        """The libgcc helper behind __builtin_powi{f,,l}, declared once with a
        prototype (T x, int n) so that x is converted to its C type."""
        helper, ty = POWI_HELPERS[bname]
        vs = self.find_var(helper)
        if vs is not None and vs.var is not None:
            return vs.var
        fty = T.func_type(ty)
        fty.params = [ty, T.TY_INT]
        fn = N.Function(helper, fty, loc=tok.loc)
        self.scopes[0].vars.setdefault(helper, VarScope()).var = fn
        self.objs.append(fn)
        return fn

    def generic_selection(self):
        t = self.next()
        self.expect("(")
        ctrl = self.typed(self.assign())
        cty = ctrl.ty
        if cty.kind == T.TypeKind.ARRAY:
            cty = T.pointer_to(cty.base)
        elif cty.kind == T.TypeKind.FUNC:
            cty = T.pointer_to(cty)
        cty = cty.unqual()
        chosen = None
        default = None
        while self.accept(","):
            if self.accept("default"):
                self.expect(":")
                default = self.assign()
                continue
            ty = self.typename()
            self.expect(":")
            node = self.assign()
            if chosen is None and is_compatible(cty, ty.unqual()):
                chosen = node
        self.expect(")")
        if chosen is None:
            chosen = default
        if chosen is None:
            self.error("_Generic selector of type '%s' is not compatible with any association" % cty, t)
        return chosen
