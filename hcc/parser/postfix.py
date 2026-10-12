"""Postfix expressions.

PostfixMixin provides:
    postfix()                    primary followed by [] () . -> ++ --
    postfix_ops(node)            the suffix loop on an existing node
    compound_literal()           `(type){init}` at file or block scope
    funcall(fn_node, loc)        argument conversion + Call node
    build_call(fn, args, loc)    the Call node for already-parsed arguments
    struct_ref(node, name_tok)   member access via find_member path
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.parser.libc_builtins import fabs_float_call, int_abs_call
from hcc.parser.structs import find_member
from hcc.parser.plan9_ptr import plan9_pointer_conv
from hcc.sema.arith import inc_dec, new_add
from hcc.parser import complex as CPX
from hcc.sema.conversions import default_arg_promote, new_cast
from hcc.typesys import TypeKind
from hcc.typesys import TypeKind, is_struct_like
from hcc.parser.va_pack import is_pack_placeholder


def _array_chain_is_vla(ty):
    """True when ty is a VLA or an array (of arrays) whose element type is a VLA.

    C forbids compound literals of variable length type (gcc: "compound literal
    has variable size"); pointers to VLAs are fine and are not followed here.
    """
    while ty is not None and ty.kind == TypeKind.ARRAY:
        ty = ty.base
    return ty is not None and ty.kind == TypeKind.VLA


class PostfixMixin:

    def postfix(self):
        if self.at("(") and self.is_typename(self.peek()):
            save = self.pos
            allow = self.compound_lit_follows()
            is_tls = allow and self.compound_lit_is_static(("_Thread_local",))
            is_static = allow and (is_tls or self.compound_lit_is_static())
            self.next()
            ty = self.typename(allow_storage=allow)
            self.expect(")")
            if self.at("{"):
                if _array_chain_is_vla(ty):
                    self.error("compound literal has variable size", self.tok)
                ty, pre = self.vla_bind(ty, self.tok.loc)
                lit = self.compound_literal(ty, is_static, is_tls)
                return self.postfix_ops(self.vla_prefix(pre, lit, self.tok.loc))
            self.pos = save
        return self.postfix_ops(self.primary())

    def postfix_ops(self, node):
        while True:
            t = self.tok
            s = t.text
            if s == "[" and t.kind.name == "PUNCT":
                self.next()
                idx = self.expr()
                self.expect("]")
                ptr = new_add(node, idx, t.loc)
                self.typed(ptr)
                if ptr.ty.base is not None and ptr.ty.base.kind == TypeKind.FUNC:
                    diag.error(t.loc, "subscripted value is pointer to function", "sema")
                node = self.typed(N.Deref(t.loc, ptr))
            elif s == "(" and t.kind.name == "PUNCT":
                self.next()
                node = self.funcall(node, t.loc)
            elif s == "." and t.kind.name == "PUNCT":
                self.next()
                node = self.struct_ref(node, self.expect_ident())
            elif s == "->" and t.kind.name == "PUNCT":
                self.next()
                self.typed(node)
                if node.ty.base is None:
                    self.error("invalid type argument of '->'", t)
                node = self.struct_ref(self.typed(N.Deref(t.loc, node)), self.expect_ident())
            elif s in ("++", "--") and t.kind.name == "PUNCT":
                self.next()
                node = inc_dec(node, 1 if s == "++" else -1, True, t.loc, self.new_temp)
            else:
                return node

    def struct_ref(self, node, name_tok):
        self.typed(node)
        ty = node.ty
        if not is_struct_like(ty):
            self.error(f"request for member '{name_tok.text}' in something not a structure or union", name_tok)
        path = find_member(ty, name_tok.text)
        if path is None:
            self.error(f"'{ty}' has no member named '{name_tok.text}'", name_tok)
        for m in path:
            mty = m.ty
            if ty.is_const and not mty.is_const:
                mty = mty.copy()
                mty.is_const = True
            node = N.Member_(name_tok.loc, node, m, ty=mty)
        return node

    def funcall(self, fn, loc):
        self.call_type(fn, loc)
        args = []
        while not self.accept(")"):
            if args:
                self.expect(",")
            args.append(self.typed(self.assign()))
        return self.build_call(fn, args, loc)

    def call_type(self, fn, loc):
        """The FUNC type a call of `fn` uses (function or function pointer)."""
        self.typed(fn)
        fty = fn.ty
        if fty.kind == TypeKind.PTR and fty.base.kind == TypeKind.FUNC:
            fty = fty.base
        if fty.kind != TypeKind.FUNC:
            self.error("called object is not a function or function pointer", loc)
        return fty

    def build_call(self, fn, args, loc):
        """Call node for `fn(args...)`; `args` are typed expressions (parser-side tgmath too)."""
        fty = self.call_type(fn, loc)
        params = fty.params if fty.has_proto else []
        # __builtin_va_arg_pack() stands for an unknown number of arguments until
        # parser/va_pack.py specialises the call: skip arity checks and conversions
        # from the placeholder on.
        pack_at = next((i for i, a in enumerate(args) if is_pack_placeholder(a)), len(args))
        if pack_at == len(args):
            if fty.has_proto and len(args) < len(params):
                self.error("too few arguments to function", loc)
            if fty.has_proto and not fty.is_variadic and len(args) > len(params):
                self.error("too many arguments to function", loc)
        conv = []
        for i, a in enumerate(args):
            if i >= pack_at:
                conv.append(a)
                continue
            if i < len(params):
                pty = params[i]
                if pty.is_complex or a.ty.is_complex:
                    a = CPX.convert(self, a, pty, loc)
                elif not is_struct_like(pty):
                    a = plan9_pointer_conv(self, a, pty)
                    a = new_cast(a, pty)
            else:
                if a.ty.kind in (TypeKind.ARRAY, TypeKind.FUNC):
                    a = new_cast(a, a.ty) if False else a
                a = default_arg_promote(a)
            conv.append(a)
        if fn.__class__ is N.VarRef and isinstance(fn.var, N.Function) and len(conv) == 1:
            abs_node = int_abs_call(loc, fn.var.name, conv[0])
            if abs_node is not None:
                return self.typed(abs_node)
            fabs_node = fabs_float_call(loc, fn.var.name, args[0])
            if fabs_node is not None:
                return self.typed(fabs_node)
        node = N.Call(loc, fn, conv, fty)
        if is_struct_like(fty.return_ty) or fty.return_ty.is_vector:
            node.ret_buffer = self.new_temp(fty.return_ty)
        return self.typed(node)

    def compound_literal(self, ty, is_static=False, is_tls=False):
        loc = self.tok.loc
        if self.fn is None or self.in_static_init or is_static:   # `static` => static storage
            # Static storage: the literal is itself a static object whose address/value
            # the enclosing constant initializer refers to (see sema/initializers.py).
            var = self.new_anon_gvar(ty, loc)
            var.is_compound_literal = True
            var.is_tls = is_tls                 # `thread_local` => per-thread storage
            self.global_initializer(var)
            return self.typed(N.VarRef(loc, var))
        var = self.new_temp(ty, loc)
        init = self.local_initializer(var)
        return self.typed(N.Comma(loc, init, self.typed(N.VarRef(loc, var)), lvalue=True))


__all__ = ["PostfixMixin", "diag"]
