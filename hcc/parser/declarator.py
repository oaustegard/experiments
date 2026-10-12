"""Declarators.

DeclaratorMixin provides:
    pointers(ty) -> Type                 `* const * volatile ...`
    declarator(ty) -> (Type, Token | None)   name is None for abstract declarators
    type_suffix(ty) -> Type              `(params)` / `[dims]`
    func_params(ret_ty) -> Type          FUNC type; params decay array->ptr
    array_dims(ty) -> Type
    typename() -> Type                   for casts/sizeof/_Alignof/va_arg
"""
from __future__ import annotations

from hcc.parser.scope import VarAttr
from hcc.sema.constexpr import try_eval_const
from hcc.sema.typing import add_type
from hcc.tokens import TokenKind
from hcc import typesys as T


class DeclaratorMixin:

    def pointers(self, ty):
        while self.accept("*"):
            ty = T.pointer_to(ty)
            while True:
                if self.at("const"):
                    self.next()
                    ty = ty.copy()
                    ty.is_const = True
                elif self.at("volatile") or self.at("_Atomic"):
                    self.next()
                    ty = ty.copy()
                    ty.is_volatile = True
                elif self.at("restrict"):
                    self.next()
                elif self.at("__attribute__"):
                    self.parse_attributes()
                else:
                    break
        return ty

    def declarator(self, ty):
        ty = self.pointers(ty)
        self.parse_attributes()
        if self.at("(") and not self._paren_starts_params():
            start = self.pos
            self.next()
            # skip the inner declarator to find what follows it
            self._skip_declarator()
            self.expect(")")
            ty = self.type_suffix(ty)
            end = self.pos
            self.pos = start + 1
            ty, name = self.declarator(ty)
            self.expect(")")
            self.pos = end
            return ty, name
        name = None
        if self.tok.kind == TokenKind.IDENT:
            name = self.next()
        ty = self.type_suffix(ty)
        return ty, name

    def _paren_starts_params(self) -> bool:
        """In an abstract declarator, `(` may begin a parameter list: `int (int)`."""
        nxt = self.peek()
        return nxt.text == ")" and nxt.kind == TokenKind.PUNCT or self.is_typename(nxt)

    def _skip_declarator(self) -> None:
        depth = 0
        while True:
            t = self.tok
            if t.kind == TokenKind.EOF:
                self.error("unterminated declarator")
            if t.text in ("(", "[") and t.kind == TokenKind.PUNCT:
                depth += 1
            elif t.text in (")", "]") and t.kind == TokenKind.PUNCT:
                if depth == 0:
                    return
                depth -= 1
            self.next()

    def type_suffix(self, ty):
        if self.at("("):
            self.next()
            return self.func_params(ty)
        if self.at("["):
            self.next()
            return self.array_dims(ty)
        return ty

    def func_params(self, ret_ty):
        fty = T.func_type(ret_ty)
        if self.at("void") and self.peek().text == ")":
            self.next()
            self.next()
            return fty
        if self.accept(")"):
            fty.has_proto = False
            return fty
        # K&R identifier list: int f(a, b) int a; ...
        if self.tok.kind == TokenKind.IDENT and not self.is_typename() and \
                self.peek().text in (",", ")"):
            names = []
            while not self.accept(")"):
                if names:
                    self.expect(",")
                names.append(self.expect_ident())
            fty.has_proto = False
            fty.param_names = names
            fty.params = [T.TY_INT] * len(names)
            fty.is_variadic = False
            fty.enum_consts = "knr"       # marker: K&R definition follows
            return fty
        params, names = [], []
        while not self.accept(")"):
            if params:
                self.expect(",")
            if self.accept("..."):
                fty.is_variadic = True
                self.expect(")")
                break
            attr = VarAttr()
            base = self.declspec(attr)
            pty, name = self.declarator(base)
            self.parse_attributes()
            if pty.kind == T.TypeKind.ARRAY or pty.kind == T.TypeKind.VLA:
                q = pty
                pty = T.pointer_to(pty.base)
                pty.is_const = q.is_const
            elif pty.kind == T.TypeKind.FUNC:
                pty = T.pointer_to(pty)
            params.append(pty)
            names.append(name)
        fty.params = params
        fty.param_names = names
        return fty

    def array_dims(self, ty):
        while self.at("static") or self.at("const") or self.at("volatile") or self.at("restrict"):
            self.next()
        if self.accept("]"):
            ty = self.type_suffix(ty)
            return T.array_of(ty, -1)
        if self.at("*") and self.peek().text == "]":
            self.next()
            self.next()
            ty = self.type_suffix(ty)
            return T.vla_of(ty, None)
        expr = self.conditional()
        self.expect("]")
        ty = self.type_suffix(ty)
        n = try_eval_const(expr)
        if n is None or ty.kind == T.TypeKind.VLA:
            add_type(expr)
            return T.vla_of(ty, expr)
        if n < 0:
            self.error("size of array is negative", expr)
        return T.array_of(ty, n)

    def typename(self):
        ty = self.declspec(None)
        return self.abstract_declarator(ty)

    def abstract_declarator(self, ty):
        ty, name = self.declarator(ty)
        if name is not None:
            self.error("unexpected identifier in type name", name)
        return ty
