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
from hcc.nodes import Var
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
                elif self.at("__attribute__") or self.at_std_attribute():
                    self.parse_attributes()
                else:
                    break
        return ty

    def declarator(self, ty):
        ty = self.pointers(ty)
        self.parse_attributes()
        ty = self.pointers(ty)          # `int (__attribute__((x)) *)(void)`
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
            self.skip_std_attributes()  # `void f [[]] (void);`
        ty = self.type_suffix(ty)
        return ty, name

    def _paren_starts_params(self) -> bool:
        """In an abstract declarator, `(` may begin a parameter list: `int (int)`."""
        i = 1
        # skip attribute groups: `(__attribute__((x)) *)` starts a declarator, not params
        while self.peek(i).text in ("__attribute__", "__declspec"):
            i += 1
            depth = 0
            while self.peek(i).kind != TokenKind.EOF:
                t = self.peek(i)
                i += 1
                if t.kind == TokenKind.PUNCT and t.text == "(":
                    depth += 1
                elif t.kind == TokenKind.PUNCT and t.text == ")":
                    depth -= 1
                    if depth == 0:
                        break
        nxt = self.peek(i)
        if nxt.kind == TokenKind.PUNCT and nxt.text in (")", "..."):
            return True          # `int (void-less)` or variadic `int (...)`
        return self.is_typename(nxt)

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
        params, names, pvars, pdims = [], [], [], []
        # Prototype scope: earlier named params are visible in later array
        # dimensions (`int b[a++]`); their Vars are reused by the definition.
        self.enter_scope()
        while not self.accept(")"):
            if params:
                self.expect(",")
            if self.accept("..."):
                fty.is_variadic = True
                self.expect(")")
                break
            self.skip_std_attributes()
            attr = VarAttr()
            base = self.declspec(attr)
            pty, name = self.declarator(base)
            self.parse_attributes()
            dim = None
            if (pty.kind == T.TypeKind.ARRAY and not pty.is_vector) or pty.kind == T.TypeKind.VLA:
                q = pty
                dim = q.vla_len if q.kind == T.TypeKind.VLA else None
                pty = T.pointer_to(pty.base)
                pty.is_const = q.is_const
            elif pty.kind == T.TypeKind.FUNC:
                pty = T.pointer_to(pty)
            pvar = None
            if name is not None:
                pvar = Var(name.text, pty, loc=name.loc, is_local=True,
                           is_definition=True, align=pty.align)
                pvar.is_param = True
                self.push_scope_entry(name.text).var = pvar
            params.append(pty)
            names.append(name)
            pvars.append(pvar)
            pdims.append(dim)
        fty.proto_scope = self.scopes[-1]
        self.leave_scope()
        fty.params = params
        fty.param_names = names
        fty.param_vars = pvars
        fty.param_dims = pdims
        return fty

    def array_dims(self, ty):
        while (self.at("static") or self.at("const") or self.at("volatile")
               or self.at("restrict") or self.at("_Atomic")):
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

    def typename(self, allow_storage=False):
        # allow_storage: C23 compound literal `(static int){1}`; storage class
        # specifiers are otherwise rejected (casts, sizeof, _Alignof).
        ty = self.declspec(VarAttr() if allow_storage else None)
        return self.abstract_declarator(ty)

    def compound_lit_follows(self):
        """At '(' : True if the matching ')' is followed by '{'."""
        end = self._paren_end()
        if end is None:
            return False
        nxt = self.toks[min(end + 1, len(self.toks) - 1)]
        return nxt.kind == TokenKind.PUNCT and nxt.text == "{"

    def compound_lit_is_static(self):
        """At '(' : True if `static` appears directly inside the parens."""
        end = self._paren_end()
        if end is None:
            return False
        depth = 0
        for t in self.toks[self.pos:end]:
            if t.kind == TokenKind.PUNCT and t.text in ("(", "[", "{"):
                depth += 1
            elif t.kind == TokenKind.PUNCT and t.text in (")", "]", "}"):
                depth -= 1
            elif depth == 1 and t.text == "static":
                return True
        return False

    def _paren_end(self):
        depth = 0
        for i in range(self.pos, len(self.toks)):
            t = self.toks[i]
            if t.kind == TokenKind.PUNCT and t.text in ("(", "[", "{"):
                depth += 1
            elif t.kind == TokenKind.PUNCT and t.text in (")", "]", "}"):
                depth -= 1
                if depth == 0:
                    return i
        return None

    def scoped_typename(self, allow_storage=False):
        """typename() for sizeof/_Alignof/typeof: tags and enumerators declared
        inside the type name are local to it (PR c/67784), so they do not leak
        into the enclosing block scope."""
        self.enter_scope()
        try:
            return self.typename(allow_storage)
        finally:
            self.leave_scope()

    def abstract_declarator(self, ty):
        ty, name = self.declarator(ty)
        if name is not None:
            self.error("unexpected identifier in type name", name)
        return ty
