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

from dataclasses import fields

from hcc.parser.scope import VarAttr
from hcc.parser.declspec import constexpr_type
from hcc import nodes as N
from hcc.nodes import Node, Var
from hcc.sema.constexpr import try_eval_const
from hcc.sema.typing import add_type
from hcc.tokens import TokenKind
from hcc import typesys as T


def _names_variable(node) -> bool:
    """True if the expression tree refers to a variable (not a function)."""
    if isinstance(node, Node):
        if isinstance(node, N.VarRef) and not isinstance(node.var, N.Function):
            return True
        return any(_names_variable(getattr(node, f.name)) for f in fields(node))
    if isinstance(node, (list, tuple)):
        return any(_names_variable(x) for x in node)
    return False


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
        self.skip_std_attributes()  # `int g [2] [[]] [3] [[]];`
        if self.at("("):
            self.next()
            fty = self.func_params(ty)
            self.skip_std_attributes()  # `void (int) [[gnu::format(...)]]`
            return fty
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
            # C23 6.7.6.3p15: `()` means `(void)` (a prototype); earlier
            # standards leave the parameter list unspecified.
            fty.has_proto = self.c23_mode()
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
        fwd = self._param_forward_decls()
        outer_proto = getattr(self, "in_proto_params", False)
        self.in_proto_params = True     # sizeof(*b) of `[*]` params is allowed here
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
            if attr.is_static or attr.is_extern or attr.is_typedef:
                self.error("storage class specified for parameter")
            saved_proto = self.in_proto_params
            saved_pdecl = self.in_param_decl
            self.in_proto_params = True     # sizeof in a prototype is unevaluated
            self.in_param_decl = True
            try:
                pty, name = self.declarator(base)
            finally:
                self.in_proto_params = saved_proto
                self.in_param_decl = saved_pdecl
            if pty.kind == T.TypeKind.VOID and (pty.is_const or pty.is_volatile or attr.is_register):
                # C23 6.7.6.3p10: a parameter of type void may not be qualified
                # (`const void`, `cv` typedef, `register void`); `const void *` is fine.
                self.error("'void' as only parameter may not be qualified")
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
            if name is not None and name.text in fwd:
                pvar = fwd[name.text]       # the real param is the forward-declared Var
                pvar.ty, pvar.loc, pvar.align = pty, name.loc, pty.align
                self.push_scope_entry(name.text).var = pvar
            elif name is not None:
                pvar = Var(name.text, pty, loc=name.loc, is_local=True,
                           is_definition=True, align=pty.align)
                pvar.is_param = True
                self.push_scope_entry(name.text).var = pvar
            params.append(pty)
            names.append(name)
            pvars.append(pvar)
            pdims.append(dim)
        fty.proto_scope = self.scopes[-1]
        self.in_proto_params = outer_proto
        self.leave_scope()
        if len(params) == 1 and names[0] is None and params[0].kind == T.TypeKind.VOID:
            # C23 6.7.6.3p10: a sole unnamed parameter of type void (e.g. from
            # `typeof (*vp)`, not just the keyword) means an empty `(void)` list.
            params, names, pvars, pdims = [], [], [], []
        fty.params = params
        fty.param_names = names
        fty.param_vars = pvars
        fty.param_dims = pdims
        return fty

    def _param_forward_decls(self) -> dict:
        """GNU parameter forward declarations: `f(int n; int m, int n)`. Each `;`
        at depth 0 before the closing `)` ends one list of forward-declared
        parameters; the final list is the real parameter list. Forward names are
        declared in the prototype scope so array bounds can use them; the real
        parameter with the same name reuses that Var. Returns name -> Var."""
        fwd = {}
        for _ in range(self._count_fwd_lists()):
            while True:
                self.skip_std_attributes()
                pty, name = self.declarator(self.declspec(VarAttr()))
                self.parse_attributes()
                if name is not None:
                    pvar = Var(name.text, pty, loc=name.loc, is_local=True,
                               is_definition=True, align=pty.align)
                    pvar.is_param = True
                    self.push_scope_entry(name.text).var = pvar
                    fwd[name.text] = pvar
                if not self.accept(","):
                    break
            self.expect(";")
        return fwd

    def _count_fwd_lists(self) -> int:
        """Number of `;` at depth 0 between the current position and the `)`
        that closes the parameter list."""
        depth, count = 0, 0
        for t in self.toks[self.pos:]:
            if t.kind == TokenKind.EOF:
                break
            if t.kind != TokenKind.PUNCT:
                continue
            if t.text in ("(", "[", "{"):
                depth += 1
            elif t.text in (")", "]", "}"):
                if depth == 0:
                    break
                depth -= 1
            elif t.text == ";" and depth == 0:
                count += 1
        return count

    def check_array_elem(self, ty):
        """Reject arrays whose element type is incomplete (C11 6.7.6.2p1)."""
        if ty.kind == T.TypeKind.VLA:
            return
        if ty.kind == T.TypeKind.VOID:
            # C11 6.7.6.2p1: the element type of an array must be a complete object type
            self.error("declaration of type name as array of voids", self.tok)
        if ty.kind == T.TypeKind.FUNC:
            # C11 6.7.6.2p1: an element type must not be a function type
            self.error("declaration as array of functions", self.tok)
        if (ty.kind == T.TypeKind.ARRAY and ty.array_len < 0) or \
                (ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION) and not ty.is_complete):
            self.error("array type has incomplete element type", self.tok)
        elif ty.align > 0 and ty.size % ty.align != 0 and ty.is_complete:
            # GCC: `aligned` on a typedef raises the alignment without padding the size,
            # so the element stride would be misaligned; gcc rejects such arrays.
            self.error("size of array element is not a multiple of its alignment", self.tok)

    def array_dims(self, ty):
        has_static = False
        while (self.at("static") or self.at("const") or self.at("volatile")
               or self.at("restrict") or self.at("_Atomic")):
            if self.at("static"):
                has_static = True
            self.next()
        if has_static and self.at("]"):
            # C99 6.7.5.2p1: `static` requires a size expression
            self.error("'static' may not be used without an array size")
        if has_static and self.at("*") and self.peek().text == "]":
            self.error("'static' may not be used with an unspecified variable length array size")
        if self.accept("]"):
            ty = self.type_suffix(ty)
            self.check_array_elem(ty)
            return T.array_of(ty, -1)
        if self.at("*") and self.peek().text == "]":
            # C11 6.7.6.2p4: `[*]` only in a prototype's parameter list
            if not getattr(self, "in_proto_params", False):
                self.error("'[*]' not allowed in other than function prototype scope")
            self.next()
            self.next()
            ty = self.type_suffix(ty)
            return T.vla_of(ty, None)
        expr = self.assign()
        self.expect("]")
        add_type(expr)
        if not T.is_integer(expr.ty):
            self.error("size of array has non-integer type", expr)
        ty = self.type_suffix(ty)
        self.check_array_elem(ty)
        n = try_eval_const(expr)
        # C: a block-scope object (even const-qualified) is not an integer constant
        # expression, so `char [i]` with `const int i = 1;` is still a VLA.
        if self.fn is not None and n is not None and _names_variable(expr):
            n = None
        if n is None or ty.kind == T.TypeKind.VLA or T.is_dyn_struct(ty):
            # an array of run-time-sized structs has a run-time size too (`struct S a[2][n]`)
            add_type(expr)
            return T.vla_of(ty, expr)
        if n < 0:
            self.error("size of array is negative", expr)
        return T.array_of(ty, n)

    def typename(self, allow_storage=False):
        # allow_storage: C23 compound literal `(static int){1}`; storage class
        # specifiers are otherwise rejected (casts, sizeof, _Alignof).
        attr = VarAttr() if allow_storage else None
        ty = self.declspec(attr)
        ty = self.abstract_declarator(ty)
        if attr is not None and attr.is_constexpr:
            ty = constexpr_type(ty)          # `(constexpr T){...}`: the literal object is const
        return ty

    def compound_lit_follows(self):
        """At '(' : True if the matching ')' is followed by '{'."""
        end = self._paren_end()
        if end is None:
            return False
        nxt = self.toks[min(end + 1, len(self.toks) - 1)]
        return nxt.kind == TokenKind.PUNCT and nxt.text == "{"

    def compound_lit_is_static(self, words=("static",)):
        """At '(' : True if one of `words` (default `static`) appears directly inside the parens."""
        end = self._paren_end()
        if end is None:
            return False
        depth = 0
        for t in self.toks[self.pos:end]:
            if t.kind == TokenKind.PUNCT and t.text in ("(", "[", "{"):
                depth += 1
            elif t.kind == TokenKind.PUNCT and t.text in (")", "]", "}"):
                depth -= 1
            elif depth == 1 and t.text in words:
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
        """typename() for sizeof/_Alignof/typeof. Tags and enumerators declared
        inside the type name go to the innermost scope: the block, or the
        if/while/do/for/switch statement whose controlling expression holds them
        (PR c/67784)."""
        return self.typename(allow_storage)

    def abstract_declarator(self, ty):
        ty, name = self.declarator(ty)
        if name is not None:
            self.error("unexpected identifier in type name", name)
        return ty
