"""Translation unit and file-scope declarations.

TopLevelMixin provides:
    translation_unit() -> TranslationUnit
    declare_function(ty, name_tok, attr, dattr) -> Function    (find or create)
    declare_global_var(ty, name_tok, attr, dattr) -> Var       (find or create)
    function_definition(fn)                                    parse the body
Redeclarations reuse the first Obj, so `extern int x; int x = 1;` is one Var.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.parser.scope import VarAttr, VarScope
from hcc.sema.compat import is_compatible
from hcc.sema.conversions import new_cast
from hcc import typesys as T
from hcc.parser.attributes import mode_type
from hcc.parser.pragma_pack import at_pack_marker, consume_pack_marker
from hcc.parser.incomplete_check import is_incomplete_object
from hcc.parser.redecl import check_redecl
from hcc.parser.knr_check import knr_param_index

GNU89_STDS = ("c89", "c90", "gnu89", "gnu90")


def is_incomplete_object(ty) -> bool:
    """True for a type that cannot name an object's storage: an incomplete
    struct/union, or an enum that was only forward-declared (`enum E;`)."""
    if ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
        return not ty.is_complete
    if ty.kind == T.TypeKind.ENUM:
        return ty.enum_consts == "fwd"
    return False


class TopLevelMixin:

    def translation_unit(self) -> N.TranslationUnit:
        while not self.at_eof():
            if self.accept(";"):
                continue
            if at_pack_marker(self):
                consume_pack_marker(self)
                continue
            if self.at_std_attribute():
                self.skip_std_attributes()
                continue
            if self.at("__extension__"):     # GNU no-op marker: `__extension__ [[]] int a;`
                self.next()
                continue
            if self.at("_Static_assert"):
                self.static_assert()
                continue
            if self.at("asm"):
                t = self.next()
                node = self.asm_stmt(t)
                self.expect(";")
                self.objs.append(node)
                continue
            attr = VarAttr()
            basety = self.declspec(attr)
            if self.accept(";"):
                continue
            if basety is None:                  # file-scope __auto_type x = expr;
                self.auto_type_global(attr)
                continue
            self.file_declaration(basety, attr)
        self.finish_tentatives()
        return N.TranslationUnit(self.objs)

    def file_declaration(self, basety, attr) -> None:
        first = True
        while True:
            if not first:
                self.expect(",")
            first = False
            ty, name = self.declarator(basety)
            dattr = VarAttr()
            self.parse_attributes(dattr)
            self.parse_asm_label(dattr)
            self.parse_attributes(dattr)
            if name is None:
                self.error("identifier expected in declaration")
            if dattr.vector_size:
                ty = T.vector_of(ty, dattr.vector_size)
            if dattr.mode:
                ty = mode_type(ty, dattr.mode)
            if not attr.is_typedef and ty.kind != T.TypeKind.FUNC:
                ty = self.var_aligned(ty, attr, dattr)      # `T x __attribute__((aligned(N)))`
            if attr.is_typedef:
                ty = self.typedef_aligned(ty, attr, dattr, name.text)
                check_redecl(self, self.scopes[0], name, "typedef")
                self.scopes[0].vars[name.text] = VarScope(type_def=ty)
            elif ty.kind == T.TypeKind.FUNC:
                if attr.is_register:
                    self.error(f"invalid storage class for function '{name.text}'", name)
                check_redecl(self, self.scopes[0], name, "func")
                fn = self.declare_function(ty, name, attr, dattr)
                if ty.enum_consts == "knr" and not self.at(";") and not self.at(","):
                    self.knr_params(ty, fn.ty)
                if self.at("{"):
                    fn.ty = ty
                    self.function_definition(fn, ty, self.gnu89_extern_inline(attr))
                    return
            else:
                check_redecl(self, self.scopes[0], name, "var")
                if not attr.is_extern and ty.kind == T.TypeKind.ENUM and is_incomplete_object(ty):
                    # An enum forward reference can never gain a body, so the size stays unknown.
                    self.error(f"storage size of '{name.text}' isn't known", name)
                var = self.declare_global_var(ty, name, attr, dattr)
                var.is_constexpr = attr.is_constexpr
                if self.accept("="):
                    self.global_initializer(var)
                    var.is_definition = True
                    var.is_tentative = False
            if self.accept(";"):
                return

    def auto_type_global(self, attr) -> None:
        """`__auto_type name = expr;` at file scope: the variable takes the
        type of the initializer expression (array decays to pointer)."""
        name = self.auto_declarator_name()
        self.expect("=")
        start = self.pos
        ty = self.typed(self.assign()).ty
        if ty.kind == T.TypeKind.ARRAY:
            ty = T.pointer_to(ty.base)
        self.pos = start                    # re-parse the initializer with the real type
        var = self.declare_global_var(ty, name, attr, VarAttr())
        self.global_initializer(var)
        var.is_definition = True
        var.is_tentative = False
        self.expect(";")

    def auto_declarator_name(self):
        """Name in a type-inferred `auto x = e;`; redundant parentheses around it
        (`auto ((x)) = e;`) are allowed, as in any declarator."""
        depth = 0
        while self.accept("("):
            depth += 1
        name = self.expect_ident()
        for _ in range(depth):
            self.expect(")")
        self.skip_std_attributes()          # `auto x [[attr]] = e;`
        return name

    # ---- objects -----------------------------------------------------------------
    def declare_function(self, ty, name, attr, dattr) -> N.Function:
        vs = self.scopes[0].vars.get(name.text)
        fn = vs.var if vs is not None and isinstance(vs.var, N.Function) else None
        if fn is None:
            fn = N.Function(name.text, ty, loc=name.loc)
            self.scopes[0].vars[name.text] = VarScope(var=fn)
            self.objs.append(fn)
        else:
            if (dattr.alias or attr.alias or dattr.ifunc or attr.ifunc) and \
                    (fn.is_definition or fn.alias_of or fn.ifunc_of):
                self.error(f"redefinition of '{name.text}'", name.loc)
            if ty.has_proto or not fn.ty.has_proto:
                fn.ty = ty
        fn.is_static |= attr.is_static
        fn.is_inline |= attr.is_inline
        fn.is_noreturn |= attr.is_noreturn or dattr.is_noreturn
        fn.is_weak |= attr.is_weak or dattr.is_weak
        fn.fn_attrs |= attr.fn_attrs | dattr.fn_attrs
        fn.asm_name = dattr.asm_name or attr.asm_name or fn.asm_name
        fn.section = dattr.section or attr.section or fn.section
        fn.visibility = dattr.visibility or attr.visibility or fn.visibility
        fn.alias_of = dattr.alias or attr.alias or fn.alias_of
        fn.ifunc_of = dattr.ifunc or attr.ifunc or fn.ifunc_of
        for a in (attr, dattr):
            if a.init_kind:
                fn.init_kind, fn.init_prio = a.init_kind, a.init_prio
        fn.align = max(fn.align, attr.align, dattr.align)
        return fn

    def declare_global_var(self, ty, name, attr, dattr) -> N.Var:
        vs = self.scopes[0].vars.get(name.text)
        var = vs.var if vs is not None and isinstance(vs.var, N.Var) else None
        if var is None:
            var = N.Var(name.text, ty, loc=name.loc, align=ty.align)
            self.scopes[0].vars[name.text] = VarScope(var=var)
            self.objs.append(var)
            var.is_definition = False
        elif ty.is_complete and not var.ty.is_complete or var.ty.kind == T.TypeKind.ARRAY and \
                var.ty.array_len < 0 and ty.kind == T.TypeKind.ARRAY and ty.array_len >= 0:
            var.ty = ty
        var.is_static |= attr.is_static
        var.is_tls |= attr.is_tls
        var.is_weak |= attr.is_weak or dattr.is_weak
        var.is_nocommon |= attr.is_nocommon or dattr.is_nocommon
        var.asm_name = dattr.asm_name or var.asm_name
        var.section = dattr.section or attr.section or var.section
        var.visibility = dattr.visibility or attr.visibility or var.visibility
        var.alias_of = dattr.alias or attr.alias or var.alias_of
        al = max(attr.align, dattr.align)
        var.align = max(var.align, al, var.ty.align)
        if not attr.is_extern:
            var.is_extern = False
            if var.init_data is None:
                var.is_tentative = True
        elif not var.is_tentative and var.init_data is None:
            var.is_extern = True
        return var

    def finish_tentatives(self) -> None:
        for o in self.objs:
            if isinstance(o, N.Var) and not o.is_local and o.init_data is None:
                if o.is_tentative and not o.is_extern:
                    # C11 6.9.2p2: a tentative definition must have a complete type
                    # by the end of the translation unit (gcc: "storage size isn't known").
                    if is_incomplete_object(o.ty):
                        self.error(f"storage size of '{o.name}' isn't known", o.loc)
                    o.is_definition = True
                    if o.ty.kind == T.TypeKind.ARRAY and o.ty.array_len < 0:
                        o.ty = T.array_of(o.ty.base, 1)

    # ---- function bodies ---------------------------------------------------------
    def knr_params(self, fty, proto=None) -> None:
        """K&R parameter declarations between `)` and `{`. `proto` is the
        prior prototype of the same function, if any, that they must match."""
        decls = {}
        while not self.at("{"):
            attr = VarAttr()
            basety = self.declspec(attr)
            while True:
                ty, name = self.declarator(basety)
                dattr = VarAttr()           # `int foo __attribute__((unused));`
                self.parse_attributes(dattr)
                self.skip_std_attributes()
                if dattr.mode:
                    ty = mode_type(ty, dattr.mode)
                if name is not None:        # empty declaration (`int;`) is ignored
                    if ty.kind == T.TypeKind.ARRAY:
                        ty = T.pointer_to(ty.base)
                    elif ty.kind == T.TypeKind.FUNC:
                        ty = T.pointer_to(ty)
                    decls[name.text] = ty
                if not self.accept(","):
                    break
            self.expect(";")
        declared = [decls.get(n.text, T.TY_INT) for n in fty.param_names]
        if proto is not None and proto.has_proto:
            if len(proto.params) != len(declared):
                self.error("number of arguments doesn't match prototype")
            i = knr_param_index(proto, declared)
            if i is not None:
                pname = fty.param_names[i]
                self.error(f"argument '{pname.text}' doesn't match prototype", pname.loc)
        params = []
        for n in fty.param_names:
            ty = decls.get(n.text, T.TY_INT)
            if ty.kind == T.TypeKind.FLOAT and not ty.is_floatn:
                ty = T.TY_DOUBLE        # default argument promotion (not for _Float32)
            params.append(ty)
        fty.params = params

    def check_param_complete(self, i, pty, pname, loc) -> None:
        """C11 6.9.1p10: parameters of a function definition need complete
        non-void types (arrays/pointers were already adjusted by the declarator)."""
        bad = pty.kind == T.TypeKind.VOID or \
            (pty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION) and not pty.is_complete) or \
            (pty.kind == T.TypeKind.ENUM and pty.enum_consts == "fwd")
        if not bad:
            return
        what = f" ('{pname.text}')" if pname is not None else ""
        self.error(f"parameter {i + 1}{what} has incomplete type", pname.loc if pname else loc)

    def bind_vla_param(self, ty, inits, loc):
        """Parameter types like `char a[2][N]` (pointer to a VLA): the length
        is copied into a hidden local on entry (appended to `inits`) and the
        VLA type refers to that local, so element sizes are known at run time."""
        if ty.kind == T.TypeKind.PTR:
            base = self.bind_vla_param(ty.base, inits, loc)
            if base is ty.base:
                return ty
            new = T.pointer_to(base)
            new.is_const = ty.is_const
            return new
        if ty.kind not in (T.TypeKind.ARRAY, T.TypeKind.VLA):
            return ty
        base = self.bind_vla_param(ty.base, inits, loc)
        if ty.kind == T.TypeKind.ARRAY:
            return ty if base is ty.base else T.array_of(base, ty.array_len)
        if ty.vla_len is None:
            return ty if base is ty.base else T.vla_of(base, None)
        tmp = self.new_temp(T.TY_LONG, loc)
        var_ref = self.typed(N.VarRef(loc, tmp))
        inits.append(N.ExprStmt(loc, self.typed(N.Assign(loc, var_ref,
                                                         new_cast(ty.vla_len, T.TY_LONG)))))
        return T.vla_of(base, self.typed(N.VarRef(loc, tmp)))

    def gnu89_extern_inline(self, attr) -> bool:
        """gnu89 inline semantics: `extern inline` with a body is only an
        inline definition and provides no external symbol."""
        o = self.opts
        gnu89 = o is not None and (o.gnu89_inline or o.std in GNU89_STDS) or attr.is_gnu_inline
        return bool(gnu89 and attr.is_extern and attr.is_inline)

    def function_definition(self, fn, ty, inline_only=False) -> None:
        if fn.is_definition:
            self.error(f"redefinition of '{fn.name}'", fn.loc)
        if not inline_only:
            fn.is_definition = True
        self.fn = fn
        # keep temps made while parsing the prototype: VLA parameter dimensions
        # (`int a[i++]`) are parsed before this point and may create temps
        self.label_names = set()
        self.goto_names = []
        # parameter-list tags and names share the body's outermost scope (C11 6.2.1p4)
        if ty.proto_scope is not None:
            self.push_scope(ty.proto_scope)
        else:
            self.enter_scope()
        params = []
        vla_inits = []
        for i, pty in enumerate(ty.params):
            pname = ty.param_names[i] if i < len(ty.param_names) else None
            pvar = ty.param_vars[i] if i < len(ty.param_vars) else None
            self.check_param_complete(i, pty, pname, fn.loc)
            pty = self.bind_vla_param(pty, vla_inits, pname.loc if pname else fn.loc)
            if pname is not None and pvar is not None:
                # reuse the Var the prototype parse already put in scope
                v = pvar
                v.ty = pty
                self.locals.append(v)
                self.push_scope_entry(pname.text).var = v
            elif pname is not None:
                v = self.new_lvar(pname.text, pty, pname.loc)
            else:
                v = self.new_temp(pty)
            v.is_param = True
            params.append(v)
        if fn.outer is not None:                 # static link slot (codegen/nested.py)
            fn.chain_var = self.new_temp(T.pointer_to(T.TY_VOID))
        if ty.is_variadic:
            fn.va_area = self.new_temp(T.array_of(T.TY_CHAR, 176))  # SysV register save area
            fn.va_tmp = self.new_temp(T.array_of(T.TY_CHAR, 16))  # va_arg scratch for split structs
        body_tok = self.expect("{")
        fn.body = self.compound_stmt_body(body_tok.loc, same_scope=True)
        # VLA-typed params (`int b[a++]`): the dimension is evaluated on entry
        dims = [d for d in ty.param_dims if d is not None]
        if dims:
            fn.body.items[0:0] = [N.ExprStmt(d.loc, d) for d in dims]
        if vla_inits:
            fn.body.items[0:0] = vla_inits
        self.leave_scope()
        for g in self.goto_names:
            # a nested function's gotos may target labels of the enclosing
            # function; nested_fn.py forwards them to the enclosing check
            if g.text not in self.label_names and fn.outer is None:
                self.error(f"label '{g.text}' used but not defined", g)
        fn.params = params
        fn.labels = set(self.label_names)
        fn.locals = self.locals
        self.fn = None
        self.locals = []
