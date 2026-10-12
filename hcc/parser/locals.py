"""Block-scope declarations.

LocalDeclMixin.local_declaration() -> list[Stmt]
Handles typedefs, function prototypes, extern and static locals, automatic
variables with initializers (lowered to ExprStmt), __auto_type.
VLA typedefs evaluate their length at the declaration (see vla.py); VLA
objects: TODO (currently a parse error).
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser.scope import VarAttr
from hcc.parser.attributes import mode_type

# 64-bit general registers accepted in `register T x __asm__("reg")` local bindings.
GP64_REGS = frozenset("rax rbx rcx rdx rsi rdi rbp rsp r8 r9 r10 r11 r12 r13 r14 r15".split())


class LocalDeclMixin:

    def local_declaration(self) -> list:
        attr = VarAttr()
        self.vm_typeof = []
        basety = self.declspec(attr)
        items = [N.ExprStmt(e.loc, e) for e in self.vm_typeof]
        self.vm_typeof = []
        first = True
        while not self.accept(";"):
            if not first:
                self.expect(",")
            first = False
            if basety is None:                       # __auto_type x = expr;
                name = self.auto_declarator_name()
                self.expect("=")
                e = self.typed(self.assign())
                ty = e.ty
                if ty.kind in (T.TypeKind.ARRAY, T.TypeKind.VLA):   # arrays decay (VLAs too)
                    ty = T.pointer_to(ty.base)
                var = self.new_lvar(name.text, ty, name.loc)
                items.append(N.ExprStmt(name.loc, self.typed(N.Assign(name.loc, self.typed(N.VarRef(name.loc, var)), e))))
                continue
            ty, name = self.declarator(basety)
            dattr = VarAttr()
            self.parse_attributes(dattr)
            self.skip_std_attributes()
            self.parse_asm_label(dattr)
            if name is None:
                self.error("identifier expected in declaration")
            if dattr.mode:
                ty = mode_type(ty, dattr.mode)
            if not attr.is_typedef and ty.kind != T.TypeKind.FUNC:
                ty = self.typedef_aligned(ty, attr, dattr)     # `T x __attribute__((aligned(N)))`
            if attr.is_typedef:
                ty = self.typedef_aligned(ty, attr, dattr, name.text)
                ty, pre = self.vla_bind(ty, name.loc)
                items.extend(pre)
                self.push_scope_entry(name.text).type_def = ty
                continue
            if ty.kind == T.TypeKind.FUNC and ty.enum_consts == "knr" \
                    and not self.at(";") and not self.at(","):
                self.knr_params(ty)              # GNU nested K&R definition
            if ty.kind == T.TypeKind.FUNC and self.at("{"):
                self.nested_function(ty, name, attr, dattr)
                return items
            if ty.kind == T.TypeKind.FUNC:
                ty = self.block_func_composite(ty, name.text)
                fn = self.declare_function(ty, name, attr, dattr)
                self.push_scope_entry(name.text).var = fn
                if self.at("{"):                     # GNU nested function definition
                    self.nested_function_definition(fn, ty)
                    return items
                continue
            if attr.is_extern:
                var = self.declare_global_var(ty, name, attr, dattr)
                self.push_scope_entry(name.text).var = var
                continue
            if attr.is_static:
                self.anon_count += 1
                var = N.Var(f"{name.text}.{self.anon_count}", ty, is_static=True, loc=name.loc,
                            is_definition=True, align=max(attr.align, dattr.align) or ty.align,
                            is_tls=attr.is_tls)
                self.objs.append(var)
                self.push_scope_entry(name.text).var = var
                if self.accept("="):
                    self.global_initializer(var)
                elif not ty.is_complete and ty.kind != T.TypeKind.ARRAY:
                    self.error(f"storage size of '{name.text}' isn't known", name)
                continue
            if ty.kind == T.TypeKind.VOID:
                self.error(f"variable '{name.text}' declared void", name)
            if ty.kind == T.TypeKind.VLA:
                ty, pre = self.vla_bind(ty, name.loc)
                items.extend(pre)
                var = self.new_lvar(name.text, ty, name.loc)
                size = self.vla_size_expr(ty, name.loc)
                items.append(N.VLAAlloc(name.loc, var, self.vla_block_slot(), size))
                self.block_vla_saves.append(var)   # later VLAs of this block stack below it
                if self.accept("="):
                    # Only empty braces are allowed (zero-filled), as GCC does.
                    if not (self.at("{") and self.peek().text == "}"):
                        self.error("variable-sized object may not be initialized", name)
                    self.next()
                    self.next()
                    items.append(N.VLAZero(name.loc, var, self.vla_size_expr(ty, name.loc)))
                continue
            if ty.kind == T.TypeKind.PTR:          # `int (*p)[n];` evaluates n here
                ty, pre = self.vla_bind(ty, name.loc)
                items.extend(pre)
            var = self.new_lvar(name.text, ty, name.loc)
            reg = (dattr.asm_name or "").lstrip("%")
            if reg in GP64_REGS and self.fn is not None:
                # Register-bound local: its value is the register's contents at function entry.
                self.fn.reg_vars.append((var, reg))
            al = max(attr.align, dattr.align)
            if al:
                var.align = max(al, ty.align)
            if self.accept("="):
                e = self.local_initializer(var)
                items.append(N.ExprStmt(name.loc, e))
            if not var.ty.is_complete or var.ty.size < 0:
                self.error(f"storage size of '{name.text}' isn't known", name)
        return items

    def block_func_composite(self, ty, name: str):
        """Block-scope function declaration `T *f(void);` that names a function
        visible from an outer scope: an incomplete array element type in the
        return pointer (`IA *`) takes the complete one (`A10 *`) from the prior
        declaration, as the composite type rule requires."""
        vs = self.find_var(name)
        prior = vs.var if vs is not None else None
        if not isinstance(prior, N.Function) or prior.ty.kind != T.TypeKind.FUNC:
            return ty
        new_r, old_r = ty.return_ty, prior.ty.return_ty
        if new_r is None or old_r is None or new_r.kind != T.TypeKind.PTR \
                or old_r.kind != T.TypeKind.PTR:
            return ty
        if new_r.base.kind == T.TypeKind.ARRAY and new_r.base.array_len < 0 \
                and old_r.base.kind == T.TypeKind.ARRAY and old_r.base.array_len >= 0:
            ty = ty.copy()
            ty.return_ty = old_r
        return ty

    def nested_function_definition(self, fn, ty) -> None:
        """GNU nested function: parsed as a file-level function.

        The enclosing function's per-function parser state is saved and
        restored around the body. References to enclosing locals are not
        supported yet (no static chain); the body is emitted as a plain
        function.
        """
        saved = (self.fn, self.locals, self.label_names, self.goto_names,
                 self.brk_label, self.cont_label, self.cur_switch)
        self.fn, self.locals = None, []
        self.brk_label = self.cont_label = self.cur_switch = None
        self.function_definition(fn, ty)
        (self.fn, self.locals, self.label_names, self.goto_names,
         self.brk_label, self.cont_label, self.cur_switch) = saved
