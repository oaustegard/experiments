"""Initializer parsing -> nodes.Initializer tree.

InitializerMixin provides:
    parse_initializer(ty) -> (Initializer, Type)   Type may be completed
                                                   (e.g. int a[] = {1,2})
    local_initializer(var) -> Expr                  parse + lower to assignments
    global_initializer(var)                         parse + serialize to var.init_data
Supports designators (.m / [i] / chains), brace elision, string literal
initialization of char/wchar arrays, flexible (incomplete) arrays.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.parser import complex as CPX
from hcc.parser import complex_const as CC
from hcc.parser.structs import find_member
from hcc.sema.constexpr import try_eval_const
from hcc.sema.initializers import lower_local_init, serialize_global_init
from hcc.tokens import TokenKind
from hcc import typesys as T


def new_initializer(ty, is_flexible: bool) -> N.Initializer:
    init = N.Initializer(ty)
    if ty.kind == T.TypeKind.ARRAY:
        if is_flexible and ty.array_len < 0:
            init.is_flexible = True
            init.children = []
        else:
            base = ty.base
            if ty.array_len > 0:
                init.children = N.ChildTable(ty.array_len,
                                             lambda: new_initializer(base, False))
            else:       # zero-length (may grow as a flexible member): plain list
                init.children = []
    elif ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION):
        init.children = [new_initializer(m.ty, False) for m in ty.members]
        # A trailing flexible array member (stored with length 0) grows on demand.
        if ty.members and ty.members[-1].ty.kind == T.TypeKind.ARRAY and \
                ty.members[-1].ty.array_len == 0:
            init.children[-1].is_flexible = True
    return init


class InitializerMixin:

    def parse_initializer(self, ty):
        init = new_initializer(ty, True)
        self.initializer2(init)
        self._check_flex_nesting(init, root=True)
        if ty.kind == T.TypeKind.ARRAY and ty.array_len < 0 and init.expr is not None:
            ty = init.expr.ty           # `int a[] = (int[4]){...};` takes the literal's size
            init.ty = ty
        elif ty.kind == T.TypeKind.ARRAY and ty.array_len < 0:
            ty = T.array_of(ty.base, len(init.children))
            init.ty = ty
        return init, ty

    def local_initializer(self, var) -> N.Expr:
        init, ty = self.parse_initializer(var.ty)
        var.ty = ty
        if ty.is_const and not ty.is_volatile and T.is_integer(ty) and init.expr is not None:
            # GCC folds `const int a = 0;` when it is read in a constant initializer.
            var.const_value = try_eval_const(init.expr)
        return lower_local_init(init, var, self.typed)

    def global_initializer(self, var) -> None:
        saved = self.in_static_init
        self.in_static_init = True
        try:
            init, ty = self.parse_initializer(var.ty)
        finally:
            self.in_static_init = saved
        var.ty = ty
        var.init_data = serialize_global_init(init, ty)

    # ---- helpers --------------------------------------------------------------
    def _check_flex_nesting(self, init, root: bool) -> None:
        """GCC: a trailing `T x[]` member may only be initialized when its struct
        is the object being initialized itself, not a sub-object of an aggregate."""
        if not root and init.ty.kind == T.TypeKind.STRUCT and init.ty.members and \
                init.ty.members[-1].open_array and isinstance(init.children, list) and \
                init.children:
            last = init.children[-1]
            if last.children or last.expr is not None:
                self.error("initialization of flexible array member in a nested context")
        for _, child in N.present_children(init.children):
            self._check_flex_nesting(child, root=False)

    def _at_end(self) -> bool:
        return self.at("}") or (self.at(",") and self.peek().text == "}")

    def _at_designator_after_comma(self) -> bool:
        """True when the next tokens are `,` followed by a designator."""
        return self.at(",") and self.peek().text in ("[", ".")

    def _consume_end(self) -> bool:
        if self.accept("}"):
            return True
        if self.at(",") and self.peek().text == "}":
            self.next()
            self.next()
            return True
        return False

    def _grow(self, init, i: int) -> bool:
        """Make child i exist for flexible arrays. False if out of range."""
        if init.is_flexible:
            while len(init.children) <= i:
                init.children.append(new_initializer(init.ty.base, False))
            return True
        return i < len(init.children)

    def _init_element(self, child) -> None:
        """Initialize one brace-list element. A brace-elided aggregate with no
        room (zero-length array, empty struct) consumes no tokens; its excess
        elements are then skipped (GCC: "excess elements in initializer")."""
        save = self.pos
        self.initializer2(child)
        if self.pos == save:
            self._skip_excess()

    def _skip_excess(self) -> None:
        if self.at("{"):
            self.next()
            self._skip_excess()
            while not self._consume_end():
                self.expect(",")
                self._skip_excess()
            return
        self.assign()

    # ---- dispatch -------------------------------------------------------------
    def initializer2(self, init) -> None:
        ty = init.ty
        while ty.kind == T.TypeKind.ARRAY and self.at("__extension__"):
            self.next()             # `char s[] = __extension__ "..."`
        if ty.kind == T.TypeKind.ARRAY and self.tok.kind == TokenKind.STRING and \
                T.is_integer(ty.base):
            self.string_initializer(init)
            return
        if ty.kind == T.TypeKind.ARRAY and self.at("{") is False and self.at("(") and \
                self.peek().kind == TokenKind.STRING and self.peek(2).text == ")":
            self.next()
            self.string_initializer(init)
            self.expect(")")
            return
        if ty.kind == T.TypeKind.ARRAY and ty.is_vector and not self.at("{"):
            save = self.pos
            expr = self.typed(self.assign())
            if expr.ty.is_vector:     # `vec v = other_vec;` copies the value
                init.expr = expr
                return
            self.pos = save
        if ty.kind == T.TypeKind.ARRAY:
            if self.at("(") and self.is_typename(self.peek()):
                # GNU: `int a[] = (int[4]){1, 2};` copies a compound literal.
                save = self.pos
                expr = self.typed(self.assign())
                if expr.ty.kind == T.TypeKind.ARRAY and not expr.ty.is_vector:
                    init.expr = expr
                    return
                self.pos = save
            if self.at("{"):
                self.array_initializer1(init)
            else:
                self.array_initializer2(init, 0)
            return
        if ty.kind in (T.TypeKind.STRUCT, T.TypeKind.UNION) and not ty.is_complex:
            if self.at("{"):
                self.struct_initializer1(init)
                return
            save = self.pos
            expr = self.typed(self.assign())
            if T.is_struct_like(expr.ty):
                init.expr = expr
                return
            self.pos = save
            self.struct_initializer2(init, 0)
            return
        if self.at("{"):            # scalar in braces: int x = {3};
            self.next()
            if self.at("}"):
                self.next()
                init.expr = N.IntLit(self.tok.loc, 0, ty=T.TY_INT)
                return
            self.initializer2(init)
            while not self._consume_end():   # `int *p = {0, NULL};`: excess elements are skipped
                self.expect(",")
                self._skip_excess()
            return
        init.expr = self.typed(self.assign())
        if init.ty.is_complex or init.expr.ty.is_complex:
            if self.fn is None:
                CC.set_global_init(self, init)
            else:
                init.expr = CPX.convert(self, init.expr, init.ty, init.expr.loc)

    def string_initializer(self, init) -> None:
        tok = self.next()
        elem = init.ty.base
        sz = elem.size
        data = tok.value
        n = len(data) // tok.ty.base.size
        units = [int.from_bytes(data[i * tok.ty.base.size:(i + 1) * tok.ty.base.size], "little")
                 for i in range(n)]
        if init.is_flexible:
            self._grow(init, n - 1)
        count = min(len(init.children), n)
        for i in range(count):
            v = units[i]
            if not elem.is_unsigned and v >= 1 << (sz * 8 - 1):
                v -= 1 << (sz * 8)
            init.children[i].expr = N.IntLit(tok.loc, v, ty=elem)

    def array_initializer1(self, init) -> None:
        self.expect("{")
        if self.tok.kind == TokenKind.STRING and T.is_integer(init.ty.base) and \
                self.peek().text in ("}", ","):
            # `char s[] = {"foo"};`: braces around a string literal initialize the array
            self.string_initializer(init)
            self._consume_end()
            return
        i = 0
        first = True
        while not self._consume_end():
            if not first:
                self.expect(",")
                if self._consume_end():
                    break
            first = False
            if self.at("["):
                i = self.array_designator(init)
                continue
            if self._grow(init, i):
                self._init_element(init.children[i])
            else:
                self._skip_excess()
            i += 1

    def array_initializer2(self, init, i: int) -> None:
        start = i
        while (init.is_flexible or i < len(init.children)) and not self._at_end():
            if i > start:
                if self._at_designator_after_comma():
                    return      # leave ',' for the enclosing initializer
                self.expect(",")
            if self.at("[") or self.at("."):
                return
            self._grow(init, i)
            self._init_element(init.children[i])
            i += 1

    def array_designator(self, init) -> int:
        """`[i] ...= x` inside an array initializer; returns next index."""
        self.expect("[")
        lo = self.const_expr()
        hi = lo
        if self.accept("..."):
            hi = self.const_expr()
        self.expect("]")
        save = self.pos
        for idx in range(lo, hi + 1):
            self.pos = save
            if not self._grow(init, idx):
                self.error("array index in initializer exceeds array bounds")
            self.designation(init.children[idx])
        return hi + 1

    def designation(self, init) -> None:
        """Continue a designator chain into `init`, then parse `= value`."""
        if init.expr is not None and (self.at("[") or self.at(".")):
            # A sub-object was already set as a whole (`T y[] = {x, [0].b=3}`):
            # the designator re-initializes it from zero (other members -> 0).
            fresh = new_initializer(init.ty, init.is_flexible)
            init.expr, init.children = None, fresh.children
        if self.at("["):
            if init.ty.kind != T.TypeKind.ARRAY:
                self.error("array index in non-array initializer")
            nxt = self.array_designator(init)
            # Following brace-elided values continue inside the designated
            # sub-aggregate (C11 6.7.9p17), e.g. `[0][1]=7, 8` sets x[0][2].
            if self.at(",") and (init.is_flexible or nxt < len(init.children)) and \
                    not self._at_designator_after_comma() and not self._at_end():
                self.next()
                self.array_initializer2(init, nxt)
            return
        if self.at("."):
            if not T.is_struct_like(init.ty):
                self.error("field name not in record or union initializer")
            self.member_designator(init)
            return
        self.accept("=")
        self.initializer2(init)

    def at_member_designator(self) -> bool:
        """`.name` or the obsolete GNU form `name:` starting a member designator."""
        if self.at("."):
            return True
        return self.tok.kind == TokenKind.IDENT and self.peek().text == ":"

    def member_designator(self, init) -> int:
        """`.name ...= x` (or GNU `name: x`) inside a struct/union initializer;
        returns next index."""
        if self.at("."):
            self.expect(".")
        name = self.expect_ident()
        if self.at(":"):
            self.expect(":")
        path = find_member(init.ty, name.text)
        if path is None:
            self.error(f"unknown field '{name.text}' specified in initializer", name)
        cur = init
        for m in path[:-1]:
            if cur.ty.kind == T.TypeKind.UNION:
                cur.mem = m
            cur = cur.children[m.idx]
        m = path[-1]
        if cur.ty.kind == T.TypeKind.UNION:
            cur.mem = m
        self.designation(cur.children[m.idx])
        return path[0].idx + 1

    def struct_initializer1(self, init) -> None:
        self.expect("{")
        i = 0
        first = True
        members = init.ty.members
        is_union = init.ty.kind == T.TypeKind.UNION
        while not self._consume_end():
            if not first:
                self.expect(",")
                if self._consume_end():
                    break
            first = False
            if self.at_member_designator():
                i = self.member_designator(init)
                continue
            i = self._next_named(members, i)
            if i < len(members) and not (is_union and i > 0):
                if is_union:
                    init.mem = members[i]
                self._init_element(init.children[i])
                i += 1
            else:
                self._skip_excess()

    def struct_initializer2(self, init, i: int) -> None:
        members = init.ty.members
        is_union = init.ty.kind == T.TypeKind.UNION
        first = True
        while True:
            i = self._next_named(members, i)
            if i >= len(members) or self._at_end() or (is_union and not first):
                return
            if not first:
                if self._at_designator_after_comma():
                    return      # leave ',' for the enclosing initializer
                self.expect(",")
            if self.at_member_designator() or self.at("["):
                return
            first = False
            if is_union:
                init.mem = members[i]
            self._init_element(init.children[i])
            i += 1

    @staticmethod
    def _next_named(members, i: int) -> int:
        """Skip unnamed bitfields (they take no initializer)."""
        while i < len(members) and members[i].is_bitfield and members[i].name is None:
            i += 1
        return i
