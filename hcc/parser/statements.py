"""Statements.

StatementMixin provides:
    compound_stmt_body(loc) -> Block   after '{' (consumes '}'), new scope
    stmt() -> Stmt
One `_stmt_<keyword>` method per statement keyword; see STMT_KEYWORDS.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.parser import complex as CPX
from hcc.sema.constexpr import eval_const
from hcc.sema.conversions import new_cast
from hcc.tokens import TokenKind
from hcc.typesys import TypeKind

STMT_KEYWORDS = frozenset(("return", "if", "switch", "case", "default", "for", "while", "do",
                           "goto", "break", "continue", "asm"))


class StatementMixin:

    def compound_stmt_body(self, loc, same_scope=False) -> N.Block:
        """`same_scope`: a function body shares the scope that already holds its
        parameters (C11 6.2.1p4), so it neither opens nor closes one."""
        items = []
        if not same_scope:
            self.enter_scope()
        outer_saves = getattr(self, "block_vla_saves", None)
        self.block_vla_saves = []
        while not self.accept("}"):
            if self.at_eof():
                self.error("expected '}' at end of input")
            if self.at("_Static_assert"):
                self.static_assert()
                continue
            if self.at("__extension__"):
                self.next()
                continue
            if self.at_std_attribute():
                self.skip_std_attributes()
                continue
            if self.is_typename() and not (self.tok.kind == TokenKind.IDENT and self.peek().text == ":"):
                items.extend(self.local_declaration())
                continue
            items.append(self.stmt())
        if not same_scope:
            self.leave_scope()
        saves = self.block_vla_saves
        self.block_vla_saves = outer_saves
        if saves:
            if self.fn is not None and self.fn.alloca_bottom is None:
                self.fn.alloca_bottom = self.new_temp(T.TY_ULONG)   # alloca low-water mark
            items = [N.VLAEnter(loc, saves[0])] + items + [N.VLAExit(loc, saves[0])]
        return N.Block(loc, items)

    def stmt(self) -> N.Stmt:
        if self.at_std_attribute():
            self.skip_std_attributes()
        t = self.tok
        if t.kind == TokenKind.KEYWORD and t.text in STMT_KEYWORDS:
            return getattr(self, "_stmt_" + t.text)(self.next())
        if t.kind == TokenKind.IDENT and self.peek().text == ":" and self.peek().kind == TokenKind.PUNCT:
            self.next()
            self.next()
            self.parse_attributes()
            self.label_names.add(t.text)
            body = None if self.at("}") else self.stmt()
            return N.Label(t.loc, t.text, body)
        if t.kind == TokenKind.PUNCT and t.text == "{":
            self.next()
            return self.compound_stmt_body(t.loc)
        if t.kind == TokenKind.PUNCT and t.text == ";":
            self.next()
            return N.Block(t.loc, [])
        if self.is_typename() and t.text != "__extension__":
            return N.Block(t.loc, self.local_declaration())     # declaration after label
        node = self.expr()
        self.expect(";")
        return N.ExprStmt(t.loc, node)

    def _stmt_return(self, t):
        if self.accept(";"):
            return N.Return(t.loc, None)
        e = self.typed(self.expr())
        self.expect(";")
        rty = self.fn.ty.return_ty
        if rty.kind == TypeKind.VOID:
            return N.Return(t.loc, e)
        if rty.is_complex or e.ty.is_complex:
            CPX.check_abi(self, rty, t.loc)
            e = CPX.convert(self, e, rty, t.loc)
            return N.Return(t.loc, e)
        if rty.kind not in (TypeKind.STRUCT, TypeKind.UNION):
            e = new_cast(e, rty)
        return N.Return(t.loc, e)

    def _stmt_if(self, t):
        self.expect("(")
        cond = CPX.truth(self, self.typed(self.expr()))
        self.expect(")")
        then = self.stmt()
        els = self.stmt() if self.accept("else") else None
        return N.If(t.loc, cond, then, els)

    def _stmt_switch(self, t):
        self.expect("(")
        cond = CPX.truth(self, self.typed(self.expr()))
        self.expect(")")
        from hcc.sema.conversions import int_promote
        cond = new_cast(cond, int_promote(cond.ty))
        sw = N.Switch(t.loc, cond, None, [], None, self.new_label())
        saved = (self.cur_switch, self.brk_label)
        self.cur_switch, self.brk_label = sw, sw.brk_label
        sw.body = self.stmt()
        self.cur_switch, self.brk_label = saved
        return sw

    def _stmt_case(self, t):
        if self.cur_switch is None:
            self.error("case label not within a switch statement", t)
        lo = eval_const(self.conditional())
        hi = lo
        if self.accept("..."):
            hi = eval_const(self.conditional())
        self.expect(":")
        c = N.Case(t.loc, lo, hi, None, self.new_label())
        self.cur_switch.cases.append(c)
        c.body = None if self.at("}") else self.stmt()
        return c

    def _stmt_default(self, t):
        if self.cur_switch is None:
            self.error("'default' label not within a switch statement", t)
        self.expect(":")
        c = N.Case(t.loc, None, None, None, self.new_label())
        self.cur_switch.default = c
        c.body = None if self.at("}") else self.stmt()
        return c

    def _loop_body(self, brk, cont):
        saved = (self.brk_label, self.cont_label)
        self.brk_label, self.cont_label = brk, cont
        body = self.stmt()
        self.brk_label, self.cont_label = saved
        return body

    def _stmt_for(self, t):
        self.expect("(")
        self.skip_std_attributes()  # `for ([[]] int i = 0; ...)`
        self.enter_scope()
        init = None
        if self.is_typename():
            init = N.Block(t.loc, self.local_declaration())
        elif self.at("_Static_assert"):
            self.static_assert()  # consumes its own ';'
        elif not self.accept(";"):
            init = N.ExprStmt(t.loc, self.expr())
            self.expect(";")
        cond = None if self.at(";") else CPX.truth(self, self.typed(self.expr()))
        self.expect(";")
        inc = None if self.at(")") else self.typed(self.expr())
        self.expect(")")
        brk, cont = self.new_label(), self.new_label()
        body = self._loop_body(brk, cont)
        self.leave_scope()
        return N.For(t.loc, init, cond, inc, body, brk, cont)

    def _stmt_while(self, t):
        self.expect("(")
        cond = CPX.truth(self, self.typed(self.expr()))
        self.expect(")")
        brk, cont = self.new_label(), self.new_label()
        body = self._loop_body(brk, cont)
        return N.For(t.loc, None, cond, None, body, brk, cont)

    def _stmt_do(self, t):
        brk, cont = self.new_label(), self.new_label()
        body = self._loop_body(brk, cont)
        self.expect("while")
        self.expect("(")
        cond = CPX.truth(self, self.typed(self.expr()))
        self.expect(")")
        self.expect(";")
        return N.DoWhile(t.loc, body, cond, brk, cont)

    def _stmt_goto(self, t):
        if self.accept("*"):
            target = self.typed(self.expr())
            self.expect(";")
            return N.GotoExpr(t.loc, target)
        name = self.expect_ident()
        self.goto_names.append(name)
        self.expect(";")
        return N.Goto(t.loc, name.text)

    def _stmt_break(self, t):
        if self.brk_label is None:
            self.error("break statement not within loop or switch", t)
        self.expect(";")
        return N.Break(t.loc, self.brk_label)

    def _stmt_continue(self, t):
        if self.cont_label is None:
            self.error("continue statement not within a loop", t)
        self.expect(";")
        return N.Continue(t.loc, self.cont_label)

    def _stmt_asm(self, t):
        node = self.asm_stmt(t)
        self.expect(";")
        return node

    def asm_stmt(self, t) -> N.Asm:
        is_volatile = False
        while self.at("volatile") or self.at("inline") or self.at("goto"):
            is_volatile |= self.next().text == "volatile"
        self.expect("(")
        text = ""
        while self.tok.kind == TokenKind.STRING:
            text += self.next().value.rstrip(b"\0").decode("utf-8", "replace")
        node = N.Asm(t.loc, text, is_volatile=is_volatile)
        sections = [node.outputs, node.inputs]
        for sec in sections:
            if not self.accept(":"):
                break
            while self.tok.kind == TokenKind.STRING or self.at("["):
                sym = None
                if self.accept("["):
                    sym = self.expect_ident().text
                    self.expect("]")
                cons = self.next().value.rstrip(b"\0").decode()
                self.expect("(")
                e = self.typed(self.expr())
                self.expect(")")
                sec.append((cons, e))
                if sym is not None:
                    node.names[sym] = len(node.outputs) + len(node.inputs) - 1
                if not self.accept(","):
                    break
        else:
            if self.accept(":"):
                while self.tok.kind == TokenKind.STRING:
                    node.clobbers.append(self.next().value.rstrip(b"\0").decode())
                    if not self.accept(","):
                        break
                if self.accept(":"):            # asm goto labels
                    while self.tok.kind == TokenKind.IDENT:
                        node.goto_labels.append(self.next().text)
                        if not self.accept(","):
                            break
        self.expect(")")
        return node

    def static_assert(self) -> None:
        t = self.expect("_Static_assert")
        self.expect("(")
        v = self.const_expr()
        msg = ""
        if self.accept(","):
            msg = self.next().value.rstrip(b"\0").decode("utf-8", "replace")
        self.expect(")")
        self.expect(";")
        if not v:
            self.error(f"static assertion failed: \"{msg}\"", t)
