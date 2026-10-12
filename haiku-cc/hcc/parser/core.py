"""ParserBase: token cursor, scopes, labels and current-function context.

Every other parser mixin builds on these helpers:
    self.tok            current token          self.peek(n)   lookahead
    self.next()         consume and return     self.at(s)     current is punct/keyword s?
    self.accept(s)      consume if at(s)       self.expect(s) consume or error
    self.typed(node)    run sema.typing.add_type on a freshly built Expr
    self.new_lvar / new_gvar / new_label / enter_scope / leave_scope
"""
from __future__ import annotations

from hcc import diag
from hcc.nodes import Expr, Function, StrLit, Var
from hcc.parser.scope import Scope, VarScope
from hcc.sema.typing import add_type
from hcc.tokens import Token, TokenKind
from hcc.typesys import Type


class ParserBase:

    def __init__(self, toks: list[Token], opts=None):
        self.toks = toks
        self.pos = 0
        self.opts = opts
        self.scopes: list[Scope] = [Scope()]
        self.objs: list = []                 # TranslationUnit order
        self.fn: Function | None = None      # function being parsed
        self.locals: list[Var] = []
        self.label_count = 0
        self.anon_count = 0
        self.brk_label: str | None = None
        self.cont_label: str | None = None
        self.cur_switch = None
        self.goto_names: list[Token] = []
        self.label_names: set[str] = set()
        self.in_static_init = False          # inside a static-storage initializer
        self.scope_stack: list[int] = [0]    # ids of enclosing block scopes (0 = file scope)
        self.scope_seq = 0
        self.stmt_expr_base: int | None = None  # scope depth of the outermost ({ }) being parsed

    # ---- token cursor -------------------------------------------------------
    @property
    def tok(self) -> Token:
        return self.toks[self.pos]

    def peek(self, n: int = 1) -> Token:
        i = min(self.pos + n, len(self.toks) - 1)
        return self.toks[i]

    def next(self) -> Token:
        t = self.toks[self.pos]
        if t.kind != TokenKind.EOF:
            self.pos += 1
        return t

    def at(self, s: str) -> bool:
        t = self.toks[self.pos]
        return t.text == s and t.kind in (TokenKind.PUNCT, TokenKind.KEYWORD)

    def at_eof(self) -> bool:
        return self.toks[self.pos].kind == TokenKind.EOF

    def accept(self, s: str) -> bool:
        if self.at(s):
            self.pos += 1
            return True
        return False

    def expect(self, s: str) -> Token:
        if not self.at(s):
            t = self.tok
            got = t.text or "end of file"
            diag.error(t, f"expected '{s}' before '{got}'")
        return self.next()

    def expect_ident(self) -> Token:
        t = self.tok
        if t.kind != TokenKind.IDENT:
            diag.error(t, f"expected identifier before '{t.text or 'end of file'}'")
        return self.next()

    def error(self, msg: str, where=None):
        diag.error(where if where is not None else self.tok, msg)

    # ---- AST helpers --------------------------------------------------------
    def typed(self, node: Expr) -> Expr:
        add_type(node)
        return node

    def new_label(self) -> str:
        self.label_count += 1
        return f".L..{self.label_count}"

    # ---- scopes -------------------------------------------------------------
    def enter_scope(self) -> None:
        self.push_scope(Scope())

    def push_scope(self, scope: Scope) -> None:
        self.scopes.append(scope)
        self.scope_seq += 1
        self.scope_stack.append(self.scope_seq)

    def leave_scope(self) -> None:
        self.scopes.pop()
        self.scope_stack.pop()

    def var_scope_path(self) -> tuple:
        """Block-id path a new local belongs to.  Locals of a statement expression
        are pinned to the scope enclosing the outermost ({ }), because its value
        may be used after the block's own locals would be dead."""
        n = len(self.scope_stack) if self.stmt_expr_base is None else self.stmt_expr_base
        return tuple(self.scope_stack[:n])

    def find_var(self, name: str) -> VarScope | None:
        for sc in reversed(self.scopes):
            vs = sc.vars.get(name)
            if vs is not None:
                return vs
        return None

    def find_tag(self, name: str) -> Type | None:
        for sc in reversed(self.scopes):
            t = sc.tags.get(name)
            if t is not None:
                return t
        return None

    def push_scope_entry(self, name: str) -> VarScope:
        vs = VarScope()
        self.scopes[-1].vars[name] = vs
        return vs

    def find_typedef(self, tok: Token) -> Type | None:
        if tok.kind != TokenKind.IDENT:
            return None
        vs = self.find_var(tok.text)
        return vs.type_def if vs else None

    # ---- objects ------------------------------------------------------------
    def new_lvar(self, name: str, ty: Type, loc=None) -> Var:
        v = Var(name, ty, loc=loc, is_local=True, is_definition=True, align=ty.align,
                scope=self.var_scope_path())
        self.push_scope_entry(name).var = v
        self.locals.append(v)
        return v

    def new_temp(self, ty: Type, loc=None) -> Var:
        """Anonymous local (not visible by name)."""
        v = Var("", ty, loc=loc, is_local=True, is_definition=True, align=ty.align)
        self.locals.append(v)
        return v

    def new_gvar(self, name: str, ty: Type, loc=None) -> Var:
        v = Var(name, ty, loc=loc, is_definition=True, align=ty.align)
        self.scopes[0].vars.setdefault(name, VarScope()).var = v
        if len(self.scopes) > 1:
            self.push_scope_entry(name).var = v
        self.objs.append(v)
        return v

    def new_anon_gvar(self, ty: Type, loc=None) -> Var:
        self.anon_count += 1
        v = Var(f".L..anon.{self.anon_count}", ty, loc=loc, is_static=True,
                is_definition=True, align=ty.align)
        self.objs.append(v)
        return v

    def string_literal_node(self, tok: Token) -> StrLit:
        return self.typed(StrLit(tok.loc, tok.value, ty=tok.ty))
