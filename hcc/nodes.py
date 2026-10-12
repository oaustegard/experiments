"""AST node dataclasses (STABLE, CONTRACT section 5).

Every node has `loc`. Expression nodes have `ty`, filled in by
hcc.sema.typing.add_type() right after construction. Codegen dispatches on
the class name: `expr_<ClassName>` / `stmt_<ClassName>`.

To add a node: define it here (keep the docstring one line), build it in the
parser, type it in sema/typing.py, generate it in the owning codegen file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hcc.tokens import Loc
from hcc.typesys import Member, Type


@dataclass(eq=False)
class Node:
    loc: Loc


# --------------------------------------------------------------------------
# Objects (variables and functions)

@dataclass(eq=False)
class Reloc:
    offset: int
    label: str
    addend: int


@dataclass(eq=False)
class GlobalData:
    """Serialized static initializer: raw bytes plus pointer relocations."""
    data: bytearray
    relocs: list[Reloc] = field(default_factory=list)


@dataclass(eq=False)
class Obj:
    name: str
    ty: Type
    is_static: bool = False        # internal linkage (file scope) or static local
    asm_name: str | None = None    # from `__asm__("sym")`; overrides symbol name
    loc: Loc | None = None

    @property
    def symbol(self) -> str:
        return self.asm_name or self.name


@dataclass(eq=False)
class Var(Obj):
    is_local: bool = False
    is_extern: bool = False
    is_definition: bool = False
    is_tentative: bool = False
    is_tls: bool = False
    is_param: bool = False
    align: int = 0
    offset: int = 0                         # %rbp-relative, set by codegen
    init_data: GlobalData | None = None
    is_weak: bool = False
    section: str | None = None
    visibility: str | None = None
    alias_of: str | None = None             # __attribute__((alias("sym")))


@dataclass(eq=False)
class Function(Obj):
    params: list[Var] = field(default_factory=list)
    locals: list[Var] = field(default_factory=list)
    body: "Block | None" = None
    is_inline: bool = False
    is_definition: bool = False
    is_noreturn: bool = False
    is_weak: bool = False
    va_area: Var | None = None
    alloca_bottom: Var | None = None
    section: str | None = None
    visibility: str | None = None
    alias_of: str | None = None
    align: int = 0                          # from __attribute__((aligned(N)))


@dataclass(eq=False)
class TranslationUnit:
    objs: list[Obj]


# --------------------------------------------------------------------------
# Expressions

@dataclass(eq=False)
class Expr(Node):
    ty: Type | None = field(default=None, kw_only=True)


@dataclass(eq=False)
class IntLit(Expr):
    value: int


@dataclass(eq=False)
class FloatLit(Expr):
    value: float


@dataclass(eq=False)
class StrLit(Expr):
    """String literal; `data` includes the terminating NUL (element-sized)."""
    data: bytes


@dataclass(eq=False)
class VarRef(Expr):
    var: Obj


@dataclass(eq=False)
class Unary(Expr):
    """op is one of '-', '~', '!'."""
    op: str
    operand: Expr


@dataclass(eq=False)
class AddrOf(Expr):
    operand: Expr


@dataclass(eq=False)
class Deref(Expr):
    operand: Expr


@dataclass(eq=False)
class Binary(Expr):
    """Arithmetic/bitwise/shift/comparison. Operands already converted."""
    op: str
    lhs: Expr
    rhs: Expr


@dataclass(eq=False)
class Logical(Expr):
    """op is '&&' or '||'."""
    op: str
    lhs: Expr
    rhs: Expr


@dataclass(eq=False)
class Assign(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(eq=False)
class Cond(Expr):
    cond: Expr
    then: Expr
    els: Expr


@dataclass(eq=False)
class Comma(Expr):
    lhs: Expr
    rhs: Expr
    lvalue: bool = False    # True for a local compound literal: init, then the object


@dataclass(eq=False)
class Cast(Expr):
    """Conversion of `operand` to `self.ty`."""
    operand: Expr


@dataclass(eq=False)
class Call(Expr):
    func: Expr
    args: list[Expr]
    func_ty: Type
    ret_buffer: Var | None = None     # temp for struct returns


@dataclass(eq=False)
class Member_(Expr):
    """`operand.member` (a->b is Member_(Deref(a), b)). Named Member_ to
    avoid clashing with typesys.Member; codegen method is expr_Member_."""
    operand: Expr
    member: Member


@dataclass(eq=False)
class StmtExpr(Expr):
    body: "Block"


@dataclass(eq=False)
class MemZero(Expr):
    var: Var


@dataclass(eq=False)
class BuiltinCall(Expr):
    """__builtin_* intrinsic; lowered in codegen/builtins.py."""
    name: str
    args: list[Expr]
    type_arg: Type | None = None


@dataclass(eq=False)
class LabelAddr(Expr):
    """GNU &&label. `func` is the enclosing function name (labels are per-function)."""
    label: str
    func: str = ""


@dataclass(eq=False)
class VLAPtr(Expr):
    var: Var


# --------------------------------------------------------------------------
# Statements

@dataclass(eq=False)
class Stmt(Node):
    pass


@dataclass(eq=False)
class Block(Stmt):
    items: list[Stmt]


@dataclass(eq=False)
class ExprStmt(Stmt):
    expr: Expr


@dataclass(eq=False)
class If(Stmt):
    cond: Expr
    then: Stmt
    els: Stmt | None


@dataclass(eq=False)
class For(Stmt):
    """`for` and `while` loops. init is a Stmt (ExprStmt or decl Block)."""
    init: Stmt | None
    cond: Expr | None
    inc: Expr | None
    body: Stmt | None
    brk_label: str
    cont_label: str


@dataclass(eq=False)
class DoWhile(Stmt):
    body: Stmt | None
    cond: Expr | None
    brk_label: str
    cont_label: str


@dataclass(eq=False)
class Case(Stmt):
    """`case lo ... hi:` (hi == lo normally). Default has lo=hi=None."""
    lo: int | None
    hi: int | None
    body: Stmt | None
    label: str


@dataclass(eq=False)
class Switch(Stmt):
    cond: Expr
    body: Stmt | None
    cases: list[Case]
    default: Case | None
    brk_label: str


@dataclass(eq=False)
class Break(Stmt):
    label: str


@dataclass(eq=False)
class Continue(Stmt):
    label: str


@dataclass(eq=False)
class Goto(Stmt):
    name: str


@dataclass(eq=False)
class GotoExpr(Stmt):
    target: Expr


@dataclass(eq=False)
class Label(Stmt):
    name: str
    body: Stmt | None


@dataclass(eq=False)
class Return(Stmt):
    expr: Expr | None


@dataclass(eq=False)
class Asm(Stmt):
    text: str
    outputs: list = field(default_factory=list)   # [(constraint, Expr)]
    inputs: list = field(default_factory=list)    # [(constraint, Expr)]
    clobbers: list = field(default_factory=list)
    is_volatile: bool = False


# --------------------------------------------------------------------------
# Initializers

@dataclass(eq=False)
class Initializer:
    """Tree mirroring an aggregate type. Leaves have `expr`."""
    ty: Type
    expr: Expr | None = None
    children: list["Initializer"] | None = None
    is_flexible: bool = False
    mem: Member | None = None       # for unions: the initialized member
