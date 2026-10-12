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
    """Address constant stored at `offset`: `label + addend` (or `label - minus + addend`
    when `minus` is set, a difference of two labels resolved by the assembler), `size` bytes."""
    offset: int
    label: str
    addend: int
    size: int = 8
    minus: str | None = None
    # further assembler-resolved address terms: (sign, label) pairs, e.g. the
    # `+ (&&c - &&d)` of `(&&a - &&b) + (&&c - &&d)`; emitted after `minus`
    extra: tuple = ()


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
    is_register: bool = False               # `register` storage class: address may not be taken
    align: int = 0
    offset: int = 0                         # %rbp-relative, set by codegen
    init_data: GlobalData | None = None
    is_weak: bool = False
    is_nocommon: bool = False               # __attribute__((nocommon)) on a tentative definition
    section: str | None = None
    visibility: str | None = None
    alias_of: str | None = None             # __attribute__((alias("sym")))
    is_compound_literal: bool = False       # anon static object of a compound literal in a static init
    const_value: int | None = None          # block-scope const integer with a constant initializer (GNU folding)
    scope: tuple = ()                       # enclosing block ids (outermost first); sibling blocks share frame slots


@dataclass(eq=False)
class Function(Obj):
    params: list[Var] = field(default_factory=list)
    locals: list[Var] = field(default_factory=list)
    body: "Block | None" = None
    is_inline: bool = False
    is_definition: bool = False
    is_noreturn: bool = False
    is_weak: bool = False
    is_weakref_decl: bool = False           # weakref("t") declaration with no body (a later body is an error)
    fn_attrs: set = field(default_factory=set)  # other recorded attrs (leaf, returns_nonnull)
    va_area: Var | None = None
    va_tmp: Var | None = None
    alloca_bottom: Var | None = None
    section: str | None = None
    visibility: str | None = None
    alias_of: str | None = None
    ifunc_of: str | None = None             # __attribute__((ifunc("resolver")))
    init_kind: str | None = None            # "init" (constructor) / "fini" (destructor)
    init_prio: int | None = None            # constructor(N) / destructor(N) priority (None: default)
    align: int = 0                          # from __attribute__((aligned(N)))
    outer: "Function | None" = None         # GNU nested function: enclosing function
    chain_var: Var | None = None            # nested fn: slot holding the static link
    reg_vars: list = field(default_factory=list)  # (Var, reg) for `register T x __asm__("reg")`
    labels: set = field(default_factory=set)  # names of labels defined in this body


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
    # Exact IEEE encoding for the literal's own width (float/double), when the
    # Python float would lose it (e.g. signaling NaNs are quieted by a round trip).
    # For long double/_Float128 literals: the exact 80-bit x87 encoding (ldouble_lit.py).
    bits: int | None = field(default=None, kw_only=True)


@dataclass(eq=False)
class StrLit(Expr):
    """String literal; `data` includes the terminating NUL (element-sized)."""
    data: bytes


@dataclass(eq=False)
class VarRef(Expr):
    var: Obj


@dataclass(eq=False)
class Unary(Expr):
    """op is one of '-', '~', '!'.

    result_buf: for vector operands, the anonymous local that holds the result
    (set by the parser; codegen copies the operand into it and inverts it).
    """
    op: str
    operand: Expr
    result_buf: Obj | None = field(default=None, kw_only=True)


@dataclass(eq=False)
class AddrOf(Expr):
    operand: Expr


@dataclass(eq=False)
class Deref(Expr):
    operand: Expr


@dataclass(eq=False)
class Binary(Expr):
    """Arithmetic/bitwise/shift/comparison. Operands already converted.

    result_buf: for element-wise vector operands, the anonymous local that
    holds the result (set by the parser; codegen computes into it).
    """
    op: str
    lhs: Expr
    rhs: Expr
    result_buf: Obj | None = field(default=None, kw_only=True)


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
    # __builtin_call_with_static_chain: `static_chain` stores the chain value into
    # `chain_slot` (a frame temp) before the call; codegen loads it into %r10.
    static_chain: Expr | None = None
    chain_slot: Var | None = None
    # Struct with a variably modified member returned by value: `ret_buffer` is a pointer
    # slot filled by `dyn_alloc` (a VLAAlloc codegen hoists before the enclosing statement).
    dyn_alloc: object = None


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
    """__builtin_* intrinsic; lowered by @builtin_codegen handlers (hcc/builtins/).

    result_buf: for vector-valued builtins (__builtin_shuffle), the anonymous
    local that holds the result (set by the parser).
    """
    name: str
    args: list[Expr]
    type_arg: Type | None = None
    result_buf: Obj | None = field(default=None, kw_only=True)


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
class VLAEnter(Stmt):
    """Block with VLA declarations: save %rsp into the hidden `saved` slot."""
    saved: Var


@dataclass(eq=False)
class VLAExit(Stmt):
    """Block end: restore %rsp from the hidden `saved` slot (frees its VLAs)."""
    saved: Var


@dataclass(eq=False)
class VLAAlloc(Stmt):
    """VLA declaration: rsp = saved - size (16-aligned); `var`'s slot gets the address."""
    var: Var
    saved: Var
    size: Expr


@dataclass(eq=False)
class VLAZero(Stmt):
    """`T vla[n] = {};` on a VLA: zero-fill `size` bytes at the address held in `var`."""
    var: Var
    size: Expr


@dataclass(eq=False)
class CleanupDecl(Stmt):
    """Follows the declaration of a local with __attribute__((cleanup(f))): `call` is
    f(&var). Codegen runs it when the enclosing block exits (codegen/cleanup.py)."""
    var: "Var"
    call: "Expr"


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
    size: Expr | None = None      # run-time byte size when returning a struct with a VLA member


@dataclass(eq=False)
class Asm(Stmt):
    text: str
    outputs: list = field(default_factory=list)   # [(constraint, Expr)]
    inputs: list = field(default_factory=list)    # [(constraint, Expr)]
    clobbers: list = field(default_factory=list)
    is_volatile: bool = False
    goto_labels: list = field(default_factory=list)   # asm goto: [label name]
    names: dict = field(default_factory=dict)   # [sym] operand name -> GCC operand index


# --------------------------------------------------------------------------
# Initializers

@dataclass(eq=False)
class Initializer:
    """Tree mirroring an aggregate type. Leaves have `expr`."""
    ty: Type
    expr: Expr | None = None
    children: "list[Initializer] | ChildTable | None" = None
    is_flexible: bool = False
    mem: Member | None = None       # for unions: the initialized member


class ChildTable:
    """Children of an array initializer, materialized only on first access.

    `len()` is the array length, but elements are created lazily so that huge
    arrays (e.g. `char b[1 << 29]`) cost nothing unless actually initialized.
    Iterate with `items()` (present elements only), never with `for c in ...`.
    """
    __slots__ = ("n", "_make", "_items")

    def __init__(self, n: int, make) -> None:
        self.n = n
        self._make = make
        self._items: dict[int, Initializer] = {}

    def __len__(self) -> int:
        return self.n

    def __bool__(self) -> bool:
        return self.n > 0

    def __getitem__(self, i: int) -> Initializer:
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError(i)
        child = self._items.get(i)
        if child is None:
            child = self._items[i] = self._make()
        return child

    def __iter__(self):
        raise TypeError("ChildTable: use items() to visit materialized children")

    def items(self) -> list[tuple[int, Initializer]]:
        """(index, child) for every materialized child, in index order."""
        return sorted(self._items.items(), key=lambda kv: kv[0])


def present_children(children) -> list[tuple[int, Initializer]]:
    """(index, child) pairs that exist in either a list or a ChildTable."""
    if children is None:
        return []
    if isinstance(children, ChildTable):
        return children.items()
    return list(enumerate(children))
