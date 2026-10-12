"""Extended inline asm: operands, constraints and %N substitution.

lower_extended(cg, node) lowers `asm(tpl : outs : ins : clobbers)`:
  1. addresses of register-class outputs are pushed,
  2. input values (and "+" values, loaded through the saved address) and
     memory-operand addresses are pushed, then popped into their registers,
  3. the template is emitted with %N / %kN / %wN / %bN / %qN / %cN / %% expanded,
  4. output registers are pushed and stored back through the saved addresses.
Constraints: GPR letters (r q g R l x ...), fixed a c d S D, memory (m o V),
immediate (i n ...) for constant operands, matching digits, and =, +, &.
Float and aggregate operands are rejected.
"""
from __future__ import annotations

from hcc import diag
from hcc.sema import constexpr
from hcc.typesys import TypeKind

_POOL = ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"]
_FIXED = {"a": "rax", "c": "rcx", "d": "rdx", "S": "rsi", "D": "rdi"}
_LEGACY = {"rax": "ax", "rcx": "cx", "rdx": "dx", "rsi": "si", "rdi": "di"}
_LEGACY8 = {"rax": "al", "rcx": "cl", "rdx": "dl", "rsi": "sil", "rdi": "dil"}
_REG_LETTERS = set("rqgRlxyQabcdSD")
_MEM_LETTERS = set("moV")
_IMM_LETTERS = set("inIJKLMNeEFZs")
_BAD_KINDS = (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LDOUBLE, TypeKind.FLOAT128,
              TypeKind.STRUCT, TypeKind.UNION, TypeKind.ARRAY, TypeKind.FUNC)


def _reg_name(reg: str, size: int) -> str:
    """Name of a 64-bit GPR (no '%') at the given byte size."""
    if reg in _LEGACY8:
        return {1: _LEGACY8[reg], 2: _LEGACY[reg], 4: "e" + _LEGACY[reg], 8: reg}[size]
    return {1: reg + "b", 2: reg + "w", 4: reg + "d", 8: reg}[size]


def _norm_size(size: int) -> int:
    return size if size in (1, 2, 4, 8) else 8


class _Op:
    def __init__(self, cons: str, expr, is_out: bool):
        self.cons = cons
        self.expr = expr
        self.is_out = is_out
        self.rw = "+" in cons
        self.kind = "reg"        # reg | mem | imm
        self.fixed = None        # fixed register for a/c/d/S/D
        self.match = None        # operand index for matching constraints
        self.imm = None
        self.reg = None          # assigned register (no '%')
        self.size = _norm_size(expr.ty.size)


def _classify(op: _Op) -> None:
    c = "".join(ch for ch in op.cons if ch not in "=+&%")
    where = op.expr
    if c.isdigit():
        op.match = int(c)
        return
    letters = set(c)
    if letters & _REG_LETTERS:
        op.kind = "reg"
        fixed = [_FIXED[ch] for ch in c if ch in _FIXED]
        op.fixed = fixed[0] if fixed else None
    elif letters & _MEM_LETTERS:
        op.kind = "mem"
    elif letters & _IMM_LETTERS:
        op.kind = "imm"
        op.imm = constexpr.try_eval_const(op.expr)
        if op.imm is None:
            diag.error(where, "codegen: asm immediate operand is not constant", "codegen")
    else:
        diag.error(where, f"codegen: unsupported asm constraint '{op.cons}'", "codegen")
    if op.kind != "imm" and op.expr.ty.kind in _BAD_KINDS and op.kind == "reg":
        diag.error(where, "codegen: float/aggregate asm operands are not supported", "codegen")


def _assign_regs(ops, clobbers) -> None:
    used = set()
    for op in ops:
        if op.fixed:
            used.add(op.fixed)
    used |= {c for c in clobbers if c in _POOL}
    free = [r for r in _POOL if r not in used]
    for op in ops:
        if op.kind == "imm" or op.match is not None:
            continue
        if op.fixed:
            op.reg = op.fixed
        else:
            if not free:
                diag.error(op.expr, "codegen: too many asm operands for available registers", "codegen")
            op.reg = free.pop(0)
    for op in ops:
        if op.match is not None:
            if op.match >= len(ops):
                diag.error(op.expr, "codegen: asm matching constraint refers to a missing operand", "codegen")
            op.reg = ops[op.match].reg
            op.kind = ops[op.match].kind


def _render(op: _Op, mod: str | None) -> str:
    if op.kind == "imm":
        return str(op.imm) if mod in ("c", "P") else f"${op.imm}"
    if op.kind == "mem":
        return f"(%{op.reg})"
    size = {"k": 4, "w": 2, "b": 1, "q": 8}.get(mod, op.size)
    return "%" + _reg_name(op.reg, size)


def _subst(tpl: str, ops, loc) -> str:
    out = []
    i = 0
    n = len(tpl)
    while i < n:
        ch = tpl[i]
        if ch != "%":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i < n and tpl[i] == "%":
            out.append("%")
            i += 1
            continue
        mod = None
        if i < n and tpl[i] in "kwbqcP":
            mod = tpl[i]
            i += 1
        j = i
        while j < n and tpl[j].isdigit():
            j += 1
        if j == i:
            diag.error(loc, "codegen: unsupported asm operand reference", "codegen")
        idx = int(tpl[i:j])
        i = j
        if idx >= len(ops):
            diag.error(loc, "codegen: asm operand number out of range", "codegen")
        out.append(_render(ops[idx], mod))
    return "".join(out)


def lower_extended(cg, node) -> None:
    ops = [_Op(c, e, True) for c, e in node.outputs] + \
          [_Op(c, e, False) for c, e in node.inputs]
    clobbers = {c.strip().lstrip("%") for c in node.clobbers}
    for op in ops:
        _classify(op)
    _assign_regs(ops, clobbers)

    outs = [op for op in ops if op.is_out and op.kind == "reg"]
    m = len(outs)
    # Phase 1: addresses of register outputs.
    for op in outs:
        cg.gen_addr(op.expr)
        cg.push()
    # Phase 2: input values, "+" values and memory addresses.
    pushes = []
    for op in ops:
        if op.kind == "imm" or (op.is_out and op.kind == "reg" and not op.rw):
            continue
        if op.kind == "mem":
            cg.gen_addr(op.expr)
        elif op.is_out and op.rw:
            k = outs.index(op)
            cg.emit(f"mov {8 * (len(pushes) + m - 1 - k)}(%rsp), %rax")
            cg.load(op.expr.ty)
        else:
            cg.gen_expr(op.expr)
        cg.push()
        pushes.append(op)
    for op in reversed(pushes):
        cg.pop("%" + op.reg)
    # Phase 3: the template.
    text = _subst(node.text, ops, node)
    for line in text.split("\n"):
        line = line.strip()
        if line:
            cg.emit(line)
    # Phase 4: store output registers back through the saved addresses.
    if m:
        for op in outs:
            cg.emit(f"push %{op.reg}")
        for k, op in enumerate(outs):
            cg.emit(f"mov {8 * (m + m - 1 - k)}(%rsp), %rdi")
            cg.emit(f"mov {8 * (m - 1 - k)}(%rsp), %rax")
            cg.emit(f"mov %{_reg_name('rax', op.size)}, (%rdi)")
        cg.emit(f"add ${16 * m}, %rsp")
    cg.depth -= m       # phase-1 address slots are released here
