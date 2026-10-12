"""Extended inline asm: operands, constraints and %N substitution.

lower_extended(cg, node) lowers `asm(tpl : outs : ins : clobbers)`:
  1. addresses of register-class outputs are pushed,
  2. input values (and "+" values, loaded through the saved address) and
     memory-operand addresses are pushed, then popped into their registers,
  3. the template is emitted with %N / %kN / %wN / %bN / %qN / %cN / %% expanded,
  4. output registers are pushed and stored back through the saved addresses.
Constraints: GPR letters (r q g R l x ...), fixed a c d S D, memory (m o V),
immediate (i n ...) for constant operands, matching digits, and =, +, &.
x87 operands (t/u/f; float, double or long double): inputs are converted to long
double, staged as 16-byte slots and loaded onto the FPU before the template; outputs
are stored with fstps/fstpl/fstpt according to their C type.
Float/double operands with GPR constraints travel as their raw bits (movd/movq
from %xmm0). Struct/union operands of 1, 2, 4 or 8 bytes travel as their bytes;
other aggregates are rejected.
"""
from __future__ import annotations

from hcc import diag
from hcc import nodes as N
from hcc.sema import constexpr
from hcc.typesys import TY_LDOUBLE, TypeKind
from hcc.sema.conversions import new_cast

_POOL = ["rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"]
# Callee-saved GPRs: used only when the caller-saved pool runs out, and then
# preserved around the template (see lower_extended).
_CALLEE_SAVED = ["rbx", "r12", "r13", "r14", "r15"]
_FIXED = {"a": "rax", "c": "rcx", "d": "rdx", "S": "rsi", "D": "rdi"}
_LEGACY = {"rax": "ax", "rcx": "cx", "rdx": "dx", "rsi": "si", "rdi": "di", "rbx": "bx"}
_LEGACY8 = {"rax": "al", "rcx": "cl", "rdx": "dl", "rsi": "sil", "rdi": "dil", "rbx": "bl"}
_REG_LETTERS = set("rqgRlxyQabcdSDXp")  # X (any) and p (address) use a GPR
_MEM_LETTERS = set("moV")
_X87_LETTERS = set("tuf")
_X87_STORE = {TypeKind.FLOAT: "fstps", TypeKind.DOUBLE: "fstpl", TypeKind.LDOUBLE: "fstpt"}
_IMM_LETTERS = set("inIJKLMNeEFZs")
_BAD_KINDS = (TypeKind.FLOAT16, TypeKind.LDOUBLE, TypeKind.FLOAT128)
_AGG_KINDS = (TypeKind.STRUCT, TypeKind.UNION)
# Arrays and functions decay to their address when used as register operands.
_DECAY_KINDS = (TypeKind.ARRAY, TypeKind.FUNC)


def _reg_name(reg: str, size: int) -> str:
    """Name of a 64-bit GPR (no '%') at the given byte size."""
    if reg in _LEGACY8:
        return {1: _LEGACY8[reg], 2: _LEGACY[reg], 4: "e" + _LEGACY[reg], 8: reg}[size]
    return {1: reg + "b", 2: reg + "w", 4: reg + "d", 8: reg}[size]


def _norm_size(size: int) -> int:
    return size if size in (1, 2, 4, 8) else 8


def _sse_to_gpr(cg, ty) -> None:
    """A float/double value in %xmm0 is moved to %rax as its raw bits."""
    if ty.kind == TypeKind.FLOAT:
        cg.emit("movd %xmm0, %eax")
    elif ty.kind == TypeKind.DOUBLE:
        cg.emit("movq %xmm0, %rax")


def _is_bad_operand(ty) -> bool:
    """Operand types that a GPR constraint cannot carry: x87/float16/float128
    values, and aggregates that are not 1, 2, 4 or 8 bytes big."""
    if ty.kind in _AGG_KINDS:
        return ty.size not in (1, 2, 4, 8)
    return ty.kind in _BAD_KINDS


def _bits_of_value(cg, ty) -> None:
    """Value in %rax (scalars already evaluated; aggregates as an address) as raw bits."""
    if ty.kind not in _AGG_KINDS:
        _sse_to_gpr(cg, ty)
        return
    # A 1/2/4/8-byte struct or union travels as its bytes, zero-extended.
    if ty.size == 1:
        cg.emit("movzbl (%rax), %eax")
    elif ty.size == 2:
        cg.emit("movzwl (%rax), %eax")
    elif ty.size == 4:
        cg.emit("movl (%rax), %eax")
    else:
        cg.emit("movq (%rax), %rax")


def _pick_alternative(cons: str) -> str:
    """Multi-alternative constraints ("=r,m"): keep the first alternative that
    the register/immediate/matching machinery can satisfy, else the first."""
    alts = cons.split(",")
    for alt in alts:
        c = "".join(ch for ch in alt if ch not in "=+&%")
        if c and (c.isdigit() or c.startswith("[") or set(c) & (_REG_LETTERS | _IMM_LETTERS)):
            return alt
    return alts[0]


class _Op:
    def __init__(self, cons: str, expr, is_out: bool, implicit: bool = False):
        self.cons = _pick_alternative(cons)
        self.expr = expr
        self.is_out = is_out
        self.rw = "+" in cons
        self.implicit = implicit  # GCC's input copy of a "+" output (never pushed)
        self.kind = "reg"        # reg | mem | imm | label (asm goto label operand)
        self.fixed = None        # fixed register for a/c/d/S/D
        self.match = None        # operand index for matching constraints
        self.imm = None
        self.reg = None          # assigned register (no '%')
        self.reg2 = None         # high-half register of a register pair (pair ops only)
        self.label = None        # label name (kind "label")
        self.stub = None         # jump-target stub emitted for kind "label"
        decays = expr is not None and expr.ty.kind in _DECAY_KINDS
        self.size = 8 if expr is None or decays else _norm_size(expr.ty.size)
        # A 16-byte _Complex long long travels in two GPRs (low qword, high qword).
        self.pair = expr is not None and expr.ty.is_complex and expr.ty.size == 16


def _imm_value(expr, dead: bool):
    """Integer constant, or an address constant rendered as `sym+off` for the assembler.
    In statically dead code a non-constant operand can never execute, so it gets a
    placeholder instead of a diagnostic."""
    try:
        label, v = constexpr.eval_reloc(expr)
    except constexpr.NotConstant:
        label, v = None, None
    if v is None and dead:
        return 0
    if v is None:
        diag.error(expr, "codegen: asm immediate operand is not constant", "codegen")
    if label is None:
        return v
    return label if v == 0 else f"{label}{v:+d}"


def _classify(op: _Op, dead: bool = False) -> None:
    # Constraints may list alternatives ("1,p", "=m,m"); use the first one.
    c = "".join(ch for ch in op.cons if ch not in "=+&%").split(",")[0]
    where = op.expr
    if c.isdigit():
        op.match = int(c)
        return
    letters = set(c)
    if not letters:
        # An empty constraint (`""(x)`) places no restriction: like "g",
        # let the operand live in a general register.
        op.kind = "reg"
        return
    if letters & _X87_LETTERS:
        op.kind = "x87"
        if op.expr.ty.kind not in (TypeKind.FLOAT, TypeKind.DOUBLE, TypeKind.LDOUBLE):
            diag.error(where, "codegen: x87 asm operands must be floating point", "codegen")
        if op.rw:
            diag.error(where, "codegen: read-write x87 asm operands are not supported", "codegen")
        return
    if letters & _REG_LETTERS:
        op.kind = "reg"
        fixed = [_FIXED[ch] for ch in c if ch in _FIXED]
        op.fixed = fixed[0] if fixed else None
    elif letters & _MEM_LETTERS:
        op.kind = "mem"
        if isinstance(op.expr, N.VarRef) and getattr(op.expr.var, "is_register", False):
            diag.error(where, f"address of register variable '{op.expr.var.name}' requested", "codegen")
    elif letters & _IMM_LETTERS:
        op.kind = "imm"
        op.imm = _imm_value(where, dead)
    else:
        diag.error(where, f"codegen: unsupported asm constraint '{op.cons}'", "codegen")
    if op.kind == "reg" and not op.pair and _is_bad_operand(op.expr.ty):
        diag.error(where, "codegen: float/aggregate asm operands are not supported", "codegen")


def _assign_x87(ops) -> None:
    """x87 operands get their FPU stack depth (st(N)) in op.reg."""
    for op in ops:
        if op.kind == "x87" and op.match is None:
            c = op.cons
            op.reg = 0 if "t" in c else 1 if "u" in c else None
    used = {op.reg for op in ops if op.kind == "x87" and op.match is None and op.reg is not None}
    for op in ops:
        if op.kind == "x87" and op.match is None and op.reg is None:
            d = 0
            while d in used:
                d += 1
            op.reg = d
            used.add(d)


def _assign_regs(ops, clobbers) -> None:
    used = set()
    for op in ops:
        if op.fixed:
            used.add(op.fixed)
    used |= {c for c in clobbers if c in _POOL or c in _CALLEE_SAVED}
    free = [r for r in _POOL + _CALLEE_SAVED if r not in used]
    _assign_x87(ops)
    for op in ops:
        if op.kind in ("imm", "x87", "label") or op.match is not None:
            continue
        if op.fixed:
            op.reg = op.fixed
        elif "Q" in op.cons:
            # "Q" is a/b/c/d only: those are the registers with %h high bytes.
            pick = next((r for r in ("rax", "rcx", "rdx") if r in free), None)
            if pick is None:
                diag.error(op.expr, "codegen: too many asm operands for available registers", "codegen")
            free.remove(pick)
            op.reg = pick
        else:
            if not free:
                diag.error(op.expr, "codegen: too many asm operands for available registers", "codegen")
            op.reg = free.pop(0)
        if op.pair:
            if not free:
                diag.error(op.expr, "codegen: too many asm operands for available registers", "codegen")
            op.reg2 = free.pop(0)
    for op in ops:
        if op.match is not None:
            if op.match >= len(ops):
                diag.error(op.expr, "codegen: asm matching constraint refers to a missing operand", "codegen")
            op.reg = ops[op.match].reg
            op.reg2 = ops[op.match].reg2
            op.pair = ops[op.match].pair
            op.kind = ops[op.match].kind


_HIGH8 = {"rax": "ah", "rcx": "ch", "rdx": "dh", "rbx": "bh"}


def _render(op: _Op, mod: str | None, loc) -> str:
    if op.kind == "label":
        return op.stub
    if op.kind == "imm":
        return str(op.imm) if mod in ("c", "P") else f"${op.imm}"
    if op.kind == "x87":
        return f"%st({op.reg})"
    if op.kind == "mem":
        return f"(%{op.reg})"
    if mod == "h":
        if op.reg not in _HIGH8:
            diag.error(loc, "codegen: operand has no high-byte register", "codegen")
        return "%" + _HIGH8[op.reg]
    size = {"k": 4, "w": 2, "b": 1, "q": 8}.get(mod, op.size)
    return "%" + _reg_name(op.reg, size)


def _subst(tpl: str, ops, names, loc) -> str:
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
        if i < n and tpl[i] in "kwbqcPh":
            mod = tpl[i]
            i += 1
        elif i + 1 < n and tpl[i] == "l" and (tpl[i + 1].isdigit() or tpl[i + 1] == "["):
            mod = "l"
            i += 1
        if i < n and tpl[i] == "[":
            end = tpl.find("]", i)
            if end < 0:
                diag.error(loc, "codegen: unsupported asm operand reference", "codegen")
            name = tpl[i + 1:end]
            idx = next((k for k, op in enumerate(ops) if op.kind == "label" and op.label == name), None)
            if idx is None:
                idx = names.get(name)
            if idx is None:
                diag.error(loc, f"codegen: unknown asm operand name '{name}'", "codegen")
            i = end + 1
        else:
            j = i
            while j < n and tpl[j].isdigit():
                j += 1
            if j == i:
                diag.error(loc, "codegen: unsupported asm operand reference", "codegen")
            idx = int(tpl[i:j])
            i = j
        if idx >= len(ops):
            diag.error(loc, "codegen: asm operand number out of range", "codegen")
        out.append(_render(ops[idx], mod, loc))
    return "".join(out)


_LVALUE_NODES = ("VarRef", "Deref", "Member_")


def _is_lvalue(node) -> bool:
    """True if the expression denotes an object whose address can be taken."""
    if node.ty.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.UNION, TypeKind.FUNC):
        return True
    name = type(node).__name__
    if name in _LVALUE_NODES:
        return True
    if name == "Cast":
        return _is_lvalue(node.operand)
    if name == "Comma":
        return _is_lvalue(node.rhs)
    return False


def _gen_value(cg, node) -> None:
    """Evaluate a scalar expression into %rax as raw 8-byte bits."""
    cg.gen_expr(node)
    _sse_to_gpr(cg, node.ty)


def lower_extended(cg, node) -> None:
    outs_all = [_Op(c, e, True) for c, e in node.outputs]
    ins_all = [_Op(c, e, False) for c, e in node.inputs]
    # GCC numbering: outputs, explicit inputs, then one implicit input per "+"
    # output (matching it), then the asm goto labels.
    implicit = [_Op(str(k), op.expr, False, implicit=True)
                for k, op in enumerate(outs_all) if op.rw]
    labels = []
    for name in node.goto_labels:
        lop = _Op("", None, False)
        lop.kind = "label"
        lop.label = name
        lop.stub = cg.new_label()
        labels.append(lop)
    ops = outs_all + ins_all + implicit + labels
    for op in ops:
        # Symbolic matching constraint "[name]" refers to the operand with that name.
        if op.cons.startswith("["):
            idx = node.names.get(op.cons[1:-1])
            if idx is None:
                diag.error(op.expr, f"codegen: unknown asm operand name '{op.cons}'", "codegen")
            op.cons = str(idx)
    clobbers = {c.strip().lstrip("%") for c in node.clobbers}
    for op in ops:
        if op.kind != "label":
            _classify(op, getattr(cg, "dead_code", False))
    _assign_regs(ops, clobbers)

    # Callee-saved registers the template may write are preserved on the stack
    # around the whole sequence (saved first, so they sit deepest).
    touched = {op.reg for op in ops if op.reg} | {op.reg2 for op in ops if op.reg2} | clobbers
    saved = [r for r in _CALLEE_SAVED if r in touched]
    for r in saved:
        cg.emit(f"push %{r}")
        cg.depth += 1

    outs = [op for op in ops if op.is_out and op.kind in ("reg", "x87")]
    regs_out = [op for op in outs if op.kind == "reg"]
    x87_out = sorted([op for op in outs if op.kind == "x87"], key=lambda o: o.reg)
    x87_in = [op for op in ops if op.kind == "x87" and not op.is_out]
    m = len(outs)
    r = sum(2 if op.pair else 1 for op in regs_out)  # stack slots for register outputs
    nx = len(x87_in)
    if sorted(op.reg for op in x87_in) != list(range(nx)):
        diag.error(node, "codegen: unsupported x87 asm operand arrangement", "codegen")
    # Phase 0: a "m" input that is not an lvalue (e.g. a statement expression)
    # is spilled to a stack slot that lives until the template is done.
    spill_depth = {}
    for op in ops:
        if op.kind == "mem" and not op.is_out and not _is_lvalue(op.expr):
            _gen_value(cg, op.expr)
            cg.push()
            spill_depth[id(op)] = cg.depth
    nspill = len(spill_depth)
    # Phase 1: addresses of outputs (register and x87).
    for op in outs:
        cg.gen_addr(op.expr)
        cg.push()
    # Phase 2a: x87 inputs, staged in 16-byte slots (first input deepest).
    units = 0
    for op in x87_in:
        val = op.expr
        if val.ty.kind != TypeKind.LDOUBLE:
            val = new_cast(val, TY_LDOUBLE)
        cg.gen_expr(val)
        cg.emit("sub $16, %rsp")
        cg.emit("fstpt (%rsp)")
        cg.depth += 2
        units += 2
    # Phase 2b: input values, "+" values and memory addresses.
    pushes = []
    for op in ops:
        if op.kind in ("imm", "x87", "label") or op.implicit or (op.is_out and op.kind == "reg" and not op.rw):
            continue
        if op.kind == "reg" and op.pair:
            # Two qwords of the 16-byte value: high pushed first, low on top.
            if op.is_out:
                k = outs.index(op)
                cg.emit(f"mov {8 * (units + m - 1 - k)}(%rsp), %rax")
            else:
                cg.gen_expr(op.expr)
            cg.emit("push 8(%rax)")
            cg.emit("push (%rax)")
            cg.depth += 2
            units += 2
            pushes.append(op)
            continue
        if op.kind == "mem" and id(op) in spill_depth:
            cg.emit(f"lea {8 * (cg.depth - spill_depth[id(op)])}(%rsp), %rax")
        elif op.kind == "mem":
            cg.gen_addr(op.expr)
        elif op.is_out and op.rw:
            k = outs.index(op)
            cg.emit(f"mov {8 * (units + m - 1 - k)}(%rsp), %rax")
            cg.load(op.expr.ty)
            _bits_of_value(cg, op.expr.ty)
        else:
            cg.gen_expr(op.expr)
            _bits_of_value(cg, op.expr.ty)
        cg.push()
        units += 1
        pushes.append(op)
    for op in reversed(pushes):
        cg.pop("%" + op.reg)
        # Only register pairs were pushed as two slots; a complex operand
        # passed by address ("m") is a single slot.
        if op.pair and op.kind == "reg":
            cg.pop("%" + op.reg2)
    # Phase 2c: load x87 inputs; the deepest st(N) goes first.
    for idx, op in sorted(enumerate(x87_in), key=lambda p: -p[1].reg):
        cg.emit(f"fldt {16 * (nx - 1 - idx)}(%rsp)")
    if nx:
        cg.emit(f"add ${16 * nx}, %rsp")
        cg.depth -= 2 * nx
    # Phase 3: the template.
    text = _subst(node.text, ops, node.names, node)
    for line in text.split("\n"):
        line = line.strip()
        if line:
            cg.emit(line)
    # Phase 4: store register outputs (pushed, then written back through the
    # saved addresses), then x87 outputs in increasing st(N) order.
    vals = []  # (op, qword half) for each pushed register output slot
    for op in regs_out:
        cg.emit(f"push %{op.reg}")
        vals.append((op, 0))
        if op.pair:
            cg.emit(f"push %{op.reg2}")
            vals.append((op, 1))
    for op in x87_out:
        k = outs.index(op)
        cg.emit(f"mov {8 * (r + m - 1 - k)}(%rsp), %rdi")
        cg.emit(_X87_STORE[op.expr.ty.kind] + " (%rdi)")
    for j, (op, half) in enumerate(vals):
        k = outs.index(op)
        cg.emit(f"mov {8 * (r + m - 1 - k)}(%rsp), %rdi")
        cg.emit(f"mov {8 * (r - 1 - j)}(%rsp), %rax")
        cg.emit(f"mov %{_reg_name('rax', op.size)}, {8 * half}(%rdi)")
    if m:
        cg.emit(f"add ${8 * (r + m)}, %rsp")
    cg.depth -= m       # phase-1 address slots are released here
    if nspill:
        cg.emit(f"add ${8 * nspill}, %rsp")
        cg.depth -= nspill
    for r in reversed(saved):
        cg.pop(f"%{r}")
    # Phase 5 (asm goto): a jump to a label leaves the template with the phase-1
    # output addresses still pushed, so each label is reached through a stub
    # that pops them, restores the saved registers and continues at the C label.
    if labels:
        skip = cg.new_label()
        cg.emit(f"jmp {skip}")
        for lop in labels:
            cg.label(lop.stub)
            if m + nspill:
                cg.emit(f"add ${8 * (m + nspill)}, %rsp")
            for r in reversed(saved):
                cg.emit(f"pop %{r}")
            cg.emit(f"jmp {cg.user_label(lop.label)}")
        cg.label(skip)
