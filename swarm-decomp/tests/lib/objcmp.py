"""Compare functions between two x86-64 ELF relocatable objects, symbolically.

Both objects are built with -ffunction-sections -fdata-sections, so every
function and variable sits in its own section. A function's bytes are compared
with every relocation field zeroed; the relocations are compared by what they
point at, never by address:

    named symbol            -> "<name>+<k>"
    merged string literal   -> 'str:"<text>"'
    .rodata.cstN constant   -> "cst:<hex bytes>"
    .text.<f> section       -> "fn:<f>+<k>"
    .data/.bss/.rodata.<v>  -> "var:<v>+<k>"   (gcc's ".N" numbering suffix dropped)

<k> is the offset inside the target the instruction really addresses (for a
PC-relative relocation the addend is corrected by the distance from the field
to the end of its instruction), so a string literal moving inside .rodata or a
variable moving inside .bss changes nothing.

A function passes when the masked bytes and the relocation lists are equal.
Otherwise `classify` names a diff category with its parameters, and `frac` is
the fraction of matching instructions (difflib ratio over normalised insns).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache

import capstone
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection

PC_REL = {2, 4, 9, 41, 42, 24, 26}       # PC32, PLT32, GOTPCREL, GOTPCRELX, REX_GOTPCRELX, PC64, GOTPC32
RSIZE = {1: 8, 2: 4, 4: 4, 9: 4, 10: 4, 11: 4, 24: 8, 41: 4, 42: 4}
RNAME = {1: "R_X86_64_64", 2: "R_X86_64_PC32", 4: "R_X86_64_PLT32", 9: "R_X86_64_GOTPCREL",
         10: "R_X86_64_32", 11: "R_X86_64_32S", 24: "R_X86_64_PC64", 41: "R_X86_64_GOTPCRELX",
         42: "R_X86_64_REX_GOTPCRELX"}


def _strip_num(name: str) -> str:
    """gcc numbers function-local statics and some clones: buf.0, buf.1. Drop a trailing .N."""
    return re.sub(r"\.\d+$", "", name)


@dataclass
class Func:
    name: str
    shndx: int
    value: int
    size: int
    bind: str
    parts: list = field(default_factory=list)   # extra (shndx, value, size, name) e.g. foo.cold


class Obj:
    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            data = f.read()
        import io
        self.elf = ELFFile(io.BytesIO(data))
        self.secs = list(self.elf.iter_sections())
        self.symtab = self.elf.get_section_by_name(".symtab")
        self.syms = list(self.symtab.iter_symbols()) if self.symtab else []
        self.sec_data = {}
        self.rels: dict[int, list] = {}
        for s in self.secs:
            if isinstance(s, RelocationSection):
                tgt = s["sh_info"]
                self.rels.setdefault(tgt, []).extend(
                    (r["r_offset"], r["r_info_type"], r["r_info_sym"], r["r_addend"]) for r in s.iter_relocations())
        self.funcs: dict[str, Func] = {}
        cold = []
        for sym in self.syms:
            if sym["st_info"]["type"] != "STT_FUNC" or sym["st_size"] == 0:
                continue
            if not isinstance(sym["st_shndx"], int):
                continue
            f = Func(sym.name, sym["st_shndx"], sym["st_value"], sym["st_size"], sym["st_info"]["bind"])
            if sym.name.endswith(".cold") or re.search(r"\.cold\.\d+$", sym.name):
                cold.append(f)
            else:
                self.funcs[sym.name] = f
        for c in cold:
            parent = self.funcs.get(re.sub(r"\.cold(\.\d+)?$", "", c.name))
            if parent:
                parent.parts.append(c)
            else:
                self.funcs[c.name] = c
        self.order = [n for n, _ in sorted(self.funcs.items(), key=lambda kv: (kv[1].shndx, kv[1].value))]

    def data(self, shndx: int) -> bytes:
        if shndx not in self.sec_data:
            s = self.secs[shndx]
            self.sec_data[shndx] = b"" if s["sh_type"] == "SHT_NOBITS" else s.data()
        return self.sec_data[shndx]

    def func_at(self, shndx: int, off: int) -> tuple[str, int] | None:
        best = None
        for f in self.funcs.values():
            for p in [f, *f.parts]:
                if p.shndx == shndx and p.value <= off < p.value + max(p.size, 1):
                    best = (p.name, off - p.value)
        return best

    def obj_at(self, shndx: int, off: int) -> tuple[str, int] | None:
        for sym in self.syms:
            if (sym["st_shndx"] == shndx and sym["st_info"]["type"] == "STT_OBJECT"
                    and sym["st_value"] <= off < sym["st_value"] + max(sym["st_size"], 1)):
                return sym.name, off - sym["st_value"]
        return None

    def describe(self, symidx: int, k: int) -> str:
        """What a relocation against symbol `symidx` with effective offset k points at."""
        sym = self.syms[symidx]
        typ = sym["st_info"]["type"]
        if typ == "STT_SECTION" or (sym.name.startswith(".L")):
            shndx = sym["st_shndx"]
            off = sym["st_value"] + k
            sec = self.secs[shndx]
            name = sec.name
            flags = sec["sh_flags"]
            if flags & 0x20 and flags & 0x10:                       # SHF_STRINGS | SHF_MERGE
                d = self.data(shndx)
                end = d.find(b"\0", off)
                s = d[off:end if end >= 0 else len(d)]
                return "str:" + _quote(s)
            if name.startswith(".rodata.cst"):
                n = int(re.sub(r"\D", "", name.split(".rodata.cst")[1]) or 8)
                return "cst:" + self.data(shndx)[off:off + n].hex()
            if name.startswith((".text", )):
                hit = self.func_at(shndx, off)
                if hit:
                    return f"fn:{hit[0]}+{hit[1]}"
                return f"sec:{name}+{off}"
            for pre in (".data.rel.ro.local.", ".data.rel.ro.", ".data.rel.local.", ".data.rel.",
                        ".rodata.", ".data.", ".bss.", ".tbss.", ".tdata."):
                if name.startswith(pre):
                    hit = self.obj_at(shndx, off)
                    if hit:
                        return f"var:{_strip_num(hit[0])}+{hit[1]}"
                    return f"var:{_strip_num(name[len(pre):])}+{off}"
            hit = self.obj_at(shndx, off)
            if hit:
                return f"var:{_strip_num(hit[0])}+{hit[1]}"
            return f"sec:{name}+{off}"
        name = sym.name
        if typ == "STT_OBJECT" and sym["st_info"]["bind"] == "STB_LOCAL":
            name = _strip_num(name)
        return f"{name}+{k}"


def _quote(b: bytes) -> str:
    out = []
    for c in b:
        ch = chr(c)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif 32 <= c < 127:
            out.append(ch)
        else:
            out.append({10: "\\n", 9: "\\t", 13: "\\r"}.get(c, f"\\x{c:02x}"))
    return '"' + "".join(out) + '"'


@lru_cache(maxsize=1)
def _cs():
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    md.syntax = capstone.CS_OPT_SYNTAX_ATT
    return md


@dataclass
class Insn:
    off: int
    size: int
    mnem: str
    ops: str
    ref: str = ""          # resolved relocation target(s) inside this instruction
    raw: bytes = b""

    def text(self) -> str:
        t = f"{self.mnem} {self.ops}".strip()
        if self.ref:
            # the operand still shows the zeroed field; the target is the reference
            if self.mnem.startswith(("call", "jmp", "j")) and re.fullmatch(r"0x[0-9a-f]+", self.ops or ""):
                t = f"{self.mnem} {self.ref}"
            else:
                t += f"   <{self.ref}>"
        return t


@dataclass
class Body:
    name: str
    masked: bytes
    relocs: list            # (offset in part, reloc type, target description)
    insns: list             # list[Insn]
    size: int


def body(obj: Obj, name: str) -> Body:
    f = obj.funcs[name]
    masked = bytearray()
    relocs, insns = [], []
    base = 0
    for p in [f, *f.parts]:
        raw = obj.data(p.shndx)[p.value:p.value + p.size]
        rels = sorted(r for r in obj.rels.get(p.shndx, []) if p.value <= r[0] < p.value + p.size)
        buf = bytearray(raw)
        for off, typ, _, _ in rels:
            n = RSIZE.get(typ, 4)
            o = off - p.value
            buf[o:o + n] = b"\0" * n
        pins = list(_cs().disasm(bytes(buf), base))
        ends = [(i.address - base, i.address - base + i.size) for i in pins]
        for off, typ, symidx, add in rels:
            o = off - p.value
            k = add
            if typ in PC_REL:
                end = next((e for s, e in ends if s <= o < e), o + RSIZE.get(typ, 4))
                k = add + (end - o)
            relocs.append((base + o, RNAME.get(typ, str(typ)), obj.describe(symidx, k)))
        rmap: dict[int, list] = {}
        for o, _, d in relocs:
            rmap.setdefault(o, []).append(d)
        for i in pins:
            refs = [d for o in range(i.address, i.address + i.size) for d in rmap.get(o, [])]
            insns.append(Insn(i.address, i.size, i.mnemonic, i.op_str, ", ".join(refs), bytes(i.bytes)))
        masked += buf
        base += p.size
    return Body(name, bytes(masked), relocs, insns, base)


REG = re.compile(r"%[a-z][a-z0-9]*")
NUM = re.compile(r"\$?-?0x[0-9a-f]+|\$?-?\b\d+\b")
BRANCH = re.compile(r"^(j[a-z]+|call|jmp|loop\w*)$")


def _norm(i: Insn) -> str:
    """Instruction text with intra-function branch targets masked (they shift with any insertion)."""
    if BRANCH.match(i.mnem) and not i.ref:
        return f"{i.mnem} <tgt>"
    return i.text()


def frac(a: Body, b: Body) -> float:
    sa, sb = [_norm(i) for i in a.insns], [_norm(i) for i in b.insns]
    if not sa and not sb:
        return 1.0
    return round(difflib.SequenceMatcher(None, sa, sb, autojunk=False).ratio(), 4)


def _calls(b: Body) -> list[str]:
    return [_sym(i.ref) for i in b.insns if i.mnem.startswith("call") and i.ref]


def _hx(s: str) -> str:
    v = int(s.lstrip("$"), 0)
    return ("m" if v < 0 else "") + format(abs(v), "x")


def classify(want: Body, got: Body, suite: str) -> tuple[str, float]:
    """(reason, frac) for a mismatch. Reason = '<category>: <params>', one line, specifics unmasked
    only where they should cluster (struct displacement pattern, callee name)."""
    fr = frac(want, got)
    wm, gm = [i.mnem for i in want.insns], [i.mnem for i in got.insns]
    if wm == gm:
        cats, disps, refs = set(), [], []
        for a, b in zip(want.insns, got.insns):
            ta, tb = a.text(), b.text()
            if ta == tb:
                continue
            if a.ref != b.ref:
                cats.add("ref")
                refs.append((a, b))
                continue
            if REG.sub("R", ta) == REG.sub("R", tb):
                cats.add("reg")
                continue
            na, nb = NUM.findall(a.ops), NUM.findall(b.ops)
            if NUM.sub("N", a.ops) == NUM.sub("N", b.ops) or NUM.sub("N", REG.sub("R", a.ops)) == NUM.sub("N", REG.sub("R", b.ops)):
                if "%rsp" in a.ops or "%rbp" in a.ops:
                    cats.add("frame")
                elif "(" in a.ops:
                    cats.add("offset")
                    disps += [(x, y) for x, y in zip(na, nb) if x != y and not x.startswith("$")]
                else:
                    cats.add("imm")
                continue
            cats.add("operand")
        if "ref" in cats:
            a, b = refs[0]
            if a.mnem.startswith("call") or a.mnem == "jmp":
                return f"call: expected call to {_sym(a.ref)}", fr
            return f"ref: wrong reference {_kind(a.ref)} in {suite}", fr
        if "frame" in cats:
            return f"frame: stack frame layout differs in {suite}", fr
        if "offset" in cats:
            pairs = sorted(set(disps))[:3]
            pat = " ".join(f"{_hx(x)}to{_hx(y)}" for x, y in pairs)
            return f"offset: displacement {pat}", fr
        if "imm" in cats:
            return f"imm: constant operand differs in {suite}", fr
        if cats == {"reg"}:
            return f"regalloc: only register names differ in {suite}", fr
        return f"operand: same opcodes, operands differ in {suite}", fr
    if sorted(wm) == sorted(gm):
        return f"cfg-layout: same instructions, block order differs in {suite}", fr
    cw, cg = _calls(want), _calls(got)
    missing = [c for c in dict.fromkeys(cw) if c not in cg]
    extra = [c for c in dict.fromkeys(cg) if c not in cw]
    if missing:
        return f"call: expected call to {missing[0]}", fr
    if extra:
        return f"call: unexpected call to {extra[0]}", fr
    bucket = "near" if fr >= 0.9 else "mid" if fr >= 0.6 else "far"
    return f"insn-delta: {bucket} in {suite}", fr


def _sym(ref: str) -> str:
    return re.sub(r"\+-?\d+$", "", ref.split(",")[0]).replace("fn:", "")


def _kind(ref: str) -> str:
    r = ref.split(",")[0]
    return r.split(":", 1)[0] if ":" in r else "symbol"


def compare(want_obj: Obj, got_obj: Obj, name: str, suite: str) -> dict:
    want = body(want_obj, name)
    got = body(got_obj, name)
    if want.masked == got.masked and want.relocs == got.relocs:
        return {"status": "PASS", "reason": "", "stage": "match", "frac": 1.0}
    reason, fr = classify(want, got, suite)
    return {"status": "FAIL", "reason": reason, "stage": "match", "frac": fr}


def listing(obj: Obj, name: str) -> str:
    b = body(obj, name)
    f = obj.funcs[name]
    out = [f"# {name}  ({'static' if f.bind == 'STB_LOCAL' else 'global'}, {b.size} bytes, "
           f"{len(b.insns)} instructions)"]
    for i in b.insns:
        out.append(f"{i.off:5x}:  {i.text()}")
    return "\n".join(out) + "\n"


def diff_text(want: Body, got: Body, context: int = 3) -> str:
    sa = [_norm(i) for i in want.insns]
    sb = [_norm(i) for i in got.insns]
    return "\n".join(difflib.unified_diff(sa, sb, "target", "yours", lineterm="", n=context))
