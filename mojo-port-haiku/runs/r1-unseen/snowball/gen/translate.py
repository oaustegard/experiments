#!/usr/bin/env python3
"""Translate snowballstemmer 3.1.1's pure-Python stemmers into one Mojo file.

Reads the generated modules english/german/russian/french from the reference
install and writes msnowball/kernel.mojo. Run from anywhere:

    python3 gen/translate.py

The generated Python uses `raise labN()` / `except labN: pass` as a goto. Those
jumps are lowered to a flag (`_jmp`) that is checked after each statement that
may leave a jump pending, at the end of loops that a jump leaves, and at the
end of the try that owns the label. Every label id is unique per routine.

The runtime (BaseStemmer) is copied into each language struct as a template,
so a language struct is self-contained and no inheritance is needed.
"""
from __future__ import annotations

import ast
import pathlib
import sys

REF = pathlib.Path("/usr/local/lib/python3.13/dist-packages/snowballstemmer")
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "msnowball" / "kernel.mojo"
LANGS = ['arabic', 'armenian', 'basque', 'catalan', 'czech', 'danish', 'dutch', 'dutch_porter', 'english', 'esperanto', 'estonian', 'finnish', 'french', 'german', 'greek', 'hindi', 'hungarian', 'indonesian', 'irish', 'italian', 'lithuanian', 'nepali', 'norwegian', 'persian', 'polish', 'porter', 'portuguese', 'romanian', 'russian', 'serbian', 'sesotho', 'spanish', 'swedish', 'tamil', 'turkish', 'yiddish']

# Runtime methods that return a value the generated code may discard.
RET_BOOL_RUNTIME = {
    "in_grouping", "in_grouping_b", "go_in_grouping", "go_in_grouping_b",
    "out_grouping", "out_grouping_b", "go_out_grouping", "go_out_grouping_b",
    "eq_s", "eq_s_b", "find_among", "find_among_b",
}
RET_NONE_RUNTIME = {"slice_from", "slice_del"}


def mname(attr: str) -> str:
    """Python method name -> Mojo method name."""
    if attr.startswith("__r_"):
        return "r_" + attr[4:]
    if attr == "_stem":
        return "stem_impl"
    return attr


def mojo_str(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    assert all(ord(c) >= 32 for c in out), repr(s)
    return f'"{out}"'


def cp_of(s: str) -> int:
    assert len(s) == 1, s
    return ord(s)


# ---------------------------------------------------------------- analysis

def annotate_tries(fn: ast.FunctionDef) -> int:
    """Give each Try an id and each Raise the id of the try it jumps to."""
    counter = [0]

    def walk(stmts, stack):
        for s in stmts:
            if isinstance(s, ast.Try):
                tid = counter[0]
                counter[0] += 1
                s._tid = tid
                assert len(s.handlers) == 1 and s.handlers[0].type is not None
                assert [type(x) for x in s.handlers[0].body] == [ast.Pass]
                assert not s.orelse and not s.finalbody
                walk(s.body, stack + [(s.handlers[0].type.id, tid)])
            elif isinstance(s, ast.Raise):
                lab = s.exc.func.id
                for name, tid in reversed(stack):
                    if name == lab:
                        s._target = tid
                        break
                else:
                    raise ValueError(f"{fn.name}: raise {lab} has no handler")
            elif isinstance(s, ast.If):
                walk(s.body, stack)
                walk(s.orelse, stack)
            elif isinstance(s, ast.While):
                walk(s.body, stack)

    walk(fn.body, [])
    return counter[0]


def raises_of(stmts) -> set:
    """Try ids targeted by raises lexically inside these statements."""
    out: set = set()
    for s in stmts:
        if isinstance(s, ast.Raise):
            out.add(s._target)
        elif isinstance(s, ast.Try):
            out |= raises_of(s.body)
        elif isinstance(s, ast.If):
            out |= raises_of(s.body) | raises_of(s.orelse)
        elif isinstance(s, ast.While):
            out |= raises_of(s.body)
    return out


def tries_of(stmts) -> set:
    out: set = set()
    for s in stmts:
        if isinstance(s, ast.Try):
            out.add(s._tid)
            out |= tries_of(s.body)
        elif isinstance(s, ast.If):
            out |= tries_of(s.body) | tries_of(s.orelse)
        elif isinstance(s, ast.While):
            out |= tries_of(s.body)
    return out


def pend(stmts) -> set:
    """Jump targets that may leave this sequence still pending."""
    return raises_of(stmts) - tries_of(stmts)


def collect_locals(fn: ast.FunctionDef) -> set:
    names: set = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


# ---------------------------------------------------------------- one language

class Language:
    def __init__(self, lang: str):
        self.lang = lang
        self.cls_name = "".join(p.capitalize() for p in lang.split("_")) + "Stemmer"
        mod = ast.parse((REF / f"{lang}_stemmer.py").read_text(encoding="utf-8"))
        cls = next(n for n in mod.body
                   if isinstance(n, ast.ClassDef) and n.name == self.cls_name)
        self.groups: dict[str, list[str]] = {}
        self.tables: dict[str, list[tuple]] = {}
        self.tuples: dict[str, list[str]] = {}
        self.scalars: dict[str, object] = {}
        self.methods: list[ast.FunctionDef] = []
        for n in cls.body:
            if isinstance(n, ast.FunctionDef):
                self.methods.append(n)
            elif isinstance(n, ast.Assign):
                name = n.targets[0].id
                v = n.value
                if name.startswith("g_"):
                    if isinstance(v, ast.Set):
                        chars = [e.value for e in v.elts]
                    elif isinstance(v, ast.Constant) and isinstance(v.value, str):
                        chars = list(v.value)
                    else:
                        raise ValueError(name)
                    for c in chars:
                        cp_of(c)
                    self.groups[name] = sorted(set(chars), key=ord)
                elif name.startswith("a_"):
                    rows = []
                    for call in v.elts:
                        assert isinstance(call, ast.Call) and call.func.id == "Among"
                        assert len(call.args) == 3 and not call.keywords
                        s, sub, res = (ast.literal_eval(a) for a in call.args)
                        rows.append((s, sub, res))
                    self.tables[name] = rows
                elif isinstance(v, ast.Tuple):
                    self.tuples[name] = [e.value for e in v.elts]
                elif isinstance(v, ast.Constant):
                    self.scalars[name] = v.value
                else:
                    raise ValueError(f"unexpected class attribute {name}")
        self.group_index = {g: i for i, g in enumerate(self.groups)}
        self.table_index = {t: i for i, t in enumerate(self.tables)}

    # -- expressions
    def ex(self, n, loc: set) -> str:
        if isinstance(n, ast.Constant):
            v = n.value
            if isinstance(v, bool):
                return "True" if v else "False"
            if isinstance(v, int):
                return str(v)
            if isinstance(v, str):
                return mojo_str(v)
            raise ValueError(repr(v))
        if isinstance(n, ast.Name):
            assert n.id in loc, n.id
            return n.id
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and n.value.id == "self":
                return f"self.{n.attr}"
            if isinstance(n.value, ast.Name) and n.value.id == self.cls_name:
                a = n.attr
                if a in self.group_index:
                    return f"Self.GR_{a}"
                if a in self.table_index:
                    return f"Self.TB_{a}"
                if a in self.tuples:
                    return f"self.ST_{a}"
                return f"self.{a}"
            raise ValueError(ast.dump(n))
        if isinstance(n, ast.Subscript):
            return f"{self.ex(n.value, loc)}[{self.ex(n.slice, loc)}]"
        if isinstance(n, ast.BinOp):
            op = {ast.Add: "+", ast.Sub: "-"}[type(n.op)]
            return f"({self.ex(n.left, loc)} {op} {self.ex(n.right, loc)})"
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return f"(not {self.ex(n.operand, loc)})"
            if isinstance(n.op, ast.USub):
                return f"(-{self.ex(n.operand, loc)})"
            raise ValueError(ast.dump(n))
        if isinstance(n, ast.BoolOp):
            op = " and " if isinstance(n.op, ast.And) else " or "
            return "(" + op.join(self.ex(v, loc) for v in n.values) + ")"
        if isinstance(n, ast.Compare):
            assert len(n.ops) == 1
            ops = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
                   ast.Gt: ">", ast.GtE: ">="}
            return (f"({self.cmp_operand(n.left, loc)} {ops[type(n.ops[0])]} "
                    f"{self.cmp_operand(n.comparators[0], loc)})")
        if isinstance(n, ast.Call):
            f = n.func
            assert isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
            assert f.value.id == "self", ast.dump(f)
            args = ", ".join(self.ex(a, loc) for a in n.args)
            name = mname(f.attr)
            # ASCII literals take the byte-wise variants (no UTF-8 decode).
            if f.attr in ("eq_s", "eq_s_b", "slice_from") and n.args \
                    and isinstance(n.args[0], ast.Constant) \
                    and isinstance(n.args[0].value, str) and n.args[0].value.isascii():
                name += "_a"
            return f"self.{name}({args})"
        raise ValueError(ast.dump(n))

    def cmp_operand(self, n, loc: set) -> str:
        """A one-char string literal compared against a char is its code point."""
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return str(cp_of(n.value))
        return self.ex(n, loc)

    # -- statements
    def block(self, stmts, ind: str, loc: set) -> list[str]:
        out: list[str] = []
        guarded = False
        for s in stmts:
            if guarded:
                out.append(f"{ind}if _jmp == 0:")
                out += self.stmt(s, ind + "    ", loc)
            else:
                out += self.stmt(s, ind, loc)
            if pend([s]):
                guarded = True
        if not out:
            out.append(f"{ind}pass")
        return out

    def if_stmt(self, s, ind: str, loc: set, kw: str) -> list[str]:
        out = [f"{ind}{kw} {self.ex(s.test, loc)}:"]
        out += self.block(s.body, ind + "    ", loc)
        if s.orelse:
            if len(s.orelse) == 1 and isinstance(s.orelse[0], ast.If):
                out += self.if_stmt(s.orelse[0], ind, loc, "elif")
            else:
                out.append(f"{ind}else:")
                out += self.block(s.orelse, ind + "    ", loc)
        return out

    def target(self, t, loc: set) -> str:
        if isinstance(t, ast.Name):
            assert t.id in loc
            return t.id
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                and t.value.id == "self":
            return f"self.{t.attr}"
        raise ValueError(ast.dump(t))

    def stmt(self, s, ind: str, loc: set) -> list[str]:
        if isinstance(s, ast.Assign):
            assert len(s.targets) == 1
            return [f"{ind}{self.target(s.targets[0], loc)} = {self.ex(s.value, loc)}"]
        if isinstance(s, ast.AugAssign):
            op = {ast.Add: "+=", ast.Sub: "-="}[type(s.op)]
            return [f"{ind}{self.target(s.target, loc)} {op} {self.ex(s.value, loc)}"]
        if isinstance(s, ast.Expr):
            if isinstance(s.value, ast.Constant):
                return [f"{ind}pass"]
            call = self.ex(s.value, loc)
            attr = s.value.func.attr
            if attr in RET_NONE_RUNTIME:
                return [f"{ind}{call}"]
            return [f"{ind}_ = {call}"]
        if isinstance(s, ast.If):
            return self.if_stmt(s, ind, loc, "if")
        if isinstance(s, ast.While):
            assert isinstance(s.test, ast.Constant) and s.test.value is True
            out = [f"{ind}while True:"]
            out += self.block(s.body, ind + "    ", loc)
            if pend(s.body):
                out.append(f"{ind}    if _jmp != 0:")
                out.append(f"{ind}        break")
            return out
        if isinstance(s, ast.Try):
            out = self.block(s.body, ind, loc)
            if s._tid in raises_of(s.body):
                out.append(f"{ind}if _jmp == {s._tid + 1}:")
                out.append(f"{ind}    _jmp = 0")
            return out
        if isinstance(s, ast.Raise):
            return [f"{ind}_jmp = {s._target + 1}"]
        if isinstance(s, ast.Break):
            return [f"{ind}break"]
        if isinstance(s, ast.Continue):
            return [f"{ind}continue"]
        if isinstance(s, ast.Return):
            assert s.value is not None
            return [f"{ind}return {self.ex(s.value, loc)}"]
        if isinstance(s, ast.Pass):
            return [f"{ind}pass"]
        raise ValueError(ast.dump(s))

    # -- methods
    def method(self, fn: ast.FunctionDef) -> list[str]:
        assert [a.arg for a in fn.args.args] == ["self"]
        ntries = annotate_tries(fn)
        loc = collect_locals(fn)
        out = [f"    def {mname(fn.name)}(mut self) -> Bool:"]
        for v in sorted(loc):
            out.append(f"        var {v}: Int = 0")
        if ntries:
            out.append("        var _jmp: Int = 0")
        out += self.block(fn.body, "        ", loc)
        if not out[-1].startswith("        return "):
            out.append("        return False")
        return out

    # -- the struct
    def struct(self) -> str:
        cls = self.cls_name
        fields = []
        for name, val in self.scalars.items():
            ty = "Bool" if isinstance(val, bool) else "Int"
            fields.append(f"    var {name}: {ty}")
        for name in self.tuples:
            fields.append(f"    var ST_{name}: List[StaticString]")
        consts = [f"    comptime GR_{g} = {i}" for g, i in self.group_index.items()]
        consts += [f"    comptime TB_{t} = {i}" for t, i in self.table_index.items()]

        init = ["        self.current = List[UInt32]()", "        self.cursor = 0",
                "        self.limit = 0", "        self.limit_backward = 0",
                "        self.bra = 0", "        self.ket = 0",
                "        self.grp_mask = List[UInt32]()", "        self.pool = List[UInt32]()",
                "        self.ent_off = List[Int]()", "        self.ent_len = List[Int]()",
                "        self.ent_sub = List[Int]()", "        self.ent_res = List[Int]()",
                "        self.tab_base = List[Int]()", "        self.tab_size = List[Int]()",
                "        self.n_groups = 0"]
        # Every field must be set before any method call on self.
        for name in self.tuples:
            init.append(f"        self.ST_{name} = List[StaticString]()")
        for name, val in self.scalars.items():
            lit = ("True" if val else "False") if isinstance(val, bool) else str(val)
            init.append(f"        self.{name} = {lit}")
        assert len(self.groups) <= 32, self.lang
        for g, chars in self.groups.items():
            init.append(f"        self.add_group({mojo_str(''.join(chars))})")
        for t, rows in self.tables.items():
            init.append("        self.begin_table()")
            for s, sub, res in rows:
                init.append(f"        self.add_entry({mojo_str(s)}, {sub}, {res})")
        for name, vals in self.tuples.items():
            for v in vals:
                init.append(f"        self.ST_{name}.append({mojo_str(v)})")

        body = "\n".join("\n".join(self.method(m)) for m in self.methods)
        head = (
            f"struct {cls}(Movable):\n"
            "    var current: List[UInt32]\n"
            "    var cursor: Int\n"
            "    var limit: Int\n"
            "    var limit_backward: Int\n"
            "    var bra: Int\n"
            "    var ket: Int\n"
            "    var grp_mask: List[UInt32]\n"
            "    var pool: List[UInt32]\n"
            "    var ent_off: List[Int]\n"
            "    var ent_len: List[Int]\n"
            "    var ent_sub: List[Int]\n"
            "    var ent_res: List[Int]\n"
            "    var n_groups: Int\n"
            "    var tab_base: List[Int]\n"
            "    var tab_size: List[Int]\n"
            + "\n".join(fields) + "\n"
            + "\n".join(consts) + "\n\n"
            "    def __init__(out self):\n"
            + "\n".join(init) + "\n\n"
        )
        return head + RUNTIME.replace("@CLS@", cls) + "\n" + body + "\n\n" + \
            RUN.replace("@CLS@", cls) + "\n"


RUNTIME = '''\
    # ---- BaseStemmer runtime (snowballstemmer/basestemmer.py), code-point based

    def in_grouping(mut self, g: Int) -> Bool:
        if self.cursor >= self.limit:
            return False
        if not self.has(g, self.current[self.cursor]):
            return False
        self.cursor += 1
        return True

    def in_grouping_b(mut self, g: Int) -> Bool:
        if self.cursor <= self.limit_backward:
            return False
        if not self.has(g, self.current[self.cursor - 1]):
            return False
        self.cursor -= 1
        return True

    def go_in_grouping(mut self, g: Int) -> Bool:
        var cur = self.current.unsafe_ptr()
        while self.cursor < self.limit:
            if not self.has(g, cur[unsafe_offset=self.cursor]):
                return True
            self.cursor += 1
        return False

    def go_in_grouping_b(mut self, g: Int) -> Bool:
        var cur = self.current.unsafe_ptr()
        while self.cursor > self.limit_backward:
            if not self.has(g, cur[unsafe_offset=self.cursor - 1]):
                return True
            self.cursor -= 1
        return False

    def out_grouping(mut self, g: Int) -> Bool:
        if self.cursor >= self.limit:
            return False
        if not self.has(g, self.current[self.cursor]):
            self.cursor += 1
            return True
        return False

    def go_out_grouping(mut self, g: Int) -> Bool:
        var cur = self.current.unsafe_ptr()
        while self.cursor < self.limit:
            if self.has(g, cur[unsafe_offset=self.cursor]):
                return True
            self.cursor += 1
        return False

    def out_grouping_b(mut self, g: Int) -> Bool:
        if self.cursor <= self.limit_backward:
            return False
        if not self.has(g, self.current[self.cursor - 1]):
            self.cursor -= 1
            return True
        return False

    def go_out_grouping_b(mut self, g: Int) -> Bool:
        var cur = self.current.unsafe_ptr()
        while self.cursor > self.limit_backward:
            if self.has(g, cur[unsafe_offset=self.cursor - 1]):
                return True
            self.cursor -= 1
        return False

    def eq_s(mut self, s: StaticString) -> Bool:
        var n = s.count_codepoints()
        if self.cursor + n > self.limit:
            return False
        var j = self.cursor
        for cp in s.codepoints():
            if self.current[j] != UInt32(Int(cp)):
                return False
            j += 1
        self.cursor += n
        return True

    def eq_s_b(mut self, s: StaticString) -> Bool:
        var n = s.count_codepoints()
        if self.cursor - n < self.limit_backward:
            return False
        var j = self.cursor - n
        for cp in s.codepoints():
            if self.current[j] != UInt32(Int(cp)):
                return False
            j += 1
        self.cursor -= n
        return True

    def find_among(mut self, t: Int) -> Int:
        var cur = self.current.unsafe_ptr()
        var pool = self.pool.unsafe_ptr()
        var elen = self.ent_len.unsafe_ptr()
        var eoff = self.ent_off.unsafe_ptr()
        var esub = self.ent_sub.unsafe_ptr()
        var eres = self.ent_res.unsafe_ptr()
        var base = self.tab_base[t]
        var i = 0
        var j = self.tab_size[t]
        var c = self.cursor
        var l = self.limit
        var common_i = 0
        var common_j = 0
        var first_key_inspected = False
        while True:
            var k = i + ((j - i) >> 1)
            var diff = 0
            var common = min(common_i, common_j)
            var wl = elen[unsafe_offset=base + k]
            var wo = eoff[unsafe_offset=base + k]
            for i2 in range(common, wl):
                if c + common == l:
                    diff = -1
                    break
                diff = Int(cur[unsafe_offset=c + common]) - Int(pool[unsafe_offset=wo + i2])
                if diff != 0:
                    break
                common += 1
            if diff < 0:
                j = k
                common_j = common
            else:
                i = k
                common_i = common
            if j - i <= 1:
                if i > 0:
                    break
                if j == i:
                    break
                if first_key_inspected:
                    break
                first_key_inspected = True
        while True:
            var wl = elen[unsafe_offset=base + i]
            if common_i >= wl:
                self.cursor = c + wl
                return eres[unsafe_offset=base + i]
            i = esub[unsafe_offset=base + i]
            if i < 0:
                return 0

    def find_among_b(mut self, t: Int) -> Int:
        var cur = self.current.unsafe_ptr()
        var pool = self.pool.unsafe_ptr()
        var elen = self.ent_len.unsafe_ptr()
        var eoff = self.ent_off.unsafe_ptr()
        var esub = self.ent_sub.unsafe_ptr()
        var eres = self.ent_res.unsafe_ptr()
        var base = self.tab_base[t]
        var i = 0
        var j = self.tab_size[t]
        var c = self.cursor
        var lb = self.limit_backward
        var common_i = 0
        var common_j = 0
        var first_key_inspected = False
        while True:
            var k = i + ((j - i) >> 1)
            var diff = 0
            var common = min(common_i, common_j)
            var wl = elen[unsafe_offset=base + k]
            var wo = eoff[unsafe_offset=base + k]
            for i2 in range(wl - 1 - common, -1, -1):
                if c - common == lb:
                    diff = -1
                    break
                diff = Int(cur[unsafe_offset=c - 1 - common]) - Int(pool[unsafe_offset=wo + i2])
                if diff != 0:
                    break
                common += 1
            if diff < 0:
                j = k
                common_j = common
            else:
                i = k
                common_i = common
            if j - i <= 1:
                if i > 0:
                    break
                if j == i:
                    break
                if first_key_inspected:
                    break
                first_key_inspected = True
        while True:
            var wl = elen[unsafe_offset=base + i]
            if common_i >= wl:
                self.cursor = c - wl
                return eres[unsafe_offset=base + i]
            i = esub[unsafe_offset=base + i]
            if i < 0:
                return 0

    def has(self, g: Int, cp: UInt32) -> Bool:
        var i = Int(cp)
        if i >= len(self.grp_mask):
            return False
        return ((self.grp_mask[i] >> UInt32(g)) & 1) == 1

    def add_group(mut self, s: StaticString):
        """Group g is bit g of grp_mask[cp]; the group index is the call order."""
        var cps = to_cps(s)
        var g = self.n_groups
        self.n_groups += 1
        var mx = 0
        for c in cps:
            if Int(c) > mx:
                mx = Int(c)
        while len(self.grp_mask) <= mx:
            self.grp_mask.append(0)
        for c in cps:
            self.grp_mask[Int(c)] = self.grp_mask[Int(c)] | (UInt32(1) << UInt32(g))

    def begin_table(mut self):
        self.tab_base.append(len(self.ent_off))
        self.tab_size.append(0)

    def add_entry(mut self, s: StaticString, sub: Int, res: Int):
        var cps = to_cps(s)
        self.ent_off.append(len(self.pool))
        self.ent_len.append(len(cps))
        for c in cps:
            self.pool.append(c)
        self.ent_sub.append(sub)
        self.ent_res.append(res)
        self.tab_size[len(self.tab_size) - 1] += 1

    def eq_s_a(mut self, s: StaticString) -> Bool:
        var n = s.byte_length()
        if self.cursor + n > self.limit:
            return False
        var j = self.cursor
        for b in s.as_bytes():
            if self.current[j] != UInt32(b):
                return False
            j += 1
        self.cursor += n
        return True

    def eq_s_b_a(mut self, s: StaticString) -> Bool:
        var n = s.byte_length()
        if self.cursor - n < self.limit_backward:
            return False
        var j = self.cursor - n
        for b in s.as_bytes():
            if self.current[j] != UInt32(b):
                return False
            j += 1
        self.cursor -= n
        return True

    def slice_from_a(mut self, s: StaticString):
        var bra = self.bra
        var ket = self.ket
        var n = s.byte_length()
        self.adjust_region(bra, ket, n - (ket - bra))
        var j = bra
        for b in s.as_bytes():
            self.current[j] = UInt32(b)
            j += 1
        self.ket = bra + n

    def adjust_region(mut self, c_bra: Int, c_ket: Int, adj: Int):
        """Resize the buffer by adj at [c_bra, c_ket); the tail moves with it."""
        var old_len = len(self.current)
        if adj > 0:
            for _ in range(adj):
                self.current.append(0)
            var j = old_len - 1
            while j >= c_ket:
                self.current[j + adj] = self.current[j]
                j -= 1
        elif adj < 0:
            for j in range(c_ket, old_len):
                self.current[j + adj] = self.current[j]
            self.current.shrink(old_len + adj)
        self.limit += adj
        if self.cursor >= c_ket:
            self.cursor += adj
        elif self.cursor > c_bra:
            self.cursor = c_bra

    def slice_from(mut self, s: StaticString):
        var bra = self.bra
        var ket = self.ket
        var n = s.count_codepoints()
        self.adjust_region(bra, ket, n - (ket - bra))
        var j = bra
        for cp in s.codepoints():
            self.current[j] = UInt32(Int(cp))
            j += 1
        self.ket = bra + n

    def slice_del(mut self):
        self.adjust_region(self.bra, self.ket, self.bra - self.ket)
        self.ket = self.bra
'''

RUN = '''\
    def run(
        mut self,
        n: Int,
        src: Pointer[UInt32, MutAnyOrigin],
        lens: Pointer[UInt32, MutAnyOrigin],
        dst: Pointer[UInt32, MutAnyOrigin],
        cap: Int,
    ) raises -> Int:
        """Stem n words. Input lengths are overwritten with output lengths."""
        var pos = 0
        var out = 0
        for i in range(n):
            var length = Int(lens[unsafe_offset=i])
            self.current.clear()
            for j in range(length):
                self.current.append(src[unsafe_offset=pos + j])
            pos += length
            self.cursor = 0
            self.limit = length
            self.limit_backward = 0
            self.bra = 0
            self.ket = length
            _ = self.stem_impl()
            var ol = len(self.current)
            if out + ol > cap:
                raise Error("msnowball: output buffer too small")
            for j in range(ol):
                dst[unsafe_offset=out + j] = self.current[j]
            out += ol
            lens[unsafe_offset=i] = UInt32(ol)
        return out
'''

HEADER = '''\
# GENERATED by gen/translate.py from snowballstemmer 3.1.1 (pure Python).
# Do not edit by hand: change the translator and regenerate.
#
# Four stemmers (english, german, russian, french), each a struct holding the
# BaseStemmer state. Text is code points (UInt32), matching Python str indexing.
# Python passes numpy buffers (UTF-32-LE words plus lengths) and gets back the
# stemmed words the same way; see __init__.py.

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer, Layout, alloc


def to_cps(s: StaticString) -> List[UInt32]:
    var out = List[UInt32]()
    for cp in s.codepoints():
        out.append(UInt32(Int(cp)))
    return out^


struct Grouping(Copyable, Movable):
    """A character set (Python grouping string/set) as a code-point bitmap."""
    var bits: List[Bool]

    def __init__(out self, s: StaticString):
        var cps = to_cps(s)
        var mx = 0
        for c in cps:
            if Int(c) > mx:
                mx = Int(c)
        self.bits = List[Bool]()
        for _ in range(mx + 1):
            self.bits.append(False)
        for c in cps:
            self.bits[Int(c)] = True

    def has(self, cp: UInt32) -> Bool:
        var i = Int(cp)
        if i >= len(self.bits):
            return False
        return self.bits[i]


struct Among(Copyable, Movable):
    """One row of a search table: string, substring link, result."""
    var s: List[UInt32]
    var substring_i: Int
    var result: Int

    def __init__(out self, s: StaticString, substring_i: Int, result: Int):
        self.s = to_cps(s)
        self.substring_i = substring_i
        self.result = result


'''

ENTRY = '''\
def new_@LANG@() raises -> PythonObject:
    """Allocate and initialise a @CLS@ on the heap; return its address."""
    var a = alloc(Layout[@CLS@](count=1))
    var p = a^.unsafe_leak()
    p.unsafe_write(@CLS@())
    return PythonObject(Int(p))


def stem_@LANG@(
    handle: PythonObject,
    n: PythonObject,
    src: PythonObject,
    lens: PythonObject,
    dst: PythonObject,
    cap: PythonObject,
) raises -> PythonObject:
    var st = Pointer[@CLS@, MutAnyOrigin](unsafe_from_address=Int(py=handle))
    var src_p = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(py=src))
    var lens_p = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(py=lens))
    var dst_p = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(py=dst))
    return PythonObject(st[].run(Int(py=n), src_p, lens_p, dst_p, Int(py=cap)))


'''

FOOTER_HEAD = '''\
@export
def PyInit__kernel() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_kernel")
'''
FOOTER_TAIL = '''\
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
'''


def main() -> int:
    langs = [Language(lang) for lang in LANGS]
    parts = [HEADER]
    registrations = []
    for lg in langs:
        parts.append(lg.struct())
        parts.append("\n")
        entry = ENTRY.replace("@LANG@", lg.lang).replace("@CLS@", lg.cls_name)
        parts.append(entry)
        registrations.append(f'        m.def_function[new_{lg.lang}]("new_{lg.lang}")')
        registrations.append(f'        m.def_function[stem_{lg.lang}]("stem_{lg.lang}")')
    parts.append(FOOTER_HEAD + "\n".join(registrations) + "\n" + FOOTER_TAIL)
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
