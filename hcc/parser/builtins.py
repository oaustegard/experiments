"""Parsing of special __builtin_* forms (those taking types, or needing
compile-time evaluation). Plain library builtins such as __builtin_memcpy
are handled in primary.py by calling the libc function of the same name.

BuiltinParseMixin.parse_builtin(name_tok) -> Expr
To add one: put the name in SPECIAL_BUILTINS and add a `_b_<name>` method
(name without the `__builtin_` prefix); lower it in codegen/builtins.py if it
produces a BuiltinCall.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.parser.structs import find_member
from hcc.sema.compat import is_compatible
from hcc.sema.constexpr import try_eval_const
from hcc.sema.conversions import new_cast
from hcc import typesys as T

SPECIAL_BUILTINS = frozenset("""
__builtin_va_start __builtin_va_end __builtin_va_arg __builtin_va_copy __builtin_offsetof
__builtin_types_compatible_p __builtin_expect __builtin_constant_p __builtin_unreachable
__builtin_trap __builtin_alloca __builtin_choose_expr __builtin_classify_type
__builtin_add_overflow __builtin_sub_overflow __builtin_mul_overflow __builtin_mul_overflow_p
__builtin_prefetch
__builtin_bswap16 __builtin_bswap32 __builtin_bswap64
""".split())

# Bit-counting builtins: {clz,ctz,popcount,parity,ffs}{,l,ll}. The suffix picks
# the operand width (int = 32 bit, long/long long = 64 bit).
BIT_BUILTINS = frozenset(
    f"__builtin_{base}{suffix}"
    for base in ("clz", "ctz", "popcount", "parity", "ffs", "clrsb")
    for suffix in ("", "l", "ll"))
SPECIAL_BUILTINS = SPECIAL_BUILTINS | BIT_BUILTINS


class BuiltinParseMixin:

    def parse_builtin(self, tok):
        name = tok.text[len("__builtin_"):]
        self.expect("(")
        if tok.text in BIT_BUILTINS:
            node = self._bitop(tok, name)
        else:
            node = getattr(self, "_b_" + name)(tok)
        self.expect(")")
        return node

    def _bitop(self, tok, name):
        """clz/ctz/popcount/parity/ffs: argument converted to unsigned int
        (or unsigned long for the l/ll forms); result is int."""
        base = name.rstrip("l")
        bits = 64 if name[len(base):] else 32
        ty = T.TY_ULONG if bits == 64 else T.TY_UINT
        if base in ("ffs", "clrsb"):  # signed operands
            ty = T.TY_LONG if bits == 64 else T.TY_INT
        args = [new_cast(a, ty) for a in self._args(1)]
        return self._call(tok, f"{base}{bits}", args, T.TY_INT)

    def _args(self, n=None):
        args = []
        while not self.at(")"):
            if args:
                self.expect(",")
            args.append(self.typed(self.assign()))
        if n is not None and len(args) != n:
            self.error(f"builtin expects {n} arguments")
        return args

    def _call(self, tok, name, args, ty, type_arg=None):
        return N.BuiltinCall(tok.loc, name, args, type_arg, ty=ty)

    def _b_va_start(self, tok):
        args = self._args()
        return self._call(tok, "va_start", args[:1], T.TY_VOID)

    def _b_va_end(self, tok):
        self._args()
        return N.Cast(tok.loc, N.IntLit(tok.loc, 0, ty=T.TY_INT), ty=T.TY_VOID)

    def _b_va_copy(self, tok):
        return self._call(tok, "va_copy", self._args(2), T.TY_VOID)

    def _b_va_arg(self, tok):
        ap = self.typed(self.assign())
        self.expect(",")
        ty = self.typename()
        return self._call(tok, "va_arg", [ap], ty, ty)

    def _b_prefetch(self, tok):
        """__builtin_prefetch(addr[, rw[, locality]]) is only a cache hint: it
        evaluates its arguments (for side effects) and yields void."""
        args = self._args()
        if not 1 <= len(args) <= 3:
            self.error("builtin expects 1 to 3 arguments")
        node = N.Cast(tok.loc, N.IntLit(tok.loc, 0, ty=T.TY_INT), ty=T.TY_VOID)
        for a in reversed(args):
            node = N.Comma(tok.loc, N.Cast(tok.loc, a, ty=T.TY_VOID), node, ty=T.TY_VOID)
        return node

    def _bswap(self, tok, bits, ty):
        args = [new_cast(a, ty) for a in self._args(1)]
        return self._call(tok, f"bswap{bits}", args, ty)

    def _b_bswap16(self, tok):
        return self._bswap(tok, 16, T.TY_USHORT)

    def _b_bswap32(self, tok):
        return self._bswap(tok, 32, T.TY_UINT)

    def _b_bswap64(self, tok):
        return self._bswap(tok, 64, T.TY_ULONG)

    def _b_alloca(self, tok):
        args = [new_cast(a, T.TY_ULONG) for a in self._args(1)]
        return self._call(tok, "alloca", args, T.pointer_to(T.TY_VOID))

    def _overflow(self, tok, name):
        """__builtin_{add,sub,mul}_overflow(a, b, &res): operands are widened to
        exact 64-bit LONG/ULONG (by their own signedness); codegen does the check."""
        args = self._args(3)
        ops = [new_cast(a, T.TY_ULONG if a.ty.is_unsigned else T.TY_LONG) for a in args[:2]]
        res = args[2]
        if res.ty.kind != T.TypeKind.PTR or res.ty.base is None:
            self.error(f"'__builtin_{name}_overflow' expects a pointer as third argument")
        return self._call(tok, name + "_overflow", ops + [res], T.TY_INT, res.ty.base)

    def _b_add_overflow(self, tok):
        return self._overflow(tok, "add")

    def _b_sub_overflow(self, tok):
        return self._overflow(tok, "sub")

    def _b_mul_overflow(self, tok):
        return self._overflow(tok, "mul")

    def _b_mul_overflow_p(self, tok):
        """__builtin_mul_overflow_p(a, b, c): 1 iff a*b does not fit the type of
        c. Only the type of c is used (its value is never read or stored)."""
        args = self._args(3)
        ops = [new_cast(a, T.TY_ULONG if a.ty.is_unsigned else T.TY_LONG) for a in args[:2]]
        return self._call(tok, "mul_overflow_p", ops, T.TY_INT, args[2].ty)

    def _b_expect(self, tok):
        args = self._args(2)
        # The second argument (the expected value) is still evaluated for its
        # side effects; the result is the first argument.
        seq = self.typed(N.Comma(tok.loc, args[1], args[0]))
        return new_cast(seq, T.TY_LONG)

    def _b_constant_p(self, tok):
        args = self._args(1)
        return N.IntLit(tok.loc, int(try_eval_const(args[0]) is not None), ty=T.TY_INT)

    def _b_unreachable(self, tok):
        return self._call(tok, "trap", [], T.TY_VOID)

    def _b_trap(self, tok):
        return self._call(tok, "trap", [], T.TY_VOID)

    def _b_classify_type(self, tok):
        args = self._args(1)
        t = args[0].ty
        v = 1 if T.is_integer(t) else 8 if T.is_flonum(t) else 5 if t.base is not None else 12
        return N.IntLit(tok.loc, v, ty=T.TY_INT)

    def _b_types_compatible_p(self, tok):
        a = self.typename().unqual()
        self.expect(",")
        b = self.typename().unqual()
        return N.IntLit(tok.loc, int(is_compatible(a, b)), ty=T.TY_INT)

    def _b_choose_expr(self, tok):
        c = self.const_expr()
        self.expect(",")
        a = self.assign()
        self.expect(",")
        b = self.assign()
        return a if c else b

    def _b_offsetof(self, tok):
        ty = self.typename()
        self.expect(",")
        off = 0
        cur = ty
        first = True
        while True:
            if first or self.accept("."):
                first = False
                name = self.expect_ident()
                path = find_member(cur, name.text)
                if path is None:
                    self.error(f"'{cur}' has no member named '{name.text}'", name)
                for m in path:
                    off += m.offset
                    cur = m.ty
            elif self.accept("["):
                idx = self.const_expr()
                self.expect("]")
                off += idx * cur.base.size
                cur = cur.base
            else:
                break
        return N.IntLit(tok.loc, off, ty=T.TY_ULONG)
