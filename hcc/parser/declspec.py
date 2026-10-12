"""Declaration specifiers.

DeclSpecMixin provides:
    is_typename(tok=None) -> bool    does a type/declaration start here?
    declspec(attr: VarAttr | None) -> Type
Base type keywords are counted chibicc-style (e.g. `long long unsigned int`)
and mapped through _BASE_TYPES. Struct/union/enum/typedef/typeof/
__builtin_va_list/__auto_type yield OTHER.
"""
from __future__ import annotations

from hcc.sema.typing import add_type
from hcc.tokens import TokenKind
from hcc.parser.scope import VarAttr
from hcc.parser.attributes import mode_type
from hcc import typesys as T
from hcc.parser.vla import _has_vla as _has_vla_type
from hcc.parser.complex import complex_decl_type

VOID, BOOL, CHAR, SHORT, INT, LONG = 1 << 0, 1 << 2, 1 << 4, 1 << 6, 1 << 8, 1 << 10
FLOAT, DOUBLE, OTHER, SIGNED, UNSIGNED, INT128 = 1 << 12, 1 << 14, 1 << 16, 1 << 17, 1 << 18, 1 << 19

_BASE_TYPES = {
    VOID: T.TY_VOID, BOOL: T.TY_BOOL,
    CHAR: T.TY_CHAR, SIGNED + CHAR: T.TY_CHAR, UNSIGNED + CHAR: T.TY_UCHAR,
    SHORT: T.TY_SHORT, SHORT + INT: T.TY_SHORT, SIGNED + SHORT: T.TY_SHORT,
    SIGNED + SHORT + INT: T.TY_SHORT,
    UNSIGNED + SHORT: T.TY_USHORT, UNSIGNED + SHORT + INT: T.TY_USHORT,
    INT: T.TY_INT, SIGNED: T.TY_INT, SIGNED + INT: T.TY_INT,
    UNSIGNED: T.TY_UINT, UNSIGNED + INT: T.TY_UINT,
    LONG: T.TY_LONG, LONG + INT: T.TY_LONG, SIGNED + LONG: T.TY_LONG, SIGNED + LONG + INT: T.TY_LONG,
    LONG + LONG: T.TY_LLONG, LONG + LONG + INT: T.TY_LLONG, SIGNED + LONG + LONG: T.TY_LLONG,
    SIGNED + LONG + LONG + INT: T.TY_LLONG,
    UNSIGNED + LONG: T.TY_ULONG, UNSIGNED + LONG + INT: T.TY_ULONG,
    UNSIGNED + LONG + LONG: T.TY_ULLONG, UNSIGNED + LONG + LONG + INT: T.TY_ULLONG,
    FLOAT: T.TY_FLOAT, DOUBLE: T.TY_DOUBLE, LONG + DOUBLE: T.TY_LDOUBLE,
    INT128: T.TY_INT128, SIGNED + INT128: T.TY_INT128, UNSIGNED + INT128: T.TY_UINT128,
}

_COUNTS = {"void": VOID, "_Bool": BOOL, "char": CHAR, "short": SHORT, "int": INT, "long": LONG,
           "float": FLOAT, "double": DOUBLE, "signed": SIGNED, "unsigned": UNSIGNED,
           "__int128": INT128}

TYPE_KEYWORDS = frozenset(_COUNTS) | frozenset("""
typedef static extern inline _Thread_local auto register _Noreturn const volatile restrict
_Atomic _Alignas struct union enum typeof typeof_unqual __attribute__ __declspec __extension__
__builtin_va_list __auto_type __float128 __float80 __uint128_t _Complex __complex__ __complex __label__
constexpr
""".split())

# Identifiers that C23 turns into keywords (mapped to their keyword spelling).
C23_KEYWORDS = {"bool": "_Bool", "thread_local": "_Thread_local"}

STORAGE = frozenset(("typedef", "static", "extern", "inline", "_Thread_local", "auto",
                     "register", "_Noreturn"))

# ISO/IEC TS 18661-3 _FloatN / _FloatNx names. They are plain identifiers to the
# lexer (so headers may still typedef them); they map onto the base float type
# with the same representation; _Float16 is IEEE binary16 (see codegen/half.py).
FLOATN_TYPES = {"_Float16": T.TY_FLOAT16, "_Float32": T.TY_FLOAT32, "_Float64": T.TY_FLOAT64,
                "_Float32x": T.TY_FLOAT32X, "_Float64x": T.TY_FLOAT64X,
                "_Float128": T.TY_FLOAT128, "_Float128x": T.TY_FLOAT128}
# _Decimal32/64/128 (TR 24732): BID-encoded decimal floating types, see codegen/decimal.py.
DECIMAL_TYPES = {"_Decimal32": T.TY_DEC32, "_Decimal64": T.TY_DEC64, "_Decimal128": T.TY_DEC128}
NAMED_SCALAR_TYPES = {**FLOATN_TYPES, **DECIMAL_TYPES}


class DeclSpecMixin:

    def _index_of_tok(self, tok):
        """Index in self.toks of tok (searching forward from pos), or None."""
        for i in range(self.pos, min(self.pos + 16, len(self.toks))):
            if self.toks[i] is tok:
                return i
        return None

    def _skip_std_attr_run(self, i):
        """Index just past any C23 `[[...]]` groups starting at token index i."""
        toks = self.toks
        while i + 1 < len(toks) and toks[i].text == "[" and toks[i + 1].text == "[" \
                and toks[i].kind == TokenKind.PUNCT:
            depth = 0
            while i < len(toks) and toks[i].kind != TokenKind.EOF:
                if toks[i].kind == TokenKind.PUNCT and toks[i].text == "[":
                    depth += 1
                elif toks[i].kind == TokenKind.PUNCT and toks[i].text == "]":
                    depth -= 1
                i += 1
                if depth == 0:
                    break
        return i

    def is_typename(self, tok=None) -> bool:
        tok = tok or self.tok
        if tok.kind == TokenKind.KEYWORD and tok.text == "__extension__":
            # __extension__ is a no-op marker: it starts a declaration only if
            # the token after the marker run does (e.g. not `(__extension__ (x))`).
            i = self._index_of_tok(tok)
            if i is not None:
                while i < len(self.toks) and self.toks[i].kind == TokenKind.KEYWORD \
                        and self.toks[i].text == "__extension__":
                    i += 1
                i = self._skip_std_attr_run(i)
                if i < len(self.toks):
                    return self.is_typename(self.toks[i])
            return False
        if tok.kind == TokenKind.PUNCT and tok.text == "[":
            # C23 `[[...]] int x;`: attributes start a declaration if a specifier follows.
            i = self._index_of_tok(tok)
            if i is not None and self._skip_std_attr_run(i) > i:
                i = self._skip_std_attr_run(i)
                if i < len(self.toks):
                    return self.is_typename(self.toks[i])
            return False
        if tok.kind == TokenKind.KEYWORD:
            return tok.text in TYPE_KEYWORDS
        if tok.kind == TokenKind.IDENT and tok.text in NAMED_SCALAR_TYPES and self.find_typedef(tok) is None:
            return True
        if self.is_c23_bool(tok):
            return True
        return self.find_typedef(tok) is not None

    def is_c23_bool(self, tok):
        """C23 makes `bool` (= _Bool) and `thread_local` (= _Thread_local) keywords
        unless a typedef of that name is in scope."""
        return tok.kind == TokenKind.IDENT and tok.text in C23_KEYWORDS \
            and self.c23_mode() and self.find_typedef(tok) is None

    def declspec(self, attr=None):
        ty = T.TY_INT
        counter = 0
        vsize = 0
        auto_infer = False
        is_const = is_volatile = is_atomic = is_complex = False
        while self.is_typename():
            if self.at_std_attribute():
                self.skip_std_attributes()
                continue
            tok = self.tok
            s = tok.text
            if self.is_c23_bool(tok):
                s = C23_KEYWORDS[tok.text]
            if s in STORAGE:
                if attr is None:
                    self.error("storage class specifier is not allowed in this context")
                self.next()
                if s == "typedef":
                    attr.is_typedef = True
                elif s == "static":
                    attr.is_static = True
                elif s == "extern":
                    attr.is_extern = True
                elif s == "inline":
                    attr.is_inline = True
                elif s == "_Thread_local":
                    attr.is_tls = True
                elif s == "register":
                    attr.is_register = True
                elif s == "_Noreturn":
                    attr.is_noreturn = True
                elif s == "auto":
                    auto_infer = self.c23_mode()    # C23 `auto x = expr;`: no type given
                continue
            if s in ("_Complex", "__complex__", "__complex"):
                self.next()
                is_complex = True
                continue
            if s in ("const", "constexpr", "volatile", "restrict", "__extension__", "__label__"):
                self.next()
                is_const |= s in ("const", "constexpr")  # C23 constexpr objects are const
                is_volatile |= s == "volatile"
                continue
            if s == "_Atomic":
                self.next()
                if self.at("("):
                    self.next()
                    ty = self.typename()
                    self.expect(")")
                    counter += OTHER
                is_atomic = True
                continue
            if s in ("__attribute__", "__declspec"):
                if attr is None:      # type name: `(__attribute__((vector_size(N))) float)`
                    scratch = VarAttr()
                    self.parse_attributes(scratch)
                    vsize = scratch.vector_size or vsize
                else:
                    self.parse_attributes(attr)
                continue
            if s == "_Alignas":
                self.next()
                self.expect("(")
                if self.is_typename():
                    al = self.typename().align
                else:
                    al = self.const_expr()
                self.expect(")")
                if attr is not None:
                    attr.align = max(attr.align, al)
                continue
            # user-defined / special types
            td = self.find_typedef(tok)
            is_floatn = tok.kind == TokenKind.IDENT and s in NAMED_SCALAR_TYPES and td is None
            if s in ("struct", "union", "enum", "typeof", "typeof_unqual", "__builtin_va_list",
                     "__auto_type", "__float128", "__float80", "__uint128_t") or td is not None or is_floatn:
                if counter:
                    break          # e.g. "typedef int T; int T;" -> T is a declarator
                self.next()
                if s == "struct":
                    ty = self.struct_decl(T.TypeKind.STRUCT)
                elif s == "union":
                    ty = self.struct_decl(T.TypeKind.UNION)
                elif s == "enum":
                    ty = self.enum_specifier(attr)
                elif s == "typeof":
                    ty = self.typeof_specifier()
                elif s == "typeof_unqual":
                    ty = self.typeof_specifier(unqual=True)
                elif s == "__builtin_va_list":
                    ty = T.va_list_type()
                elif s == "__float128":
                    ty = T.TY_FLOAT128
                elif s == "__float80":
                    ty = T.TY_LDOUBLE       # x87 extended precision: long double on x86-64
                elif s == "__uint128_t":
                    ty = T.TY_UINT128
                elif s == "__auto_type":
                    ty = None       # resolved by the declaration's initializer
                elif is_floatn:
                    ty = NAMED_SCALAR_TYPES[s]
                else:
                    ty = td
                counter += OTHER
                continue
            if s not in _COUNTS:
                break
            self.next()
            counter += _COUNTS[s]
            if counter not in _BASE_TYPES:
                if counter & ~(OTHER - 1) and not (counter & OTHER) and counter in _BASE_TYPES:
                    pass
                self.error(f"invalid type: '{s}' here")
            ty = _BASE_TYPES[counter]
        if auto_infer and counter == 0 and attr is not None:
            return None         # type comes from the initializer, as for __auto_type
        if is_complex:
            ty = complex_decl_type(self, ty)
        if attr is not None and attr.vector_size and ty is not None:
            ty = T.vector_of(ty, attr.vector_size)
            attr.vector_size = 0
        elif attr is None and vsize and ty is not None:
            ty = T.vector_of(ty, vsize)
        if attr is not None and attr.mode:
            # __attribute__((mode(QI/HI/SI/DI/TI))) re-sizes an integer base type
            ty = mode_type(ty, attr.mode)
            attr.mode = None
        if counter == 0 and attr is None and not (is_const or is_volatile or is_complex):
            self.error("expected type")
        if ty is not None and (is_const or is_volatile or is_atomic):
            ty = ty.copy()
            ty.is_const |= is_const
            ty.is_volatile |= is_volatile
            ty.is_atomic |= is_atomic
        return ty

    def typeof_specifier(self, unqual=False):
        self.expect("(")
        if self.is_typename():
            ty = self.scoped_typename()
        else:
            node = self.expr()
            add_type(node)
            ty = node.ty
            if _has_vla_type(ty):
                # typeof(vm-expr) evaluates expr; locals.py emits it as a statement.
                self.vm_typeof.append(node)
        self.expect(")")
        if unqual and ty is not None and (ty.is_const or ty.is_volatile or ty.is_atomic):
            # typeof_unqual: drop the top-level qualifiers of the operand's type.
            ty = ty.copy()
            ty.is_const = ty.is_volatile = ty.is_atomic = False
        return ty
