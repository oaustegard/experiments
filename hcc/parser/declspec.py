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
from hcc import typesys as T

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
_Atomic _Alignas struct union enum typeof __attribute__ __declspec __extension__
__builtin_va_list __auto_type __float128 __uint128_t _Complex __label__
""".split())

STORAGE = frozenset(("typedef", "static", "extern", "inline", "_Thread_local", "auto",
                     "register", "_Noreturn"))


class DeclSpecMixin:

    def is_typename(self, tok=None) -> bool:
        tok = tok or self.tok
        if tok.kind == TokenKind.KEYWORD:
            return tok.text in TYPE_KEYWORDS
        return self.find_typedef(tok) is not None

    def declspec(self, attr=None):
        ty = T.TY_INT
        counter = 0
        is_const = is_volatile = is_atomic = False
        while self.is_typename():
            tok = self.tok
            s = tok.text
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
                elif s == "_Noreturn":
                    attr.is_noreturn = True
                continue
            if s in ("const", "volatile", "restrict", "__extension__", "_Complex", "__label__"):
                self.next()
                is_const |= s == "const"
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
            if s in ("struct", "union", "enum", "typeof", "__builtin_va_list", "__auto_type",
                     "__float128", "__uint128_t") or td is not None:
                if counter:
                    break          # e.g. "typedef int T; int T;" -> T is a declarator
                self.next()
                if s == "struct":
                    ty = self.struct_decl(T.TypeKind.STRUCT)
                elif s == "union":
                    ty = self.struct_decl(T.TypeKind.UNION)
                elif s == "enum":
                    ty = self.enum_specifier()
                elif s == "typeof":
                    ty = self.typeof_specifier()
                elif s == "__builtin_va_list":
                    ty = T.va_list_type()
                elif s == "__float128":
                    ty = T.TY_FLOAT128
                elif s == "__uint128_t":
                    ty = T.TY_UINT128
                elif s == "__auto_type":
                    ty = None       # resolved by the declaration's initializer
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
        if counter == 0 and attr is None and not (is_const or is_volatile):
            self.error("expected type")
        if ty is not None and (is_const or is_volatile or is_atomic):
            ty = ty.copy()
            ty.is_const |= is_const
            ty.is_volatile |= is_volatile
            ty.is_atomic |= is_atomic
        return ty

    def typeof_specifier(self):
        self.expect("(")
        if self.is_typename():
            ty = self.typename()
        else:
            node = self.expr()
            add_type(node)
            ty = node.ty
        self.expect(")")
        return ty
