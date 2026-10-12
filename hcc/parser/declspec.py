"""Declaration specifiers.

DeclSpecMixin provides:
    is_typename(tok=None) -> bool    does a type/declaration start here?
    declspec(attr: VarAttr | None) -> Type
Base type keywords are counted chibicc-style (e.g. `long long unsigned int`)
and mapped through _BASE_TYPES. Typedef names, _FloatN and the keywords
registered in hcc/parser/typespecs/ (struct, union, enum, typeof, __auto_type, ...)
yield OTHER.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc.sema.typing import add_type
from hcc.tokens import TokenKind
from hcc.parser.scope import VarAttr
from hcc.parser.attributes import mode_type
from hcc.parser.typespecs import type_specifier
from hcc import typesys as T
from hcc.parser.vla import _has_vla as _has_vla_type
from hcc.parser.complex import complex_decl_type


def constexpr_type(ty):
    """C23 `constexpr` object: the object's own (top-level) type is const-qualified,
    e.g. `constexpr int *p` is `int *const p` (not `const int *`)."""
    if ty is None:
        return None
    ty = ty.copy()
    ty.is_const = True
    return ty

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

# Keywords that start a declaration, besides the registered type specifiers
# (hcc/parser/typespecs/: struct, union, enum, typeof, __builtin_va_list, ...).
TYPE_KEYWORDS = frozenset(_COUNTS) | frozenset("""
typedef static extern inline _Thread_local auto register _Noreturn const volatile restrict
_Atomic _Alignas __attribute__ __declspec __extension__
_Complex __complex__ __complex __label__
constexpr
""".split())

# Identifiers that C23 turns into keywords (mapped to their keyword spelling).
C23_KEYWORDS = {"bool": "_Bool", "thread_local": "_Thread_local"}

STORAGE = frozenset(("typedef", "static", "extern", "inline", "_Thread_local", "auto",
                     "register", "_Noreturn"))
# Storage classes that may appear at most once and not together (C11 6.7.1p2).
STORAGE_CLASSES = frozenset(("typedef", "static", "extern", "auto", "register"))

# ISO/IEC TS 18661-3 _FloatN / _FloatNx names. They are plain identifiers to the
# lexer (so headers may still typedef them); they map onto the base float type
# with the same representation; _Float16 is IEEE binary16 (see codegen/half.py).
FLOATN_TYPES = {"_Float16": T.TY_FLOAT16, "_Float32": T.TY_FLOAT32, "_Float64": T.TY_FLOAT64,
                "_Float32x": T.TY_FLOAT32X, "_Float64x": T.TY_FLOAT64X,
                "_Float128": T.TY_FLOAT128}
# _Float128x is deliberately absent: like gcc on x86-64 it is not a supported type, so
# it stays a plain identifier and is rejected as an unknown type name.
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
            return tok.text in TYPE_KEYWORDS or tok.text in type_specifier
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
        tmode = None          # type-name __attribute__((mode(X)))
        auto_infer = False
        is_const = is_volatile = is_atomic = is_complex = is_restrict = False
        seen_storage = set()  # storage class specifiers seen in this declaration
        tag_hit = None        # struct/union tag of an empty declaration (see StructMixin.struct_decl)
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
                if s in STORAGE_CLASSES:
                    if s in seen_storage:
                        self.error(f"duplicate '{s}'")
                    if seen_storage & STORAGE_CLASSES:
                        self.error("multiple storage classes in declaration specifiers")
                if s not in ("inline", "_Noreturn"):
                    seen_storage.add(s)
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
                if s == "constexpr" and attr is not None:
                    attr.is_constexpr = True    # applied to the declared object, see constexpr_type()
                else:
                    is_const |= s in ("const", "constexpr")  # C23 constexpr objects are const
                is_volatile |= s == "volatile"
                is_restrict |= s == "restrict"
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
                    tmode = scratch.mode or tmode
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
                    self.check_align_limit(al)
                self.expect(")")
                if attr is not None:
                    attr.align = max(attr.align, al)
                    attr.alignas_min = max(attr.alignas_min, al)
                # also noted for a compound literal's type name (postfix.py checks it)
                self.typename_alignas = max(getattr(self, "typename_alignas", 0), al)
                continue
            # user-defined / special types
            td = self.find_typedef(tok)
            is_floatn = tok.kind == TokenKind.IDENT and s in NAMED_SCALAR_TYPES and td is None
            spec = type_specifier.get(s)      # hcc/parser/typespecs/<name>.py
            if spec is not None or td is not None or is_floatn:
                if counter:
                    break          # e.g. "typedef int T; int T;" -> T is a declarator
                if td is not None:
                    self.check_unavailable(tok.text, tok)
                self.next()
                if spec is not None:
                    self.empty_tag_hit = None
                    ty = spec(self, tok, attr)
                    tag_hit = self.empty_tag_hit or tag_hit
                    self.empty_tag_hit = None
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
        if tag_hit is not None:
            _check_empty_tag_decl(self, tag_hit, is_const or is_volatile or is_restrict, seen_storage - {"typedef"})
        if attr is not None and attr.is_tls:
            # __thread (_Thread_local) only applies to objects with static storage or extern
            bads = ("typedef", "register") if self.c23_mode() else ("typedef", "auto", "register")
            for bad in bads:      # C23 `auto` is type inference, not a storage class
                if bad in seen_storage:
                    self.error(f"'__thread' used with '{bad}'")
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
        elif attr is None and tmode and ty is not None:
            ty = mode_type(ty, tmode)
        if counter == 0 and attr is None and not (is_const or is_volatile or is_complex):
            self.error("expected type")
        if is_restrict and ty is not None and not _is_object_pointer(ty):
            # C11 6.7.3p2: restrict only qualifies object pointer types
            self.error("invalid use of 'restrict'")
        if ty is not None and ty.is_vector and (is_const or is_volatile):
            ty = _qualified_vector(ty, is_const, is_volatile)   # elements carry the qualifiers
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
            if isinstance(node, N.Member_) and node.member.is_bitfield:
                # C: typeof/typeof_unqual of a bit-field member is a constraint violation.
                self.error("'typeof' applied to a bit-field")
            if _has_vla_type(ty):
                # typeof(vm-expr) evaluates expr; locals.py emits it as a statement.
                self.vm_typeof.append(node)
        self.expect(")")
        if unqual and ty is not None and (ty.is_const or ty.is_volatile or ty.is_atomic):
            # typeof_unqual: drop the top-level qualifiers of the operand's type.
            ty = ty.copy()
            ty.is_const = ty.is_volatile = ty.is_atomic = False
        return ty


def _is_object_pointer(ty):
    """True for a pointer whose pointee is an object (not function) type."""
    return ty.kind == T.TypeKind.PTR and ty.base is not None \
        and ty.base.kind != T.TypeKind.FUNC


def _qualified_vector(vty, is_const, is_volatile):
    """Rebuild a GCC vector type whose element type carries const/volatile."""
    elem = vty.base.copy()
    elem.is_const |= is_const
    elem.is_volatile |= is_volatile
    return T.vector_of(elem, vty.size)


def check_empty_inline(self, attr) -> None:
    """`int inline;` has no declarator: GCC rejects the function specifier there."""
    if attr is not None and attr.is_inline and self.at(";"):
        self.error("'inline' in empty declaration", self.tok.loc)


def _check_empty_tag_decl(self, tag_hit, quals: bool, storage: set) -> None:
    """C99 6.7.2.3p8: `const struct T;` / `extern union T;` with T visible does not redeclare T."""
    if not (quals or storage):
        return
    tok, wrong_kind = tag_hit
    if wrong_kind:
        self.error(f"'{tok.text}' defined as wrong kind of tag", tok)
    if quals:
        self.error("empty declaration with type qualifier does not redeclare tag", tok)
    self.error("empty declaration with storage class specifier does not redeclare tag", tok)
