"""GNU attributes and asm labels.

AttributeMixin provides:
    parse_attributes(attr: VarAttr | None, ty: Type | None)  skips/records any
        sequence of `__attribute__((...))`; recognized ones update attr/ty
        (aligned, packed, noreturn, weak, section, visibility, alias,
        vector_size(TODO), mode(TODO)). Unknown attributes are ignored.
    parse_asm_label(attr)  `asm("symbol")` after a declarator -> attr.asm_name
    skip_balanced()        skip a parenthesized group
"""
from __future__ import annotations

from hcc.tokens import TokenKind
from hcc import typesys as T

# byte sizes for __attribute__((mode(...))) integer modes (x86-64: word = DI)
# function attributes hcc records by name only (queried via __builtin_has_attribute)
RECORDED_FN_ATTRS = frozenset({"leaf", "returns_nonnull"})

MODE_SIZES = {"QI": 1, "HI": 2, "SI": 4, "DI": 8, "TI": 16, "word": 8}

# GCC's MAX_OFILE_ALIGNMENT for x86-64 ELF: 2**28 bytes
OBJECT_FILE_MAX_ALIGN = 1 << 28

# largest vector_size byte count GCC accepts on LP64 (ssizetype maximum)
VECTOR_SIZE_MAX = (1 << 63) - 1

# every other machine mode GCC knows by name; mode(<other identifier>) is an
# "unknown machine mode" error
_OTHER_MODES = frozenset({
    "BI", "OI", "XI", "PSI", "PDI", "HF", "SF", "DF", "XF", "TF", "SD", "DD", "TD",
    "SC", "DC", "XC", "TC", "CQI", "CHI", "CSI", "CDI", "CTI", "COI", "CHF", "CSF",
    "CDF", "CXF", "CTF", "QQ", "HQ", "SQ", "DQ", "TQ", "UQQ", "UHQ", "USQ", "UDQ",
    "UTQ", "HA", "SA", "DA", "TA", "UHA", "USA", "UDA", "UTA", "CQQ", "CHQ", "CSQ",
    "CDQ", "CTQ", "CHA", "CSA", "CDA", "CTA", "CUQQ", "CUHQ", "CUSQ", "CUDQ", "CUTQ",
    "CUHA", "CUSA", "CUDA", "CUTA", "pointer", "byte", "unwind_word", "V2QI", "V4QI",
    "V8QI", "V2HI", "V4HI", "V2SI", "V1DI", "V2DI", "V4SI", "V8HI", "V16QI", "V4DI",
    "V8SI", "V16HI", "V32QI", "V2SF", "V4SF", "V2DF", "V8SF", "V4DF", "V1TI",
})

# accepted tls_model("...") arguments (as raw bytes of the string literal)
TLS_MODELS = frozenset({b"global-dynamic", b"local-dynamic", b"initial-exec", b"local-exec"})

# GCC-accepted argument counts (min, max) for attributes that take a fixed
# number of arguments; anything outside the range is a hard error.
ARG_COUNTS = {
    "copy": (1, 1),
    "assume_aligned": (1, 2),
    "alloc_align": (1, 1),
    "assume": (1, 1),
    "access": (2, 3),
    "nonnull_if_nonzero": (2, 3),
    "strict_flex_array": (1, 1),
}


def count_attr_args(args) -> int:
    """Number of comma-separated top-level arguments in an attribute's tokens."""
    if not args:
        return 0
    n, depth = 1, 0
    for t in args:
        if t.kind != TokenKind.PUNCT:
            continue
        if t.text in ("(", "[", "{"):
            depth += 1
        elif t.text in (")", "]", "}"):
            depth -= 1
        elif t.text == "," and depth == 0:
            n += 1
    return n


def mode_type(ty, mode):
    """Re-size an integer or enum type to the byte size named by a mode attribute.

    Returns `ty` unchanged for unknown modes or non-integer types; never mutates
    `ty` (enums are copied, integers come from the shared base-type table)."""
    size = MODE_SIZES.get(mode)
    if size is None or ty is None:
        return ty
    if ty.kind == T.TypeKind.ENUM:
        ty = ty.copy()
        ty.size = ty.align = size
        return ty
    if T.is_integer(ty):
        return T.int_type_for(size, ty.is_unsigned)
    return ty


# C23 / GNU standard attributes that take no arguments.
_NO_ARG_STD_ATTRS = frozenset({
    "noreturn", "__noreturn__", "maybe_unused", "__maybe_unused__",
    "fallthrough", "__fallthrough__",
})


class AttributeMixin:

    def skip_balanced(self) -> None:
        self.expect("(")
        depth = 1
        while depth:
            if self.at_eof():
                self.error("unterminated parenthesis")
            t = self.next()
            if t.text == "(" and t.kind == TokenKind.PUNCT:
                depth += 1
            elif t.text == ")" and t.kind == TokenKind.PUNCT:
                depth -= 1

    def at_attribute(self) -> bool:
        return self.at("__attribute__") or self.at("__declspec")

    def at_std_attribute(self) -> bool:
        """C23 `[[...]]` attribute list (two consecutive `[` punctuators)."""
        return self.at("[") and self.peek().kind == TokenKind.PUNCT and self.peek().text == "["

    def skip_std_attributes(self) -> None:
        """Skip any C23 `[[...]]` attribute lists (no semantics applied).

        Standard attributes that take no arguments are rejected when written
        with an argument list: `[[noreturn()]]`, `[[fallthrough(1)]]`."""
        while self.at_std_attribute():
            self.next()
            self.next()
            depth = 2
            prev = "["
            while depth:
                if self.at_eof():
                    self.error("unterminated attribute list")
                t = self.next()
                if t.kind == TokenKind.PUNCT and t.text == "[":
                    depth += 1
                elif t.kind == TokenKind.PUNCT and t.text == "]":
                    depth -= 1
                elif (t.kind == TokenKind.IDENT and prev in ("[", ",", "::")
                        and t.text in _NO_ARG_STD_ATTRS
                        and self.at("(")):
                    self.error(f"'{t.text}' attribute does not take any arguments")
                elif (t.kind == TokenKind.IDENT and prev in ("[", ",", "::")
                        and self.at("(") and self.peek().kind == TokenKind.PUNCT
                        and self.peek().text == ")"):
                    self.error("parentheses must be omitted if attribute argument list is empty")
                elif (t.kind == TokenKind.IDENT and prev in ("[", ",", ":")
                        and t.text.strip("_") == "copy" and self.at("(")
                        and self.peek().kind == TokenKind.STRING):
                    self.error("'copy' attribute argument cannot be a string")
                prev = t.text

    def parse_attributes(self, attr=None, ty=None) -> None:
        while self.at_attribute() or self.at_std_attribute():
            if self.at_std_attribute():
                self.skip_std_attributes()
                continue
            if self.next().text == "__declspec":
                self.skip_balanced()
                continue
            self.expect("(")
            self.expect("(")
            while not self.at(")"):
                if self.accept(","):
                    continue
                name_tok = self.next()
                name = name_tok.text.strip("_") if name_tok.text.startswith("__") else name_tok.text
                args_start = self.pos
                if self.at("("):
                    self.skip_balanced()
                self._apply_attribute(name, args_start, attr, ty)
                # attributes in one list are separated by commas only
                if not self.at(",") and not self.at(")"):
                    self.error("expected ',' or ';' before '%s' token" % self.tok.text)
            self.expect(")")
            self.expect(")")

    def _attr_args(self, start: int) -> list:
        """Tokens between the parens starting at token index `start`."""
        if self.toks[start].text != "(":
            return []
        return self.toks[start + 1:self.pos - 1]

    def _apply_attribute(self, name, args_start, attr, ty) -> None:
        args = self._attr_args(args_start)
        if name == "mode" and args:
            mname = args[0].text.strip("_")
            if (args[0].kind != TokenKind.IDENT
                    or (mname not in MODE_SIZES and mname not in _OTHER_MODES)):
                self.error("unknown machine mode '%s'" % args[0].text,
                           self.toks[args_start - 1] if args_start > 0 else None)
        if name == "malloc":
            from hcc.parser.attr_malloc import check_malloc_attr
            check_malloc_attr(self, args_start, args)
        if name == "tls_model":
            self._check_tls_model(args, args_start)
        counts = ARG_COUNTS.get(name)
        if counts is not None:
            n = count_attr_args(args)
            if not counts[0] <= n <= counts[1]:
                self.error("wrong number of arguments specified for '%s' attribute" % name,
                           self.toks[args_start - 1] if args_start > 0 else None)
        if name == "scalar_storage_order" and args:
            if (args[0].kind != TokenKind.STRING
                    or args[0].value.rstrip(b"\0") not in (b"big-endian", b"little-endian")):
                self.error("attribute 'scalar_storage_order' argument must be one of "
                           "'big-endian' or 'little-endian'",
                           self.toks[args_start - 1] if args_start > 0 else None)
        if name == "copy" and args and args[0].kind == TokenKind.STRING:
            self.error("'copy' attribute argument cannot be a string",
                       self.toks[args_start - 1] if args_start > 0 else None)
        if name == "aligned":
            val = 16
            if args:
                save = self.pos
                self.pos = args_start + 1
                val = self.const_expr()
                self.pos = save
                self.check_align_limit(val)
            if attr is not None:
                attr.align = max(attr.align, val)
            elif ty is not None:
                ty.align = max(ty.align, val)
        elif name == "packed":
            if attr is not None:
                attr.is_packed = True
            if ty is not None:
                ty.is_packed = True
        elif name == "ms_struct" and attr is not None:
            attr.is_ms_struct = True
        elif name == "designated_init" and attr is not None:
            attr.is_designated_init = True
        elif name == "scalar_storage_order" and attr is not None and args:
            attr.reverse_sso = args[0].value.rstrip(b"\0") == b"big-endian"
        elif name == "warn_if_not_aligned" and attr is not None:
            attr.warn_if_not_aligned = True
        elif name == "noreturn" and attr is not None:
            attr.is_noreturn = True
        elif name == "nonnull" and attr is not None and not args:
            attr.nonnull_noargs = True
        elif name == "gnu_inline" and attr is not None:
            attr.is_gnu_inline = True
        elif name == "weak" and attr is not None:
            attr.is_weak = True
        elif name == "weakref" and attr is not None:
            attr.is_weakref = True
        elif name == "unavailable" and attr is not None:
            attr.is_unavailable = True
        elif name in RECORDED_FN_ATTRS and attr is not None:
            attr.fn_attrs.add(name)
        elif name == "copy" and attr is not None and args:
            # copy(f): inherit the attributes hcc records on the function f
            from hcc import nodes as N
            vs = self.find_var(args[0].text)
            if vs is not None and isinstance(vs.var, N.Function):
                attr.is_noreturn |= vs.var.is_noreturn
                attr.is_weak |= vs.var.is_weak
                # GCC does not copy `leaf` (it only describes extern functions)
                attr.fn_attrs |= vs.var.fn_attrs - {"leaf"}
            else:
                # copy(expr): inherit `packed` from the expression's member or type
                from hcc.parser.attr_copy import apply_copy_expr
                apply_copy_expr(self, args_start, attr)
        elif name == "cleanup" and attr is not None and args:
            attr.cleanup = args[0].text
        elif name == "nocommon" and attr is not None:
            attr.is_nocommon = True
        elif name == "section" and attr is not None and args:
            attr.section = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "visibility" and attr is not None and args:
            attr.visibility = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "alias" and attr is not None and args:
            attr.alias = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "ifunc" and attr is not None and args:
            attr.ifunc = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name in ("constructor", "destructor") and attr is not None:
            attr.init_kind = "init" if name == "constructor" else "fini"
            attr.init_prio = None
            if args:
                save = self.pos
                self.pos = args_start + 1
                attr.init_prio = self.const_expr()
                self.pos = save
        elif name == "vector_size" and attr is not None and args:
            save = self.pos
            self.pos = args_start + 1
            attr.vector_size = self.const_expr()
            self.pos = save
            # GCC: the byte size must fit ssizetype (LP64: 2**63-1)
            if attr.vector_size > VECTOR_SIZE_MAX:
                self.error("'vector_size' attribute argument value '%d' exceeds %d"
                           % (attr.vector_size, VECTOR_SIZE_MAX), self.toks[args_start])
        elif name == "strict_flex_array" and attr is not None:
            attr.strict_flex_array = True
        elif name == "mode" and attr is not None and args:
            attr.mode = args[0].text.strip("_")

    def check_designated_init(self, attr, is_struct: bool, where=None) -> None:
        """`designated_init` is only valid on a struct type (GCC: hard error)."""
        if attr is not None and attr.is_designated_init and not is_struct:
            self.error("'designated_init' attribute is only valid on 'struct' type", where)

    def check_align_limit(self, val, where=None) -> None:
        """Reject alignments above the x86-64 ELF object file maximum (2**28),
        as GCC does for objects, functions and typedefs alike (MAX_OFILE_ALIGNMENT)."""
        if val > OBJECT_FILE_MAX_ALIGN:
            self.error("requested alignment '%d' exceeds object file maximum %d"
                       % (val, OBJECT_FILE_MAX_ALIGN), where)

    def _check_tls_model(self, args, args_start) -> None:
        """tls_model("...") must be a string naming one of the four TLS models."""
        loc = self.toks[args_start - 1] if args_start > 0 else None
        if not args or args[0].kind != TokenKind.STRING:
            self.error("'tls_model' argument not a string", loc)
        elif args[0].value.rstrip(b"\0") not in TLS_MODELS:
            self.error("'tls_model' argument must be one of 'global-dynamic', "
                       "'local-dynamic', 'initial-exec' or 'local-exec'", loc)

    def typedef_aligned(self, ty, attr, dattr, name=None):
        """`typedef T X __attribute__((aligned(N)))`: raise the typedef's alignment.

        Alignment is sticky across redeclarations of the same typedef in the
        same scope (GCC keeps the largest), so a later plain `typedef T X;`
        does not drop an earlier aligned attribute.
        Returns a copy of `ty` (never mutates a shared type) when N is larger."""
        al = max(attr.align, dattr.align)
        if name is not None:
            prev = self.scopes[-1].vars.get(name)
            if prev is not None and prev.type_def is not None \
                    and prev.type_def.kind == ty.kind and prev.type_def.align > al:
                al = prev.type_def.align
        if al > ty.align:
            ty = ty.copy()
            ty.align = al
        return ty

    def var_aligned(self, ty, attr, dattr, name=None):
        """`T x __attribute__((aligned(N)))` on a variable: N sets the alignment
        exactly, so it may also lower it below the type's natural alignment
        (GCC: `int c __attribute__((aligned(2)))` gives _Alignof(c) == 2).
        `_Alignas(N)` may not reduce alignment (GCC: "cannot reduce alignment")."""
        if attr.alignas_min and attr.alignas_min < ty.align:
            self.error("'_Alignas' specifiers cannot reduce alignment of '%s'"
                       % (name.text if name is not None else "object"), name)
        al = max(attr.align, dattr.align)
        if 0 < al < ty.align:
            ty = ty.copy()
            ty.align = al
            return ty
        return self.typedef_aligned(ty, attr, dattr)

    def parse_asm_label(self, attr) -> None:
        """`asm("name")` following a declarator (glibc redirects)."""
        while self.at("asm"):
            self.next()
            self.expect("(")
            parts = []
            while not self.at(")"):
                t = self.next()
                if t.kind == TokenKind.STRING:
                    parts.append(self.asm_text(t))
            self.expect(")")
            if attr is not None:
                attr.asm_name = "".join(parts)
            self.parse_attributes(attr)
