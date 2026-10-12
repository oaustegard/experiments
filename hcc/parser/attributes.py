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
        """Skip any C23 `[[...]]` attribute lists (no semantics applied)."""
        while self.at_std_attribute():
            self.next()
            self.next()
            depth = 2
            while depth:
                if self.at_eof():
                    self.error("unterminated attribute list")
                t = self.next()
                if t.kind == TokenKind.PUNCT and t.text == "[":
                    depth += 1
                elif t.kind == TokenKind.PUNCT and t.text == "]":
                    depth -= 1

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
            self.expect(")")
            self.expect(")")

    def _attr_args(self, start: int) -> list:
        """Tokens between the parens starting at token index `start`."""
        if self.toks[start].text != "(":
            return []
        return self.toks[start + 1:self.pos - 1]

    def _apply_attribute(self, name, args_start, attr, ty) -> None:
        args = self._attr_args(args_start)
        if name == "aligned":
            val = 16
            if args:
                save = self.pos
                self.pos = args_start + 1
                val = self.const_expr()
                self.pos = save
            if attr is not None:
                attr.align = max(attr.align, val)
            elif ty is not None:
                ty.align = max(ty.align, val)
        elif name == "packed":
            if attr is not None:
                attr.is_packed = True
            if ty is not None:
                ty.is_packed = True
        elif name == "scalar_storage_order" and attr is not None and args:
            attr.reverse_sso = args[0].value.rstrip(b"\0") == b"big-endian"
        elif name == "noreturn" and attr is not None:
            attr.is_noreturn = True
        elif name == "weak" and attr is not None:
            attr.is_weak = True
        elif name == "section" and attr is not None and args:
            attr.section = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "visibility" and attr is not None and args:
            attr.visibility = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "alias" and attr is not None and args:
            attr.alias = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "ifunc" and attr is not None and args:
            attr.ifunc = args[0].value.rstrip(b"\0").decode() if args[0].value else None
        elif name == "vector_size" and attr is not None and args:
            save = self.pos
            self.pos = args_start + 1
            attr.vector_size = self.const_expr()
            self.pos = save
        elif name == "mode" and attr is not None and args:
            attr.mode = args[0].text.strip("_")

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

    def parse_asm_label(self, attr) -> None:
        """`asm("name")` following a declarator (glibc redirects)."""
        while self.at("asm"):
            self.next()
            self.expect("(")
            parts = []
            while not self.at(")"):
                t = self.next()
                if t.kind == TokenKind.STRING:
                    parts.append(t.value.rstrip(b"\0").decode("utf-8", "replace"))
            self.expect(")")
            if attr is not None:
                attr.asm_name = "".join(parts)
            self.parse_attributes(attr)
