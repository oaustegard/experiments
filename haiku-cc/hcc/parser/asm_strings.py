"""Decoding of string literals inside asm statements and asm labels.

GCC only accepts narrow (or u8) string literals there; a wide literal is
diagnosed as "a wide string is invalid in this context".
"""


class AsmStringMixin:
    def asm_text(self, t) -> str:
        """Return the narrow text of string token `t` used in an asm context."""
        if t.ty is not None and t.ty.base is not None and t.ty.base.size != 1:
            self.error("a wide string is invalid in this context", t)
        return t.value.rstrip(b"\0").decode("utf-8", "replace")
