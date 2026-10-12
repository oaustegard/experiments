"""__builtin_ia32_* SSE/MMX intrinsics: signatures and parsing in
parser/x86_sse_sigs.py, table-driven lowering in codegen/x86_sse.py. A new
intrinsic is a line in SIG_TEXT; it is registered here from that table."""
from __future__ import annotations

from hcc.builtins import builtin_codegen, builtin_parser
from hcc.codegen.x86_sse import lower_x86
from hcc.parser.x86_sse_sigs import SIGS, X86_BUILTINS, x86_call


@builtin_parser(*sorted(X86_BUILTINS))
def _parse_x86(self, tok):
    return x86_call(self, tok)


@builtin_codegen(*sorted("ia32_" + n for n in SIGS))
def _gen_x86(self, node) -> None:
    lower_x86(self, node)
