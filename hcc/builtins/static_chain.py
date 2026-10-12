"""__builtin_call_with_static_chain(call, chain): codegen/calls.py loads %r10."""
from __future__ import annotations

from hcc import nodes as N
from hcc.builtins import builtin_parser


@builtin_parser("__builtin_call_with_static_chain")
def _parse_call_with_static_chain(self, tok):
    # (call, chain): perform `call` with the static chain register (%r10) set to `chain`.
    args = self._args(2)
    call = args[0]
    if type(call) is not N.Call:
        self.error("first argument of __builtin_call_with_static_chain must be a call")
    slot = self.new_temp(args[1].ty, tok.loc)
    call.chain_slot = slot
    call.static_chain = self.typed(N.Assign(tok.loc, self.typed(N.VarRef(tok.loc, slot)), args[1]))
    return call
