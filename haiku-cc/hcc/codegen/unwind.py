"""Stack unwinding for __attribute__((cleanup(f))): CFI, call-site table and landing pads.

So that _Unwind_ForcedUnwind / _Unwind_RaiseException (and signal frames) run the
cleanups of the frames they pass, every function gets .cfi directives (frame is
rbp-based: CFA = rbp+16).  A function that has pending cleanups at some call
gets a gcc-style personality (__gcc_personality_v0) and LSDA:

* every `call` emitted for the function is bracketed by labels and recorded
  (unwind_call), so the call-site table covers all of its call instructions;
* a call made while cleanups are pending jumps to a landing pad (one per distinct
  list of pending cleanup calls) which saves %rax (the exception pointer),
  resets %rsp to the frame, runs those cleanups innermost first and calls
  _Unwind_Resume.  While cleanup j runs, only the cleanups below it are pending.
"""
from __future__ import annotations


class UnwindMixin:

    def unwind_begin(self) -> None:
        self.unwind_sites = []
        self.unwind_pads = {}
        self.unwind_override = None
        self.directive(".cfi_startproc")

    def unwind_active(self) -> list:
        if self.unwind_override is not None:
            return self.unwind_override
        return self.cleanup_stack

    def unwind_call(self, line: str) -> None:
        """Emit a `call` bracketed by labels, recording its call-site range and pad."""
        start = self.new_label()
        end = self.new_label()
        self.label(start)
        self.out.append("  " + line)
        self.label(end)
        prefix = list(self.unwind_active())
        pad = None
        if prefix:
            key = tuple(id(c) for c in prefix)
            if key not in self.unwind_pads:
                self.unwind_pads[key] = (self.new_label(), prefix)
            pad = self.unwind_pads[key][0]
        self.unwind_sites.append((start, end, pad))

    def unwind_finish(self, sym: str, size: int) -> None:
        """Emit the landing pads, LSDA and personality of the function just generated,
        then close its CFI procedure."""
        done = set()
        while True:
            todo = [k for k in self.unwind_pads if k not in done]
            if not todo:
                break
            done.add(todo[0])
            self.emit_landing_pad(*self.unwind_pads[todo[0]], size)
        if self.unwind_pads:
            self.emit_lsda(sym)
        self.directive(".cfi_endproc")
        self.unwind_override = None

    def emit_landing_pad(self, label: str, prefix: list, size: int) -> None:
        self.label(label)
        self.depth = 0
        self.frame_lea(-size, "%rsp")          # the unwinder's %rsp may hold temporaries
        self.emit("sub $16, %rsp")             # keeps %rsp 16-byte aligned for the calls
        self.emit("mov %rax, (%rsp)")          # the _Unwind_Exception pointer
        for j in reversed(range(len(prefix))):
            self.unwind_override = list(prefix[:j])
            self.gen_expr(prefix[j])
            self.discard_value(prefix[j].ty)
        self.unwind_override = None
        self.emit("mov (%rsp), %rdi")
        self.directive("call _Unwind_Resume")  # not a recorded site: it does not return
        self.directive("ud2")

    def emit_lsda(self, sym: str) -> None:
        """Personality pointer (indirect, local data), call-site table in
        .gcc_except_table and the .cfi_personality / .cfi_lsda that name them."""
        lsda = self.new_label()
        cst_s = self.new_label()
        cst_e = self.new_label()
        pers = self.new_label()                # local: one copy per function, no clash
        self.directive('.pushsection .data.rel.local,"aw"')
        self.directive(".align 8")
        self.label(pers)
        self.directive(".quad __gcc_personality_v0")
        self.directive(".popsection")
        self.directive(f".cfi_personality 0x9b, {pers}")
        self.directive(f".cfi_lsda 0x1b, {lsda}")
        self.directive('.pushsection .gcc_except_table,"a",@progbits')
        self.label(lsda)
        self.directive(".byte 0xff")           # @LPStart omitted: pads relative to the function
        self.directive(".byte 0xff")           # @TType omitted: no type table
        self.directive(".byte 0x03")           # call-site fields: udata4
        self.directive(f".uleb128 {cst_e} - {cst_s}")
        self.label(cst_s)
        for start, end, pad in self.unwind_sites:
            self.directive(f".long {start} - {sym}")
            self.directive(f".long {end} - {start}")
            self.directive(f".long {pad} - {sym}" if pad else ".long 0")
            self.directive(".uleb128 0")       # no action: run the cleanups only
        self.label(cst_e)
        self.directive(".popsection")
