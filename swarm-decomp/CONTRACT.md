# swarm-decomp contract

The goal is C source that compiles, with the pinned compiler and flags, to the same
code bytes as each function of the target objects. You get the target objects
(symbol table kept, no debug info), a disassembly listing per function, and for Lua
its installed public headers. You never get, and must never fetch, the original
source.

## Layout

```
tests/suites/<proj>-<tu>/target.o     the original object for one translation unit (read only)
tests/suites/<proj>-<tu>/<function>   listing: AT&T disassembly, relocation targets resolved
tests/suites/<proj>-<tu>/_info.txt    functions in definition order, data objects with their bytes,
                                      symbols the TU uses from elsewhere
tests/include/lua/                    Lua's installed headers: lua.h luaconf.h lualib.h lauxlib.h
tests/flags.json                      the pinned compiler and flags (do not edit)
src/<proj>/types/<name>.h             shared declarations, ONE per file
src/<proj>/<tu>/<function>.c          ONE function definition per file
src/<proj>/<tu>/<variable>.c          ONE file-scope variable definition per file
src/<proj>/<tu>/<function>.h          optional: declarations only that TU needs
```

Projects: `neatroff neatpost neatmkfn neateqn neatrefer neatcc neatld neatmail` (one
author's troff suite and C compiler; they share no headers, so a struct named the same
in two projects is two separate types) and `lua` (Lua 5.4.7, built with
`-DLUA_COMPAT_5_3 -DLUA_USE_LINUX`; those defines are passed for you).

## How a translation unit is built

`tests/run.py` generates `<proj>-<tu>.c` as a list of `#include`s:

1. every `src/<proj>/types/*.h`, in byte order of the file name;
2. every `src/<proj>/<tu>/*.h`, in byte order;
3. every `src/<proj>/<tu>/*.c` that is not a stub: names the target does not define
   (inlined static helpers, variables) first, alphabetically, then the functions in the
   target's definition order.

It compiles that with `-I src/<proj>/types -I src/<proj>/<tu>` (plus
`-I tests/include/lua` for Lua) and the flags in `tests/flags.json`, then compares
every function symbol. A file containing `TODO(decomp)` is a stub and is skipped.

## Rules

- **Add a file, don't edit a shared list.** A new struct, typedef, enum, macro group
  or extern prototype is a new file in `src/<proj>/types/`, named after it
  (`types/sbuf.h` for `struct sbuf`). Start it with `#pragma once` and `#include` the
  other type files (and system headers) it needs, so the include order never matters.
  A file that must come before everything else (feature-test macros such as
  `_POSIX_C_SOURCE`) is named with a leading `0_`.
- Editing an existing type file is allowed when the type is wrong (a field missing, at
  the wrong offset, the wrong width). Every TU of the project includes it, so run the
  whole project's suites before and after (see below).
- A prototype for a non-static function goes in `types/`; a prototype for a static
  function in `src/<proj>/<tu>/<function>.h`.
- One function per `.c` file, named exactly as the function (`num_set.c`). A static
  helper gcc inlined into its callers (it has no listing) still gets its own file.
  gcc clones such as `foo.part.0`, `foo.isra.0`, `foo.constprop.0` are produced from
  your `foo`; write `foo.c`, never a file named after the clone.
- Match the target's linkage: a function the listing header calls `static` must be
  `static`.
- Never special-case the test machinery, never edit `tests/`, never copy the listing
  into an `asm` block. Inline assembly is rejected as a solution.
- No attributes or pragmas that steer gcc: `used`, `noinline`, `noipa`, `noclone`,
  `optimize`, `always_inline`, `flatten`, `section`, `#pragma GCC`. The runner fails the
  TU on them. A static function gcc drops because nothing calls it yet is not wrong:
  write its callers (they may be outside your assigned functions) or leave it for the
  worker who writes them.
- No network: no `curl`, `wget`, `git clone`, package downloads or web lookups. Write
  the code from the disassembly.

## Commands

```bash
python3 tests/run.py --suite neatroff-reg -v --filter 'neatroff-reg/num_set'   # one function, with a diff
python3 tests/run.py --suite neatroff-reg --json /tmp/x/before.json            # the whole TU
python3 tests/run.py --suite neatroff-reg --baseline /tmp/x/before.json        # regressions since then
python3 tests/run.py --suite "$(ls tests/suites | grep '^neatroff-' | paste -sd,)" --json /tmp/x/proj.json
objdump -d -r tests/suites/neatroff-reg/target.o | less                         # raw bytes, if needed
```

Stages and reasons in the results:

| stage | reason | meaning |
|---|---|---|
| compile | `compile: <gcc's first error>` | the TU does not compile; every function in it fails |
| missing | `stub: not yet written in <suite>` | no definition yet |
| missing | `absent: ...` | written, but gcc emitted no code for it: a static function whose callers are still stubs, a function gcc inlined, or a clone gcc did not make |
| match | `regalloc` | same instructions, only registers differ |
| match | `offset: displacement 18to20 ...` | struct field offsets differ: a struct layout is wrong (the hex pairs are target to yours) |
| match | `frame` | stack frame size or local layout differs |
| match | `imm` | a constant differs |
| match | `call: expected call to X` / `unexpected call to X` | a call is missing or extra (often: inlined in one, called in the other) |
| match | `ref` | a global, string or static referenced is different |
| match | `cfg-layout` | same instructions, basic blocks in a different order |
| match | `insn-delta: near/mid/far` | instruction sequences differ; `frac` says how much matches |

`frac` is the fraction of matching instructions (1.0 = PASS).
