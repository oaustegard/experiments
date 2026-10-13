# Pilot task: matching decompilation at three optimisation levels

You reconstruct C source for 4 functions of one translation unit (TU) so that gcc,
with pinned flags, produces the SAME machine code bytes as the original object.
You are one of 5 pilot workers; the results decide which optimisation level a later
swarm uses, so report honestly.

There are three copies of the project, one per optimisation level:
  /home/user/swarm-decomp-pilot/O0   (gcc -O0)
  /home/user/swarm-decomp-pilot/O1   (gcc -O1)
  /home/user/swarm-decomp-pilot/O2   (gcc -O2)
Each has the same layout. Read `CONTRACT.md` in any of them first: it explains the
layout, the one-file-per-function rule, and the commands.

For your TU (`<suite>` = `<proj>-<tu>`) each tree has:
  tests/suites/<suite>/<function>    disassembly listing of the target function at that level
  tests/suites/<suite>/_info.txt     all functions of the TU, data objects, external symbols
  tests/suites/<suite>/target.o      the original object (objdump -d -r works on it)

Procedure:
1. Read CONTRACT.md and your TU's `_info.txt` and listings at -O0 (the most literal
   translation of the source) to work out the structs and what each function does.
2. Write the types in `src/<proj>/types/` and each function in `src/<proj>/<tu>/<function>.c`
   in the O0 tree. Check with:
   cd /home/user/swarm-decomp-pilot/O0 && python3 tests/run.py --suite <suite> -v --filter '<suite>/<function>'
3. Copy your `src/` to the O1 and O2 trees and check there too. Iterate on mismatches at
   each level (the diff shows target lines with `-` and yours with `+`). You may change the
   source per level, but the same source matching at every level is the ideal.
4. Spend roughly equal effort per level. Stop after about 70 tool calls in total.
5. Write your result as JSON (Write tool) to /home/user/swarm-decomp-pilot/results/<suite>.json:
   {"suite": "<suite>",
    "functions": {"<function>": {"O0": {"status": "PASS|FAIL", "frac": 0.0, "reason": "..."},
                                 "O1": {...}, "O2": {...}}},
    "notes": "what made matching hard at each level, one or two sentences"}
   Take status/frac/reason from the runner's final output for each level (run
   `python3 tests/run.py --suite <suite> --json /tmp/<suite>-<level>.json` in each tree at the end).

Rules: never use the network (no curl, wget, git clone, pip, web lookups) and never look
for the original source anywhere; write the code from the disassembly. Edit only files
under the three trees' `src/` directories and your results file. No inline assembly.
Never edit `tests/`. Other TUs' functions are out of scope: declare (prototype) what you
call, don't implement it.
