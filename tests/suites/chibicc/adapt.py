"""Adapt chibicc's test/*.c so gcc -O0 passes: fix standard-mandated values,
and comment out (with a reason) ASSERTs relying on chibicc-specific/UB behaviour."""
import subprocess, re, os, sys
SP = sys.argv[1]  # scratch dir.  Run from the suite dir on pristine chibicc test/*.c
def edit(fn, old, new):
    s = open(fn).read(); assert old in s, (fn, old); open(fn, "w").write(s.replace(old, new, 1))
def comment_line(fn, idx, why):
    L = open(fn).read().split("\n")
    L[idx] = "/* hcc-tests removed (%s): %s */" % (why, L[idx].strip().replace("*/", "* /"))
    open(fn, "w").write("\n".join(L))
# Manual fixes first.
for a, b in [("501", "500"), ("801", "800"), ("2", "1"), ("201", "200")]:
    edit("line.c", "ASSERT(%s, __LINE__)" % a, "ASSERT(%s, __LINE__)" % b)  # #line N: next line is N
edit("macro.c", "#define M13 < include4.h\n#include M13 >\n  ASSERT(4, foo);",
     "/* hcc-tests removed: '#include < include4.h >' via macro; spacing is impl-defined */")
edit("pragma-once.c", '#include "test/pragma-once.c"', '#include "pragma-once.c"')
edit("compat.c", "_Noreturn noreturn_fn(int restrict x) {", "_Noreturn void noreturn_fn(int x) {")
edit("tls.c", "#include <stdio.h>\n", "")  # conflicts with test.h's printf(char*, ...)
edit("function.c", "typedef struct {\n  int gp_offset;", "#include <stdarg.h>\ntypedef struct {\n  int gp_offset;")
edit("function.c", "typedef __va_elem va_list[1];", "/* hcc-tests: use <stdarg.h> va_list */")
edit("function.c", "int vsprintf(char *buf, char *fmt, va_list ap);", "")
edit("function.c", "  *ap = *(__va_elem *)__va_area__;\n  vsprintf(buf, fmt, ap);",
     "  va_start(ap, fmt);\n  vsprintf(buf, fmt, ap);\n  va_end(ap);\n  return buf;")
edit("function.c", "inline int inline_fn(void) {", "static inline int inline_fn(void) {")
subprocess.run(["gcc", "-c", "-w", "-o", SP + "/common.o", "-xc", "support/common.c"], check=True)
exe = SP + "/t.exe"
for fn in sorted(f for f in os.listdir(".") if f.endswith(".c")):
    for it in range(60):
        r = subprocess.run(["gcc", "-w", "-fcommon", "-Isupport", "-o", exe, fn, SP + "/common.o", "-lm", "-pthread"],
                           capture_output=True, text=True)
        if r.returncode:
            m = re.search(r"^%s:(\d+):\d+: error: (.*)$" % re.escape(fn), r.stderr, re.M)
            if not m: print("UNHANDLED compile", fn, r.stderr[:300]); break
            comment_line(fn, int(m.group(1)) - 1, "gcc rejects: " + m.group(2)[:60]); print(fn, m.group(0)); continue
        r = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        if r.returncode == 0: break
        m = re.search(r"^(.*) => (-?\d+) expected but got (-?\d+)$", r.stdout, re.M)
        if not m: print("UNHANDLED run", fn, r.returncode, r.stdout[-300:]); break
        want = ("ASSERT(%s,%s" % (m.group(2), m.group(1))).replace(" ", "")
        L = open(fn).read().split("\n")
        cands = [i for i, l in enumerate(L) if not l.lstrip().startswith("/* hcc-tests") and want in l.replace(" ", "")]
        if not cands: print("NOLINE", fn, m.group(0)); break
        comment_line(fn, cands[0], "non-portable, gcc gives %s" % m.group(3)); print(fn, m.group(0))
