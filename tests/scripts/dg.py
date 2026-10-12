"""Minimal DejaGnu directive reader for the GCC testsuite importers
(import_gcc_compile.py, import_gcc_dg.py).

It understands just enough of dg-* to decide, for an x86_64-linux-gnu host with a
gcc -O0 reference compiler, whether a test applies and which flags it needs:

    directives(text)       -> [(name, [tcl words...])]   e.g. ('dg-do', ['run', ['target', ...]])
    selector_true(words)   -> True/False for a `{ target ... }` / dg-skip-if selector
                              (raises Unknown for a feature we do not know)
    options(directives)    -> (kept_flags, skip_reasons)
    skip_if(directives, flags) -> reason or None
"""
import fnmatch
import re

TRIPLE = "x86_64-pc-linux-gnu"


class Unknown(Exception):
    """A dg selector/feature this reader does not know: the importer skips the test."""


# ----------------------------------------------------------------- Tcl lists

def tcl_split(s):
    """Split a Tcl list into words; {braced} words become nested lists (parsed
    recursively), "quoted" words become strings."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            out.append(tcl_split(s[i + 1:j - 1]))
            i = j
        elif c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 2 if s[j] == "\\" else 1
            out.append(s[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not s[j].isspace():
                j += 1
            out.append(s[i:j])
            i = j
    return out


def directives(text):
    """All `{ dg-name args... }` directives of a source file, in order."""
    res = []
    for m in re.finditer(r"\{\s*(dg-[\w-]+)", text):
        depth, j = 1, m.start() + 1
        n = len(text)
        while j < n and depth:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == '"':  # quoted strings may contain braces
                k = text.find('"', j + 1)
                if k != -1:
                    j = k
            j += 1
        body = text[m.start() + 1:j - 1]
        body = re.sub(r"\n\s*(\*|//)?", " ", body)  # comment decoration in multi-line directives
        words = tcl_split(body)
        if words:
            res.append((words[0], words[1:]))
    return res


# ----------------------------------------------------------------- selectors

# Effective-target keywords that are true for gcc on x86_64-linux-gnu.
TRUE = set("""lp64 int128 int32 int32plus ptr32plus size20plus size32plus x86 x86_64 fpic pic pie nonpic
indirect_calls indirect_jumps label_values nonlocal_goto named_sections exceptions return_address
scheduling non_strict_prototype alias weak visibility stdint_types asm_goto_with_outputs gnu_retain
c99_runtime freestanding hosted native default_packed c tls tls_native tls_runtime
long_double large_long_double float128 float80 double64 double64plus wchar ucn
trampolines untyped_assembly run_expensive_tests""".split())
# Keywords that are false here.
FALSE = set("""ia32 llp64 ilp32 mips arm aarch64 powerpc s390 avr pdp11 bpf nvptx rx
fopenmp fgraphite lto plugin dfp dfprt arm_arch_v5t_thumb_ok thumb1 thumb2 aarch64_sve
x32 newlib vxworks_kernel int16 size16 cygwin_or_mingw""".split())

SKIP_REASON = {
    "trampolines": "nested functions with trampolines (GCC-only)",
    "untyped_assembly": "__builtin_apply/untyped calls (GCC-only)",
    "run_expensive_tests": "expensive test",
}
X86_ISA = re.compile(r"(avx\w*|sse\w*|mmx|bmi\w*|fma\w*|xop|aes|pclmul|vaes|popcnt|f16c|rdrand|"
                     r"ssse3|amx\w*|apx\w*|vpclmul\w*|gfni\w*|lzcnt|rdseed|tbm|lwp|clflushopt|clwb|"
                     r"xsave\w*|sha|adx|prfchw|movbe|fsgsbase|rtm|hle|cet\w*)")


def _feature(name):
    if name in TRUE:
        return True
    if name in FALSE or X86_ISA.fullmatch(name):
        return False
    raise Unknown(name)


def _is_triple(w):
    return "*" in w or "?" in w or re.match(r"^[\w.]+-[\w.]+-[\w.-]+$", w) is not None


def _eval(words):
    """words: list of atoms / nested lists / '!' '&&' '||' operators."""
    for op in ("||", "&&"):
        if op in words:
            parts, cur = [], []
            for w in words:
                if w == op:
                    parts.append(cur)
                    cur = []
                else:
                    cur.append(w)
            parts.append(cur)
            vals = [_eval(p) for p in parts]
            return any(vals) if op == "||" else all(vals)
    if not words:
        return True
    if words[0] == "!":
        return not _eval(words[1:])
    if len(words) == 1:
        w = words[0]
        if isinstance(w, list):
            return _eval(w)
        if _is_triple(w):
            return fnmatch.fnmatchcase(TRIPLE, w) or fnmatch.fnmatchcase("x86_64-linux-gnu", w)
        return _feature(w)
    return any(_eval([w]) for w in words)  # a plain list of triples: any matches


def selector_true(sel):
    return _eval([sel] if isinstance(sel, str) else sel)


def split_target(args):
    """Split `run { target X }` style args into (plain args, target selector or None)."""
    sel, out = None, []
    for a in args:
        if isinstance(a, list) and a and a[0] in ("target", "xfail"):
            if a[0] == "target":
                sel = a[1:]
            continue
        out.append(a)
    return out, sel


# ------------------------------------------------------------------ options

KEEP_FLAG = re.compile(r"^(-std=(gnu|c)\w+|-ansi|-fgnu89-inline|-fwrapv|-fno-strict-overflow|-funsigned-char|"
                       r"-fsigned-char|-fpermissive|-fno-builtin|-ffreestanding|-fno-common|-fcommon|"
                       r"-fno-asm|-fms-extensions|-fplan9-extensions|-fno-short-enums|-fshort-enums|"
                       r"-fpack-struct(=\d+)?|-fno-pic|-fno-pie|-fno-strict-aliasing|-fshort-wchar|"
                       r"-fallow-store-data-races|-D[A-Za-z_]\w*(=\S*)?|-U\w+|-fno-trapping-math|"
                       r"-fno-math-errno|-ftrapping-math|-fno-builtin-\w+|-trigraphs)$")
# options that make the test unsuitable for a plain `cc -S` comparison
SKIP_FLAG = [
    (re.compile(r"^-m(?!(32|64)$)\S+$"), "target-specific option {}"),
    (re.compile(r"^-(fgimple|fopenmp\S*|fsanitize\S*|fprofile\S*|fplugin\S*|ftest-coverage|"
                r"fstack-check\S*|fstack-clash\S*|fsplit-stack|flto\S*|fcf-protection\S*|fanalyzer\S*|"
                r"fpatchable\S*|fzero-call\S*|fharden\S*|fvtable\S*|ftls-model\S*|fgraphite\S*|"
                r"fno-exceptions|fexceptions|fnon-call-exceptions|save-temps|E|M|MM|MD|MMD|"
                r"nostdlib\S*|fpreprocessed|fdirectives-only)$"),
     "needs gcc-only/unsupported option {}"),
]


def options(dirs):
    """Flags from dg-options / dg-additional-options whose target guard holds.
    Returns (kept flags, skip reasons)."""
    kept, reasons = [], []
    for name, args in dirs:
        if name not in ("dg-options", "dg-additional-options"):
            continue
        a, sel = split_target(args)
        try:
            if sel is not None and not selector_true(sel):
                continue
        except Unknown as e:
            reasons.append("dg option guard uses unknown feature %s" % e)
            continue
        if not a or not isinstance(a[0], str):
            continue
        for f in a[0].split():
            for rx, why in SKIP_FLAG:
                if rx.match(f):
                    reasons.append(why.format(f))
                    break
            else:
                if KEEP_FLAG.match(f):
                    kept.append(f)
    return kept, reasons


def skip_if(dirs, flags):
    """Evaluate dg-skip-if / dg-require-* / dg-do target guards for plain gcc -O0 on
    x86_64-linux.  Returns a skip reason, or None if the test applies."""
    optstr = " ".join(["-O0"] + list(flags))
    for name, args in dirs:
        try:
            if name == "dg-require-effective-target" and args and isinstance(args[0], str):
                guard = args[1:]
                if guard and isinstance(guard[0], list) and guard[0][:1] == ["target"] \
                        and not selector_true(guard[0][1:]):
                    continue
                feat = args[0]
                if feat in SKIP_REASON or not _feature(feat):
                    return "requires %s: %s" % (feat, SKIP_REASON.get(feat, "not available on x86_64-linux/gcc -O0"))
            elif name == "dg-skip-if" and len(args) >= 2:
                if not selector_true(args[1]):
                    continue
                inc = args[2] if len(args) > 2 and isinstance(args[2], list) else ["*"]
                exc = args[3] if len(args) > 3 and isinstance(args[3], list) else []

                def hit(pats):
                    return any(p == "*" or (p and fnmatch.fnmatchcase(" %s " % optstr, "* %s *" % p))
                               for p in pats if isinstance(p, str))
                if hit(inc) and not hit(exc):
                    return "dg-skip-if applies here (%s)" % (args[0] or "no reason given")
            elif name in ("dg-require-profiling", "dg-require-dll", "dg-require-stack-check",
                          "dg-require-lto", "dg-require-plugin", "dg-require-openmp"):
                return "requires %s" % name[len("dg-require-"):]
            elif name == "dg-do" and len(args) > 1:
                _, sel = split_target(args)
                if sel is not None and not selector_true(sel):
                    return "dg-do restricted to another target"
        except Unknown as e:
            return "dg selector uses unknown feature %s" % e
    return None
