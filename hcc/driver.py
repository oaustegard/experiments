"""Command-line driver (gcc-compatible subset, CONTRACT section 8).

main(argv) -> exit status. Stages: preprocess -> parse -> codegen ->
assemble (`as`) -> link (`gcc -no-pie`, linker driver only).
Exit codes: 0 ok, 1 compile/assemble/link error, 70 internal compiler error.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback

from hcc import diag
from hcc.options import Options

# flags that take a separate argument we ignore or pass to the linker
_IGNORED_WITH_ARG = {"-MF", "-MT", "-MQ", "-arch", "-target", "--param", "-aux-info"}
_LINKER_WITH_ARG = {"-L", "-l", "-Xlinker", "-T", "-u", "-z"}
# -f flags that gcc reflects as predefined macros (non-default settings only)
_MATH_FLAG_MACROS = {"-fno-math-errno": "__NO_MATH_ERRNO__",
                     "-fassociative-math": "__ASSOCIATIVE_MATH__",
                     "-fno-trapping-math": "__NO_TRAPPING_MATH__",
                     "-freciprocal-math": "__RECIPROCAL_MATH__",
                     "-frounding-math": "__ROUNDING_MATH__",
                     "-fno-signed-zeros": "__NO_SIGNED_ZEROS__"}


def parse_args(argv: list[str]) -> Options:
    o = Options()
    i = 0
    n = len(argv)

    def arg_value(flag):
        nonlocal i
        a = argv[i]
        if len(a) > len(flag):
            return a[len(flag):]
        i += 1
        if i >= n:
            raise SystemExit(f"hcc: error: missing argument to '{flag}'")
        return argv[i]

    while i < n:
        a = argv[i]
        if a == "-c":
            o.mode = "obj"
        elif a == "-S":
            o.mode = "asm"
        elif a == "-E":
            o.mode = "preprocess"
        elif a.startswith("-o"):
            o.output = arg_value("-o")
        elif a.startswith("-I"):
            o.include_dirs.append(arg_value("-I"))
        elif a.startswith("-iquote"):
            o.quote_dirs.append(arg_value("-iquote"))
        elif a.startswith("-isystem"):
            o.system_dirs.append(arg_value("-isystem"))
        elif a.startswith("-idirafter"):
            o.after_dirs.append(arg_value("-idirafter"))
        elif a.startswith("-D"):
            o.defines.append(("-D", arg_value("-D")))
        elif a.startswith("-U"):
            o.defines.append(("-U", arg_value("-U")))
        elif a == "-include":
            i += 1
            o.includes.append(argv[i])
        elif a.startswith("-std="):
            o.std = a[5:]
        elif a == "-ansi":
            o.std = "c89"
        elif a == "-trigraphs":
            o.trigraphs = True
        elif a == "-fgnu89-inline":
            o.gnu89_inline = True
        elif a == "-fno-asm":
            o.no_asm = True
        elif a == "-nostdinc":
            o.nostdinc = True
        elif a in ("-fPIC", "-fpic", "-fPIE", "-fpie"):
            o.pic = a in ("-fPIC", "-fpic")
        elif a in _MATH_FLAG_MACROS:
            o.defines.append(("-D", _MATH_FLAG_MACROS[a]))
        elif a == "-x":
            i += 1
            o.language = argv[i]
        elif a == "-dump-ast":
            o.dump_ast = True
        elif a == "-dump-tokens":
            o.dump_tokens = True
        elif a in ("-static", "-shared", "-pthread", "-rdynamic", "-nostdlib", "-nostartfiles") \
                or a.startswith(("-Wl,", "-l", "-L")):
            if a in ("-l", "-L"):
                i += 1
                o.linker_args.append(a + argv[i])
            else:
                o.linker_args.append(a)
        elif a in ("-Xlinker", "-T", "-u", "-z"):
            o.linker_args += [a, argv[i + 1]]
            i += 1
        elif a in _IGNORED_WITH_ARG:
            i += 1
        elif a.startswith("-M") and a in ("-MF", "-MT", "-MQ"):
            i += 1
        elif a.startswith(("-O", "-g", "-W", "-w", "-f", "-m", "-M", "-pedantic", "-pipe", "-v",
                           "-ffreestanding", "--")) or a in ("-p", "-pg"):
            pass
        elif a == "-" or not a.startswith("-"):
            o.inputs.append(a)
        else:
            print(f"hcc: warning: ignoring unknown option '{a}'", file=sys.stderr)
        i += 1
    o.keep_temps = bool(os.environ.get("HCC_KEEP_TEMPS"))
    return o


def preprocess_file(path: str, opts: Options):
    from hcc.preprocess import Preprocessor
    with diag.stage("preprocess"):
        pp = Preprocessor(opts)
        if path == "-":
            return pp.run_text(sys.stdin.read(), "<stdin>")
        return pp.run(path)


def compile_to_asm(path: str, opts: Options) -> str:
    from hcc.codegen import generate
    from hcc.literals import convert_tokens
    from hcc.parser import parse
    toks = preprocess_file(path, opts)
    with diag.stage("parse"):
        c23 = (opts.std or "").lower() in ("c23", "gnu23", "c2x", "gnu2x")
        # `asm` and `typeof` are extension keywords: off in strict ISO modes and under -fno-asm.
        strict = not (opts.std or "gnu17").lower().startswith("gnu")
        off = frozenset(("asm", "typeof") if strict or opts.no_asm else ())
        toks = convert_tokens(toks, u8_unsigned=c23, c23=c23, off_keywords=off)
        tu = parse(toks, opts)
    if opts.dump_ast:
        for o in tu.objs:
            print(o, file=sys.stderr)
    with diag.stage("codegen"):
        return generate(tu, opts)


def run_tool(cmd: list[str], stage: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()
        first = msg[0] if msg else f"{cmd[0]} failed"
        sys.stderr.write((r.stderr or r.stdout))
        raise diag.CompileError(None, f"{cmd[0]} failed: {first}", stage)


def assemble(asm_path: str, obj_path: str) -> None:
    run_tool(["as", "--64", "-o", obj_path, asm_path], "assemble")


def _out_name(src: str, ext: str) -> str:
    base = os.path.basename(src)
    return os.path.splitext(base)[0] + ext


def main(argv: list[str]) -> int:
    opts = parse_args(argv)
    if not opts.inputs:
        print("hcc: fatal error: no input files", file=sys.stderr)
        return 1
    stage = "preprocess"
    tmpdir = tempfile.mkdtemp(prefix="hcc-")
    try:
        if opts.mode == "preprocess":
            from hcc.preprocess import format_tokens
            out = []
            for src in opts.inputs:
                out.append(format_tokens(preprocess_file(src, opts)))
            text = "".join(out)
            if opts.output and opts.output != "-":
                with open(opts.output, "w") as f:
                    f.write(text)
            else:
                sys.stdout.write(text)
            return 0
        objs: list[str] = []
        sources = [s for s in opts.inputs]
        if opts.output and len([s for s in sources if _is_source(s, opts)]) > 1 and opts.mode != "link":
            print("hcc: fatal error: cannot specify -o with -c or -S with multiple files", file=sys.stderr)
            return 1
        for src in sources:
            if not _is_source(src, opts) and not src.endswith((".s", ".S")):
                objs.append(src)          # .o / .a / .so -> linker
                continue
            if src.endswith(".s"):
                asm = open(src).read()
            elif src.endswith(".S"):
                from hcc.preprocess import format_tokens
                asm = format_tokens(preprocess_file(src, opts))
            else:
                stage = "parse"
                asm = compile_to_asm(src, opts)
            if opts.mode == "asm":
                out = opts.output or _out_name(src, ".s")
                if out == "-":
                    sys.stdout.write(asm)
                else:
                    with open(out, "w") as f:
                        f.write(asm)
                continue
            stage = "assemble"
            asm_path = os.path.join(tmpdir, _out_name(src, ".s"))
            with open(asm_path, "w") as f:
                f.write(asm)
            if opts.keep_temps:
                with open(_out_name(src, ".hcc.s"), "w") as f:
                    f.write(asm)
            if opts.mode == "obj":
                obj = opts.output or _out_name(src, ".o")
            else:
                obj = os.path.join(tmpdir, f"{len(objs)}_" + _out_name(src, ".o"))
            assemble(asm_path, obj)
            objs.append(obj)
        if opts.mode == "link":
            stage = "link"
            out = opts.output or "a.out"
            pie = [] if "-shared" in opts.linker_args else ["-no-pie"]
            # -lm last: libm is not pulled in by default, but C programs call its functions.
            run_tool(["gcc", *pie, "-o", out, *objs, *opts.linker_args, "-lm"], "link")
        return 0
    except diag.CompileError as e:
        if e.stage is None:
            e.stage = stage
        print(e.format(), file=sys.stderr)
        return 1
    except RecursionError:
        print(f"hcc: internal compiler error [{stage}]: recursion too deep", file=sys.stderr)
        return 70
    except Exception as e:  # noqa: BLE001 - report ICE with stage
        print(f"hcc: internal compiler error [{stage}]: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 70
    finally:
        if not opts.keep_temps:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def _is_source(path: str, opts: Options) -> bool:
    if opts.language == "c" or path == "-":
        return True
    return path.endswith((".c", ".i", ".h"))
