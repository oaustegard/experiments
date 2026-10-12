# hcc Architecture Contract

hcc is a C11 compiler written in pure Python 3.12 (stdlib only). It emits
x86-64 System V AT&T assembly. The system `as` assembles it; `gcc` is used
**only** as the linker driver. hcc never calls gcc/cc1/cpp to preprocess,
parse or compile.

This document is the contract every contributor (human or agent) codes
against. Interfaces marked **STABLE** must not change without updating this
file in the same commit. Everything else may be refactored freely as long as
the stable interfaces keep working and `python3 -m hcc.selftest` passes.

---------------------------------------------------------------------------

## 0. Ground rules (read these first)

1. **One concern per file.** If you add a feature, find its home in the module
   map (section 2). If no home fits, create a new file next to the closest
   relative and add a line to the module map here.
2. **Files stay under 600 lines.** If a file would grow past that, split it by
   sub-concern (e.g. `codegen/calls.py` -> `codegen/calls_struct.py`).
3. **No global mutable state** except inside the documented context objects:
   `Preprocessor` (preprocess), `Parser` (parser + sema calls), `CodeGen`
   (codegen), `Options` (driver). Module-level constants (tables, frozen
   sets, singleton base types in `typesys.py`) are fine, but never mutate them —
   use `ty.copy()` first.
4. **Errors go through `hcc.diag`** (`diag.error(loc, msg)`), never `print` +
   `sys.exit`, never a bare `raise Exception`.
5. **Never break the selftests.** Run `python3 -m hcc.selftest` before every
   commit. Add a selftest (`hcc/selftest/<area>_<what>.c`) for every feature
   or bug fix.
6. **Don't reformat files you aren't changing.** Small diffs merge cleanly
   when 1000 agents work concurrently.
7. Dispatch is by **naming convention**, not big if/elif chains:
   codegen for AST class `Foo` lives in a method `expr_Foo` / `stmt_Foo` in
   the mixin file that owns that concern. To add a node: add the dataclass to
   `hcc/nodes.py`, have the parser build it, have `sema/typing.py` type it (if
   it's an expression), and add the `expr_Foo`/`stmt_Foo` method.

---------------------------------------------------------------------------

## 1. Pipeline

```
  source text
     │  hcc/lexer.py            tokenize(text, path) -> list[Token]   (pp-tokens)
     ▼
  hcc/preprocess/             Preprocessor(opts).run(path) -> list[Token]
     │                        (macro-expanded, directives removed, EOF last)
     ▼
  hcc/literals.py             convert_tokens(toks): keywords, numbers, strings,
     │                        adjacent string concatenation
     ▼
  hcc/parser/  (+ hcc/sema/)  parse(tokens) -> ast.TranslationUnit
     │                        Parser builds AST; every expression node is typed
     │                        immediately by sema.typing.add_type(); implicit
     │                        conversions are inserted as ast.Cast nodes.
     ▼
  hcc/codegen/                generate(tu, opts) -> str   (AT&T asm text)
     ▼
  hcc/driver.py               writes .s, runs `as`, links with `gcc -no-pie`
```

Stage names (used in diagnostics and by tests/run.py's `stage` field):
`preprocess`, `parse`, `sema`, `codegen`, `assemble`, `link`.

### Codegen design decision: direct AST -> asm stack machine (chibicc style)

* Every expression is evaluated into the accumulator: `%rax` for integers and
  pointers, `%xmm0` for float/double, x87 `st(0)` for long double. Binary
  operators evaluate rhs, `push`, evaluate lhs, `pop` into `%rdi`/`%xmm1`.
* Struct/union values are represented by their **address** in `%rax`.
* Every local variable lives in the stack frame at a fixed `%rbp` offset.

Why not an IR: for a swarm of small agents the most important property is
*locality* — fixing `x <<= y` on `unsigned char` should touch one function in
one file, testable in isolation. A stack machine has no register allocator,
no IR invariants, no pass ordering; each AST node maps to one self-contained
code template. Performance of generated code is explicitly a non-goal for v1.
An optimizing IR backend can be added later behind the same `generate(tu)`
interface without touching the front end.

---------------------------------------------------------------------------

## 2. Module map (where things go)

```
hcc/
  __main__.py          `python3 -m hcc` -> driver.main()
  driver.py            CLI flags (gcc-compatible subset), runs stages, `as`, link
  options.py           Options dataclass (include paths, defines, -std, output)
  diag.py              Loc-tagged errors/warnings, CompileError, stage tracking
  tokens.py            Loc, Token, TokenKind, KEYWORDS  (STABLE)
  lexer.py             text -> pp-tokens (regex based), line splicing
  literals.py          pp-number -> int/float literal + type, char/string
                       literal decoding (escapes, u8/u/U/L prefixes),
                       keyword marking, string concatenation
  typesys.py             Type, Member, TypeKind, LP64 base types, constructors (STABLE)
  nodes.py               all AST node dataclasses + Obj (Var/Function)        (STABLE)

  preprocess/
    __init__.py        Preprocessor class: main expansion loop, public API
    macros.py          Macro dataclass, argument collection, substitution,
                       stringify (#), paste (##), __VA_ARGS__/__VA_OPT__
    directives.py      #define #undef #include #include_next #if... #line
                       #error #warning #pragma
    conditionals.py    #if/#elif expression evaluation (defined, __has_*)
    includes.py        include search path resolution, file reading/caching
    predefined.py      predefined macros (__x86_64__, __LP64__, __GNUC__, ...)
                       and builtin dynamic macros (__FILE__, __LINE__, ...)

  parser/
    __init__.py        parse(tokens) and the Parser class (mixin assembly)
    core.py            ParserBase: token cursor, expect/accept, scopes,
                       labels, current-function context, temp vars
    scope.py           Scope, VarScope, TagScope objects
    declspec.py        declaration specifiers: base types, storage class,
                       qualifiers, typedef names, _Alignas, __typeof__
    declarator.py      declarators, abstract declarators, typename,
                       function params, array dims
    structs.py         struct/union declarations, member layout, bitfields
    enums.py           enum declarations
    attributes.py      GNU __attribute__((...)), __asm__("label"),
                       __extension__, __declspec skipping
    toplevel.py        translation unit, global declarations, function
                       definitions, typedef at file scope
    locals.py          block-scope declarations
    initializers.py    initializer parsing -> ast.Initializer tree
                       (designators, brace elision, string init)
    statements.py      all statements
    expressions.py     expression precedence climbing (comma..unary)
    postfix.py         postfix exprs: calls, [], ., ->, ++/--, compound lits
    primary.py         primary exprs: identifiers, literals, ( ), stmt-exprs,
                       _Generic, sizeof/_Alignof
    builtins.py        parsing of __builtin_* that take types or are special
                       (va_arg, offsetof, types_compatible_p, ...)
    atomic_builtins.py __atomic_* (stdatomic.h) parsing; codegen/atomics.py lowers them
    vector_shuffle.py  __builtin_shuffle on GCC vectors, lowered to element stores

  sema/
    __init__.py        re-exports
    typing.py          add_type(node): compute node.ty for every Expr kind
    conversions.py     integer promotion, usual arithmetic conversions,
                       new_cast(), implicit conversion on assign/return/args
    arith.py           new_add/new_sub (pointer arithmetic), compound
                       assignment and ++/-- lowering
    constexpr.py       integer constant evaluation, address constants
    initializers.py    Initializer -> local assignment exprs; Initializer ->
                       static bytes + relocations (GlobalData)
    compat.py          type compatibility (is_compatible)

  codegen/
    __init__.py        generate(tu, opts) -> str ; CodeGen class assembly
    bitfield_rev.py    bitfields in scalar_storage_order("big-endian") structs
    context.py         CodeGenBase: output buffer, emit(), labels,
                       push/pop with stack-depth tracking, current function
    functions.py       frame layout (local offsets), prologue/epilogue,
                       spilling incoming params (regs + stack)
    memory.py          load(ty)/store(ty) for every type, gen_addr (lvalues),
                       struct copy
    expressions.py     expr dispatch, literals, var refs, unary, comma, cond,
                       assign, logical ops
    arith.py           integer binary operators and comparisons
    floats.py          float/double/long double arithmetic & comparisons
    casts.py           conversions between all scalar types (cast table)
    calls.py           function calls: argument evaluation & placement
    abi.py             SysV ABI classification (INTEGER/SSE/MEMORY eightbytes)
                       for struct args/returns
    varargs.py         va_start/va_arg/va_copy, register save area
    statements.py      statements, switch dispatch, goto/labels
    data.py            .data/.bss/.rodata emission, string literals, relocs
    builtins.py        __builtin_* lowering (alloca, expect, clz, ...)
    atomics.py         __atomic_* lowering: lock cmpxchg loops, xchg, sized movs
    asm.py             inline asm statements (basic + extended)

  include/             (optional) hcc's own override headers, searched before
                       the system dirs (e.g. a future stdatomic.h)

  selftest/
    __main__.py        `python3 -m hcc.selftest [-k substr] [-j N]`
    *.c                tests; expectations in header comments (section 9)
```

---------------------------------------------------------------------------

## 3. Tokens, locations and diagnostics (STABLE)

```python
# hcc/tokens.py
class TokenKind(Enum): IDENT, KEYWORD, PUNCT, NUMBER (pp-number), STRING, CHAR, EOF
@dataclass Loc(file: str, line: int, col: int)
class Token:                  # __slots__ class
    kind: TokenKind
    text: str                 # exact source spelling
    loc: Loc
    bol: bool                 # first token on its line
    space: bool               # preceded by whitespace
    hideset: frozenset[str]   # macro names that must not expand (preprocessor)
    value: object             # after literals.convert_tokens: int | float | bytes
    ty: Type | None           # type of literal (NUMBER/CHAR/STRING)
```

```python
# hcc/diag.py
class CompileError(Exception): loc, msg, stage
def error(loc_or_tok, msg) -> NoReturn        # raise CompileError
def warn(loc_or_tok, msg) -> None              # print warning to stderr
with diag.stage("parse"): ...                  # tags CompileErrors with a stage
```

On a compile error hcc prints exactly one line to stderr

    <file>:<line>:<col>: error: [<stage>] <message>

and exits with status **1**. An uncaught Python exception inside hcc is an
internal compiler error: it prints `hcc: internal compiler error [<stage>]: ...`
plus a traceback and exits with status **70**. A failing `as` prints
`hcc: error: [assemble] ...` (exit 1); a failing link prints
`hcc: error: [link] ...` (exit 1).

---------------------------------------------------------------------------

## 4. Types (STABLE) — `hcc/typesys.py`

```python
class TypeKind(Enum):
    VOID BOOL CHAR SHORT INT LONG LLONG INT128 FLOAT DOUBLE LDOUBLE FLOAT128
    ENUM PTR ARRAY VLA FUNC STRUCT UNION

class Type:
    kind; size; align; is_unsigned
    is_const; is_volatile; is_restrict; is_atomic   # qualifiers live on the Type copy
    base: Type | None         # PTR/ARRAY/VLA element type
    array_len: int            # ARRAY; -1 = incomplete ("int a[]")
    vla_len: Expr | None; vla_size: Var | None
    # STRUCT/UNION
    members: list[Member]; tag: str | None; is_flexible; is_packed; is_complete
    # FUNC
    return_ty; params: list[Type]; param_names: list[str|None]
    is_variadic; has_proto (False for K&R "int f()")
    origin: Type | None       # the type this was copied from (for compat)
    name                      # unused; Parser.declarator() returns (Type, name_tok)

class Member: name, ty, offset, align, idx, is_bitfield, bit_offset, bit_width
```

LP64 sizes/alignments:

| type | size | align | notes |
|---|---|---|---|
| `_Bool` | 1 | 1 | unsigned |
| `char` | 1 | 1 | **signed** on x86-64 |
| `short` | 2 | 2 | |
| `int`, `enum` | 4 | 4 | enum underlying type is `unsigned int` if no negative constants, else `int` (gcc) |
| `long`, `long long`, pointers | 8 | 8 | |
| `__int128` | 16 | 16 | parse + size only in v1 |
| `float` | 4 | 4 | |
| `double` | 8 | 8 | |
| `long double` | 16 | 16 | x87 80-bit |
| `_Float128`/`__float128` | 16 | 16 | parse + size only |
| `__builtin_va_list` | 24 | 8 | `struct{u32 gp_offset; u32 fp_offset; void *overflow_arg_area; void *reg_save_area}[1]` |

Helper API (all pure functions): `is_integer(t) is_flonum(t) is_numeric(t)
is_scalar(t) is_pointer_like(t)  pointer_to(t) array_of(t,n) func_type(ret)
struct_type() enum_type() vla_of(t, expr) int_type_for(size, unsigned)`.
Base singletons: `TY_VOID TY_BOOL TY_CHAR TY_SHORT TY_INT TY_LONG TY_LLONG
TY_UCHAR TY_USHORT TY_UINT TY_ULONG TY_ULLONG TY_FLOAT TY_DOUBLE TY_LDOUBLE
TY_INT128 TY_UINT128 TY_FLOAT128`. **Never mutate a singleton** — call
`t.copy()` then modify (e.g. to add `const`).

---------------------------------------------------------------------------

## 5. AST (STABLE) — `hcc/nodes.py`

All nodes are `@dataclass(eq=False)` subclasses of `Node` with `loc: Loc`.
Expressions subclass `Expr` and have `ty: Type | None` (set by
`sema.typing.add_type` as soon as the node is built; codegen may assume it is
set). Statements subclass `Stmt`.

### Objects
```python
Obj(name, ty, is_static, asm_name)                 # base
Var(Obj):      is_local, is_extern, is_definition, is_tentative, is_tls,
               align, offset (frame offset, set by codegen),
               init_data: GlobalData | None (static storage initializer),
               is_param
Function(Obj): params: list[Var], locals: list[Var], body: Block | None,
               is_inline, is_definition, va_area: Var | None,
               alloca_bottom: Var | None, is_noreturn
GlobalData:    data: bytearray, relocs: list[Reloc(offset, label, addend)]
TranslationUnit: objs: list[Obj]   # in definition order
```
Lowering conventions (performed by parser/sema so codegen stays simple):
* `p + i` on pointers becomes `Binary('+', p, Binary('*', i, IntLit(size)))`
  with `ty` = pointer type; `p - q` becomes `Binary('/', Binary('-', p, q), size)`.
* `a op= b` -> `Comma(Assign(tmp, AddrOf(a)), Assign(Deref(tmp), Binary(op, Deref(tmp), b)))`
  (`sema/arith.py`). `a++` -> `Cast(Binary('-', (a += 1), 1))` etc. Bitfields too.
* `a->b` -> `Member_(Deref(a), m)`.
* `sizeof`/`_Alignof` of non-VLA -> `IntLit`.
* A local declaration with initializer -> `ExprStmt(Comma(MemZero(var), assigns...))`.
* Implicit conversions (assignment, args, return, usual arithmetic) -> `Cast`.
* Array/function designators are *not* decayed in the tree; codegen's `load`
  of an array/function type is a no-op (address stays in %rax).

### Expressions
| class | fields | meaning |
|---|---|---|
| `IntLit` | `value: int` | integer constant (ty set) |
| `FloatLit` | `value: float` | float/double/long double constant |
| `StrLit` | `data: bytes` | reserved; the parser currently turns every string literal into an anonymous static `Var` (`.L..anon.N`, `init_data` = bytes) referenced by a `VarRef` |
| `VarRef` | `var: Obj` | variable or function designator |
| `Unary` | `op: str ('-','~','!'), operand` | |
| `AddrOf` / `Deref` | `operand` | `&x` / `*x` |
| `Binary` | `op: str, lhs, rhs` | `+ - * / % & \| ^ << >> == != < <= > >=` (operands already converted) |
| `Logical` | `op: '&&'\|'\|\|', lhs, rhs` | |
| `Assign` | `lhs, rhs` | simple assignment (rhs already cast) |
| `Cond` | `cond, then, els` | `?:` |
| `Comma` | `lhs, rhs` | |
| `Cast` | `operand` (target is `ty`) | explicit or implicit conversion |
| `Call` | `func: Expr, args: list[Expr], func_ty: Type, ret_buffer: Var\|None` | |
| `Member_` | `operand, member: typesys.Member` | `s.m` (trailing underscore avoids a clash with `typesys.Member`; codegen method `expr_Member_`) |
| `StmtExpr` | `body: Block` | GNU `({ ... })` |
| `MemZero` | `var: Var` | zero a local (initializers) |
| `BuiltinCall` | `name: str, args: list[Expr], type_arg: Type\|None` | `__builtin_*` handled in codegen/builtins.py |
| `LabelAddr` | `label: str` | GNU `&&label` |
| `VLAPtr` | `var: Var` | pointer to VLA storage |

### Statements
| class | fields |
|---|---|
| `Block` | `items: list[Stmt]` |
| `ExprStmt` | `expr` |
| `If` | `cond, then, els` |
| `For` | `init: Stmt\|None, cond, inc, body, brk_label, cont_label` (also `while`) |
| `DoWhile` | `body, cond, brk_label, cont_label` |
| `Switch` | `cond, body, cases: list[Case], default: Case\|None, brk_label` |
| `Case` | `lo, hi (int; GNU ranges), body, label` |
| `Break` / `Continue` | `label` (the target label string) |
| `Goto` | `name` ; `GotoExpr` `target: Expr` (computed goto) |
| `Label` | `name, body` |
| `Return` | `expr\|None` (already cast to return type) |
| `Asm` | `text, outputs, inputs, clobbers, is_volatile` |

Labels (`brk_label`, `cont_label`, `Case.label`) are unique strings
allocated by `Parser.new_label()` (`.L..N`). User labels for `goto` are
mangled by codegen as `.L.<function>.<name>`.

### Initializers
`Initializer(ty, expr: Expr|None, children: list[Initializer]|None,
is_flexible: bool, mem: Member|None)` — a tree mirroring the aggregate type.
Locals are lowered to assignments; statics/globals are serialized by
`sema/initializers.py` into `GlobalData` (bytes + relocations).

---------------------------------------------------------------------------

## 6. GNU extensions required by glibc headers

| extension | handling |
|---|---|
| `__attribute__((...))` / `__attribute` | `parser/attributes.py` parses the list; recognized: `aligned`, `packed`, `noreturn`, `section`, `weak`, `alias`, `visibility`, `unused`, `cleanup` (later), `vector_size` (later). Unknown attributes are skipped (balanced parens). |
| `__asm__("name")` after a declarator | sets `Obj.asm_name`; codegen uses it as the symbol (e.g. `fscanf` -> `__isoc99_fscanf`). |
| `__asm__`/`asm` statements | `ast.Asm`, emitted by `codegen/asm.py` |
| `__extension__` | ignored (parsed as a no-op prefix in decls and exprs) |
| `__restrict`, `__restrict__`, `__inline`, `__inline__`, `__const`, `__volatile__`, `__signed__`, `__alignof__`, `__typeof`, `__typeof__` | keyword aliases in `literals.py` `KEYWORD_ALIASES` |
| `__builtin_va_list` | builtin type (table in section 4) |
| `__typeof__(expr\|type)` | `parser/declspec.py` |
| `({ ... })` statement expressions | `ast.StmtExpr` |
| `__int128`, `_Float128`, `__float128`, `_Float32/64/32x/64x` | parse; sizes per section 4 |
| `__builtin_offsetof`, `__builtin_types_compatible_p`, `__builtin_expect`, `__builtin_constant_p` | `parser/builtins.py` |
| `_Static_assert` | `parser/toplevel.py` / statements |
| `__label__`, nested functions | not supported |

Predefined macros (`preprocess/predefined.py`): `__x86_64__ __x86_64 __amd64__
__LP64__ _LP64 __linux__ __linux __unix__ __unix __gnu_linux__ __ELF__
__STDC__ 1, __STDC_VERSION__ 201112L, __STDC_HOSTED__ 1, __GNUC__ 4,
__GNUC_MINOR__ 2, __GNUC_PATCHLEVEL__ 1, __CHAR_BIT__ 8, __SIZEOF_*__,
__SIZE_TYPE__, __PTRDIFF_TYPE__, __WCHAR_TYPE__, __INTMAX_TYPE__, *_MAX__`,
byte order macros, and `__hcc__ 1`. `-std=c99`/`gnu11` etc. adjusts
`__STDC_VERSION__` and `__STRICT_ANSI__`.

Include search order (after `-I` dirs): `/usr/lib/gcc/x86_64-linux-gnu/<N>/include`
(highest N found), `/usr/local/include`, `/usr/include/x86_64-linux-gnu`,
`/usr/include`. `#include_next` continues after the directory the current
file was found in.

---------------------------------------------------------------------------

## 7. Codegen conventions — `hcc/codegen/`

* `CodeGen` is assembled from mixins in `codegen/__init__.py`. Methods:
  `gen_expr(node)` (result in accumulator), `gen_stmt(node)`,
  `gen_addr(node)` (address of lvalue in %rax), `load(ty)`, `store(ty)`
  (stores %rax/%xmm0 to address on top of stack), `push()`/`pop(reg)`,
  `pushf()`/`popf(n)`, `cast(from_ty, to_ty)`, `emit(line)`,
  `new_label()`.
* `self.depth` tracks pushes; must be even (16-byte alignment) at every `call`
  — `calls.py` pads when needed.
* Dispatch: `gen_expr` calls `getattr(self, "expr_" + type(node).__name__)`;
  `gen_stmt` calls `"stmt_" + name`. Missing handler -> `diag.error(...,
  "codegen: unsupported <Name>")` (stage codegen).
* Symbols: `Obj.asm_name or Obj.name`. Globals are addressed
  `lea sym(%rip)`; we link with `-no-pie` (so extern data uses copy relocs).
  Thread-locals use `%fs:sym@tpoff`.
* Frame: `push %rbp; mov %rsp,%rbp; sub $N,%rsp`. Local `Var.offset` is a
  negative offset from %rbp. Params are spilled to locals in the prologue.
  Variadic functions get a 176-byte register save area (`va_area`).
* Integer loads sign/zero-extend to 32/64 bits: `movsbl movzbl movswl movzwl
  movslq`; 32-bit values are kept in `%eax` (upper half undefined until a
  cast). `casts.py` owns the conversion table.

---------------------------------------------------------------------------

## 8. Driver / CLI — `hcc/driver.py`

`python3 -m hcc [options] files...`

* `-c` compile+assemble, `-S` compile only, `-E` preprocess only (hcc's own
  preprocessor; output to stdout or `-o`), none: compile + link via `gcc`.
* `-o FILE`, `-I DIR`, `-isystem DIR`, `-idirafter DIR`, `-D NAME[=VAL]`,
  `-U NAME`, `-include FILE`, `-std=...`, `-x c` ; ignored: `-O*`, `-g*`,
  `-W*`, `-w`, `-f*`, `-m*`, `-pedantic`, `-pipe`, `-M*` (best effort).
* Passed to the linker: `*.o *.a *.so`, `-l*`, `-L*`, `-Wl,*`, `-static`,
  `-shared`, `-pthread`, `-rdynamic`, `-lm`...
* Inputs ending in `.s`/`.S` are assembled (`.S` preprocessed by hcc first).
* `-###`-style debugging: `HCC_KEEP_TEMPS=1` keeps the temp `.s` files.
* `-dump-ast` / `-dump-tokens` (hcc-specific) print debug output.

---------------------------------------------------------------------------

## 9. Self tests — `hcc/selftest/`

Each `hcc/selftest/<area>_<name>.c` is compiled with hcc, linked, and run.
Expectations are header comments (first lines of the file):

```c
// EXPECT-EXIT: 42            (default 0)
// EXPECT-STDOUT: hello       (one line each; omitted => stdout not checked)
// EXPECT-ERROR: [parse]      (compile must fail; stderr must contain text)
```

Run `python3 -m hcc.selftest` (all), `-k struct` (filter), `-j 4`.

---------------------------------------------------------------------------

## 10. How to add a feature (checklist)

1. Write a minimal C test in `hcc/selftest/<area>_<feature>.c` that fails now.
2. Run `python3 -m hcc file.c -S -o -` (or `-dump-ast`) to see which stage
   fails; the error line tells you the stage: `[preprocess]`, `[parse]`,
   `[sema]`, `[codegen]`.
3. Find the home in the module map (section 2). Read only that file plus
   `nodes.py`/`typesys.py` as needed.
4. Keep the change local: new node -> `nodes.py` + parser + `sema/typing.py` +
   one `expr_X`/`stmt_X` method. New type rule -> `sema/conversions.py`.
   New builtin -> `parser/builtins.py` (if it takes a type) and
   `codegen/builtins.py` (`builtin_<name>` method).
5. If a file would exceed 600 lines, split it and update section 2.
6. Run `python3 -m hcc.selftest`; commit with a message naming the feature.
7. Do not change STABLE interfaces (sections 3–5) without editing this file
   in the same commit, and keep old field names working.

### Parser conventions worth knowing
* `Parser.declarator(ty)` returns `(Type, name_token_or_None)`; `typename()`
  returns a `Type`. Struct types are shared objects (completed in place), so
  never `copy()` a struct type just to attach information.
* `Parser.typed(node)` must be called on every new Expr (sema.typing.add_type).
* `Parser.new_temp(ty)` creates an anonymous local (used by lowering).
* Statement keywords dispatch to `StatementMixin._stmt_<keyword>`;
  special builtins to `BuiltinParseMixin._b_<name>`; preprocessing directives
  to `DirectiveMixin.dir_<name>`.

### Status / not yet implemented (v1 slice)
See `docs/STATUS.md` (kept up to date by contributors) — anything listed
there as TODO is fair game.
