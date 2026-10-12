"""Lowering of GCC's x86 SSE/MMX intrinsics (`__builtin_ia32_<name>`, typed by
parser/x86_sse_sigs.py) to SSE instructions.

Operands: each argument is evaluated and pushed; afterwards they are popped in
reverse. Vector arguments are loaded from their address into %xmm0, %xmm1, ...
(in argument order); a scalar argument ends up in %r10. Vector results are
stored into the node's result_buf and %rax = its address; scalar results are
left in %eax/%rax. 64-bit (MMX) vectors use the low 8 bytes of an xmm register.
An immediate argument that is not an integer constant is only accepted where
an SSE register-count form exists (shifts); otherwise a clear codegen error.
"""
from __future__ import annotations

import re

from hcc import diag
from hcc import nodes as N

# ---- mnemonic tables (name without __builtin_ia32_ -> instruction) -----------

_BIN_128 = """
packsswb128 packuswb128 packssdw128 punpcklbw128 punpcklwd128 punpckldq128 punpcklqdq128
punpckhbw128 punpckhwd128 punpckhdq128 punpckhqdq128 pcmpgtb pcmpgtw pcmpgtd pcmpeqb pcmpeqw pcmpeqd
psubb psubw psubd psubq psubusb psubusw psubsb psubsw psubusb128 psubusw128 psubsb128 psubsw128
paddb paddw paddd paddq paddusb paddusw paddsb paddsw paddusb128 paddusw128 paddsb128 paddsw128
pand pandn por pxor pandn128 pmullw pmulhw pmulhuw pmulhw128 pmulhuw128 pmuludq pmuludq128
pmaddwd pmaddwd128 psadbw psadbw128 pminsw pmaxsw pminub pmaxub pminsw128 pmaxsw128 pminub128
pmaxub128 pavgb pavgw pavgb128 pavgw128 punpcklbw punpcklwd punpckldq
psllw psllw128 pslld pslld128 psllq psllq128 psrlw psrlw128 psrld psrld128 psrlq psrlq128
psraw psraw128 psrad psrad128
addss addsd addps addpd subss subsd mulss mulsd divss divsd divps divpd minss minsd minps minpd
maxss maxsd maxps maxpd andps andpd andnps andnpd orps orpd xorps xorpd
unpcklps unpckhps unpcklpd unpckhpd movss movsd movhlps movlhps cvtss2sd cvtsd2ss
"""
_PACK_MMX = "packsswb packuswb packssdw"          # MMX packs: operands joined in one xmm
_MMX_HI = "punpckhbw punpckhwd punpckhdq"       # MMX high unpacks: shift by 4 bytes first
_SHIMM = """pslldqi128 psrldqi128 psllwi128 psllwi psrlwi128 psrlwi psrawi128 psrawi
psradi128 psradi pslldi128 pslldi psrldi128 psrldi psllqi128 psllqi psrlqi128 psrlqi"""
_UNIMM = "pshufd pshuflw pshufhw pshufw"
_SHUF = "shufps shufpd"
_UN = """sqrtss sqrtsd sqrtps sqrtpd rsqrtss rsqrtps rcpss rcpps cvttps2dq cvtps2dq cvttpd2dq
cvtpd2dq cvtps2pd cvtpd2ps cvtdq2ps cvtdq2pd movq128"""
_SCALAR_RES = """cvttss2si cvttsd2si cvtss2si cvtsd2si movmskps movmskpd pmovmskb128 pmovmskb"""
_SCALAR_RES64 = "cvttss2si64 cvttsd2si64 cvtss2si64 cvtsd2si64"
_CVTSI = "cvtsi2ss cvtsi2sd cvtsi642ss cvtsi642sd"
_FENCE = "mfence lfence sfence emms pause"
_VEC_EXT = "vec_ext_v8hi vec_ext_v4hi vec_ext_v4si vec_ext_v2si"
_VEC_SET = "vec_set_v8hi vec_set_v4hi"
_VEC_INIT = "vec_init_v2si vec_init_v8qi vec_init_v4hi"
_CMP_SWAP = {"cmpgtps": "cmpltps", "cmpgtpd": "cmpltpd", "cmpgtss": "cmpltss", "cmpgtsd": "cmpltsd",
             "cmpgeps": "cmpleps", "cmpgepd": "cmpleps", "cmpgess": "cmpless", "cmpgesd": "cmplesd",
             "cmpngtps": "cmpnltps", "cmpngtpd": "cmpnltpd", "cmpngeps": "cmpnleps", "cmpngepd": "cmpnlepd"}
_COMI_RE = re.compile(r"^(u?comi)(sd|ss)?(eq|neq|lt|le|gt|ge)$")

_ELEM_LOAD = {(1, True): "movsbl", (1, False): "movzbl", (2, True): "movswl", (2, False): "movzwl",
              (4, True): "movl", (4, False): "movl", (8, True): "movq", (8, False): "movq"}
_ELEM_STORE = {1: ("movb", "%al"), 2: ("movw", "%ax"), 4: ("movl", "%eax"), 8: ("movq", "%rax")}


def _words(s: str) -> frozenset:
    return frozenset(s.split())


_BIN = _words(_BIN_128)
_PACK = _words(_PACK_MMX)
_MMXHI = _words(_MMX_HI)
_SHIMM_SET = _words(_SHIMM)
_UNIMM_SET = _words(_UNIMM)
_SHUF_SET = _words(_SHUF)
_UN_SET = _words(_UN)
_SCALAR_SET = _words(_SCALAR_RES) | _words(_SCALAR_RES64)
_CVTSI_SET = _words(_CVTSI)
_FENCE_SET = _words(_FENCE)
_VEC_EXT_SET = _words(_VEC_EXT)
_VEC_SET_SET = _words(_VEC_SET)
_VEC_INIT_SET = _words(_VEC_INIT)


def _mn(name: str) -> str:
    """Mnemonic of a plain (register) form: drop the 128 / MMX-era suffixes."""
    return name[:-3] if name.endswith("128") else name


def _shift_mn(name: str) -> str:
    """psllwi128 / psllwi -> psllw, pslldqi128 -> pslldq."""
    if name.endswith("i128"):
        return name[:-4]
    return name[:-1]


def _store_result(gen, node) -> None:
    """Vector result: %xmm0 -> result_buf; %rax = its address."""
    off = node.result_buf.offset
    if node.ty.size == 16:
        gen.emit(f"movdqu %xmm0, {off}(%rbp)")
    else:
        gen.emit(f"movq %xmm0, {off}(%rbp)")
    gen.emit(f"lea {off}(%rbp), %rax")


def _eval_push(gen, args) -> None:
    for a in args:
        gen.gen_expr(a)
        gen.push()


def _pop_load(gen, args) -> None:
    """Pop the operands pushed by _eval_push: vector j -> %xmmj (vectors in
    argument order), a scalar -> %r10. Only one scalar operand is supported."""
    vpos = {}
    for i, a in enumerate(args):
        if a.ty.is_vector:
            vpos[i] = len(vpos)
    for i in reversed(range(len(args))):
        gen.pop("%r11")
        if i in vpos:
            if args[i].ty.size == 16:
                gen.emit(f"movdqu (%r11), %xmm{vpos[i]}")
            else:
                gen.emit(f"movq (%r11), %xmm{vpos[i]}")
        else:
            gen.emit("mov %r11, %r10")


def _imm_value(node, idx: int):
    """The integer value of argument idx if it is a constant, else None."""
    a = node.args[idx]
    return a.value if isinstance(a, N.IntLit) else None


def _need_const(gen, node, idx: int) -> int:
    v = _imm_value(node, idx)
    if v is None:
        diag.error(node, f"codegen: '__builtin_{node.name}' needs a constant argument {idx + 1}", "codegen")
    return v


def _bin(gen, node, mn: str) -> None:
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit(f"{mn} %xmm1, %xmm0")
    _store_result(gen, node)


def _bin_cmp_swap(gen, node, mn: str, name: str) -> None:
    """cmpgt(a, b) == cmplt(b, a): compare with the operands exchanged; the upper
    lanes of the result come from a (so the ss/sd forms merge with a)."""
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit("movaps %xmm1, %xmm2")
    gen.emit(f"{mn} %xmm0, %xmm2")
    if name.endswith("ss"):
        gen.emit("movss %xmm2, %xmm0")
    elif name.endswith("sd"):
        gen.emit("movsd %xmm2, %xmm0")
    else:
        gen.emit("movaps %xmm2, %xmm0")
    _store_result(gen, node)


def _pack_mmx(gen, node, mn: str) -> None:
    """MMX pack: a in the low qword and b in the high qword of one xmm, then the
    128-bit pack reads exactly the 4 (or 2) elements of each MMX operand."""
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit("punpcklqdq %xmm1, %xmm0")
    gen.emit(f"{mn} %xmm0, %xmm0")
    _store_result(gen, node)


def _mmx_hi(gen, node, mn: str) -> None:
    """MMX punpckh*: the high half of each operand moved down, then punpckl*."""
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit("psrldq $4, %xmm0")
    gen.emit("psrldq $4, %xmm1")
    gen.emit(f"{mn} %xmm1, %xmm0")
    _store_result(gen, node)


def _shimm(gen, node, name: str) -> None:
    """Shift by an immediate (psllwi & co) or, when the count is not a constant,
    by a register count (the count vector lives in %xmm1; byte shifts have none)."""
    mn = _shift_mn(name)
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    v = _imm_value(node, 1)
    if v is not None:
        if v < 0 or v > 255:
            v = 255     # hardware shifts by >= the element width: zero / sign fill
        gen.emit(f"{mn} ${v}, %xmm0")
    elif mn.endswith("dq"):
        _need_const(gen, node, 1)
    else:
        gen.emit("movd %r10d, %xmm1")
        gen.emit(f"{mn} %xmm1, %xmm0")
    _store_result(gen, node)


def _unimm(gen, node, mn: str) -> None:
    v = _need_const(gen, node, 1)
    _eval_push(gen, node.args[:1])
    _pop_load(gen, node.args[:1])
    gen.emit(f"{mn} ${v & 255}, %xmm0, %xmm0")
    _store_result(gen, node)


def _shuf(gen, node, mn: str) -> None:
    v = _need_const(gen, node, 2)
    _eval_push(gen, node.args[:2])
    _pop_load(gen, node.args[:2])
    gen.emit(f"{mn} ${v & 255}, %xmm1, %xmm0")
    _store_result(gen, node)


def _un(gen, node, mn: str) -> None:
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit(f"{mn} %xmm0, %xmm0")
    _store_result(gen, node)


def _scalar_res(gen, node, mn: str) -> None:
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    reg = "%rax" if node.ty.size == 8 else "%eax"
    gen.emit(f"{mn} %xmm0, {reg}")


def _cvtsi(gen, node, mn: str) -> None:
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    reg = "%r10" if node.args[1].ty.size == 8 else "%r10d"
    gen.emit(f"{mn} {reg}, %xmm0")
    _store_result(gen, node)


def _comi(gen, node, mn: str, pred: str) -> None:
    """Scalar compare into an int 0/1; ordered predicates swap the operands so
    that 'above' (CF=0, ZF=0) tests the right way round. eq/neq test PF too."""
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    if pred in ("lt", "le"):
        gen.emit(f"{mn} %xmm0, %xmm1")
    else:
        gen.emit(f"{mn} %xmm1, %xmm0")
    if pred == "eq":
        gen.emit("sete %al")
        gen.emit("setnp %dl")
        gen.emit("and %dl, %al")
    elif pred == "neq":
        gen.emit("setne %al")
        gen.emit("setp %dl")
        gen.emit("or %dl, %al")
    elif pred in ("gt", "lt"):
        gen.emit("seta %al")
    else:
        gen.emit("setae %al")
    gen.emit("movzbl %al, %eax")


def _vec_ext(gen, node) -> None:
    """vec_ext_vXX(v, i): element i of a vector, widened to int."""
    idx = _need_const(gen, node, 1)
    vty = node.args[0].ty
    es = vty.base.size
    if not 0 <= idx < vty.size // es:
        diag.error(node, "codegen: vector index out of range", "codegen")
    gen.gen_expr(node.args[0])
    mn = _ELEM_LOAD[(es, not vty.base.is_unsigned)]
    dst = "%rax" if es == 8 else "%eax"
    gen.emit(f"{mn} {idx * es}(%rax), {dst}")


def _copy_vec(gen, size: int, src: str, dst_off: int) -> None:
    """Copy a vector value at src into the result buffer (through %xmm0)."""
    if size == 16:
        gen.emit(f"movdqu {src}, %xmm0")
        gen.emit(f"movdqu %xmm0, {dst_off}(%rbp)")
    else:
        gen.emit(f"movq {src}, %xmm0")
        gen.emit(f"movq %xmm0, {dst_off}(%rbp)")


def _vec_set(gen, node) -> None:
    """vec_set_vXX(v, scalar, i): copy v into the result buffer, then store the element."""
    idx = _need_const(gen, node, 2)
    vty = node.ty
    es = vty.base.size
    if not 0 <= idx < vty.size // es:
        diag.error(node, "codegen: vector index out of range", "codegen")
    off = node.result_buf.offset
    gen.gen_expr(node.args[0])
    _copy_vec(gen, vty.size, "(%rax)", off)
    gen.gen_expr(node.args[1])
    mn, reg = _ELEM_STORE[es]
    gen.emit(f"{mn} {reg}, {off + idx * es}(%rbp)")
    gen.emit(f"lea {off}(%rbp), %rax")


def _vec_init(gen, node) -> None:
    """vec_init_vXX(e0, e1, ...): elements stored in order into the result buffer."""
    vty = node.ty
    es = vty.base.size
    off = node.result_buf.offset
    for i, a in enumerate(node.args):
        gen.gen_expr(a)
        mn, reg = _ELEM_STORE[es]
        gen.emit(f"{mn} {reg}, {off + i * es}(%rbp)")
    gen.emit(f"lea {off}(%rbp), %rax")


_LOADS = {"loadlps": "movlps", "loadhps": "movhps", "loadlpd": "movlpd", "loadhpd": "movhpd"}
_STORES = {"storelps": "movlps", "storehps": "movhps", "movntps": "movntps",
           "movntpd": "movntpd", "movntdq": "movntdq"}


def _load_half(gen, node, mn: str) -> None:
    """loadlps(v, p) & co: the low or high 64 bits at p replace one half of v."""
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit(f"{mn} (%r10), %xmm0")
    _store_result(gen, node)


def _store_mem(gen, node, mn: str) -> None:
    """storelps(p, v) & co: store part of v to the address p (void result)."""
    _eval_push(gen, node.args)
    _pop_load(gen, node.args)
    gen.emit(f"{mn} %xmm0, (%r10)")


def _fence(gen, node, mn: str) -> None:
    gen.emit(mn)


def lower_x86(gen, node) -> None:
    """Emit code for BuiltinCall node named `ia32_<name>`."""
    name = node.name[len("ia32_"):]
    if name in _CMP_SWAP:
        _bin_cmp_swap(gen, node, _CMP_SWAP[name], name)
    elif name in _BIN or name.startswith("cmp"):
        _bin(gen, node, _mn(name))
    elif name in _PACK:
        _pack_mmx(gen, node, _mn(name))
    elif name in _MMXHI:
        _mmx_hi(gen, node, "punpckl" + name[len("punpckh"):])
    elif name in _SHIMM_SET:
        _shimm(gen, node, name)
    elif name in _UNIMM_SET:
        _unimm(gen, node, "pshuflw" if name == "pshufw" else name)
    elif name in _SHUF_SET:
        _shuf(gen, node, name)
    elif name in _UN_SET:
        _un(gen, node, "movq" if name == "movq128" else name)
    elif name in _SCALAR_SET:
        _scalar_res(gen, node, _mn(name[:-2]) if name.endswith("64") else _mn(name))
    elif name in _CVTSI_SET:
        _cvtsi(gen, node, name.replace("cvtsi642", "cvtsi2"))
    elif name in _FENCE_SET:
        _fence(gen, node, name)
    elif name in _VEC_EXT_SET:
        _vec_ext(gen, node)
    elif name in _VEC_SET_SET:
        _vec_set(gen, node)
    elif name in _VEC_INIT_SET:
        _vec_init(gen, node)
    elif name in _LOADS:
        _load_half(gen, node, _LOADS[name])
    elif name in _STORES:
        _store_mem(gen, node, _STORES[name])
    else:
        m = _COMI_RE.match(name)
        if m is None:
            diag.error(node, f"codegen: unsupported builtin '__builtin_{node.name}'", "codegen")
        base, width, pred = m.group(1), m.group(2) or "ss", m.group(3)
        _comi(gen, node, base + width, pred)
