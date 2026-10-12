"""Signatures of GCC's x86 SSE/MMX intrinsic builtins (__builtin_ia32_<name>).

Each line of SIG_TEXT is `name return-type param-types...` (no `__builtin_ia32_`
prefix). Type codes:
  v16qi v8hi v4si v2di v4sf v2df   128-bit vectors (SSE)
  v8qi v4hi v2si v1di v2sf         64-bit vectors (MMX, __m64)
  i ll u ull                       int, long long, unsigned int, unsigned long long
  p                                void * (any pointer argument)
  v                                void (only as return type)
Parsing a call types the arguments (scalars converted to the parameter type,
vectors must have the parameter's size) and yields a BuiltinCall named
`ia32_<name>` whose type is the return type. Lowering: codegen/x86_sse.py.
"""
from __future__ import annotations

from hcc import nodes as N
from hcc import typesys as T
from hcc.sema.constexpr import try_eval_const
from hcc.sema.conversions import new_cast

SIG_TEXT = """
pslldqi128 v2di v2di i
psrldqi128 v2di v2di i
cvtpi2ps v4sf v4sf v2si
movlhps v4sf v4sf v4sf
movhlps v4sf v4sf v4sf
shufps v4sf v4sf v4sf i
shufpd v2df v2df v2df i
movsd v2df v2df v2df
movss v4sf v4sf v4sf
unpcklps v4sf v4sf v4sf
unpckhps v4sf v4sf v4sf
unpcklpd v2df v2df v2df
unpckhpd v2df v2df v2df
punpcklwd v4hi v4hi v4hi
punpcklbw v8qi v8qi v8qi
punpckhwd v4hi v4hi v4hi
punpckhbw v8qi v8qi v8qi
punpckldq v2si v2si v2si
punpckhdq v2si v2si v2si
cvtps2pi v2si v4sf
cvttps2pi v2si v4sf
cvttpd2pi v2si v2df
cvtpd2pi v2si v2df
cvtpi2pd v2df v2si
vec_set_v8hi v8hi v8hi i i
vec_set_v4hi v4hi v4hi i i
vec_init_v2si v2si i i
vec_init_v8qi v8qi i i i i i i i i
vec_init_v4hi v4hi i i i i
vec_ext_v8hi i v8hi i
vec_ext_v4hi i v4hi i
vec_ext_v4si i v4si i
vec_ext_v2si i v2si i
pshufw v4hi v4hi i
pshuflw v8hi v8hi i
pshufhw v8hi v8hi i
pshufd v4si v4si i
prefetch v p i
pcmpgtw v4hi v4hi v4hi
pcmpgtb v8qi v8qi v8qi
pcmpgtd v2si v2si v2si
pcmpeqw v4hi v4hi v4hi
pcmpeqd v2si v2si v2si
pcmpeqb v8qi v8qi v8qi
packsswb v8qi v4hi v4hi
packssdw v4hi v2si v2si
packuswb v8qi v4hi v4hi
packsswb128 v16qi v8hi v8hi
packssdw128 v8hi v4si v4si
packuswb128 v16qi v8hi v8hi
maskmovdqu v v16qi v16qi p
maskmovq v v8qi v8qi p
cvttss2si64 ll v4sf
cvttsd2si64 ll v2df
cvtss2si64 ll v4sf
cvtsd2si64 ll v2df
cvtsi642ss v4sf v4sf ll
cvtsi642sd v2df v2df ll
cmpnltss v4sf v4sf v4sf
cmpnltsd v2df v2df v2df
cmpnless v4sf v4sf v4sf
cmpnlesd v2df v2df v2df
cmpltss v4sf v4sf v4sf
cmpltsd v2df v2df v2df
cmpless v4sf v4sf v4sf
cmplesd v2df v2df v2df
cmpunordss v4sf v4sf v4sf
cmpunordsd v2df v2df v2df
cmpunordps v4sf v4sf v4sf
cmpunordpd v2df v2df v2df
cmpordss v4sf v4sf v4sf
cmpordsd v2df v2df v2df
cmpordps v4sf v4sf v4sf
cmpordpd v2df v2df v2df
cmpnltps v4sf v4sf v4sf
cmpnltpd v2df v2df v2df
cmpnleps v4sf v4sf v4sf
cmpnlepd v2df v2df v2df
cmpngtps v4sf v4sf v4sf
cmpngtpd v2df v2df v2df
cmpngeps v4sf v4sf v4sf
cmpngepd v2df v2df v2df
cmpneqss v4sf v4sf v4sf
cmpneqsd v2df v2df v2df
cmpneqps v4sf v4sf v4sf
cmpneqpd v2df v2df v2df
cmpltps v4sf v4sf v4sf
cmpltpd v2df v2df v2df
cmpleps v4sf v4sf v4sf
cmplepd v2df v2df v2df
cmpgtps v4sf v4sf v4sf
cmpgtpd v2df v2df v2df
cmpgeps v4sf v4sf v4sf
cmpgepd v2df v2df v2df
cmpeqss v4sf v4sf v4sf
cmpeqsd v2df v2df v2df
cmpeqps v4sf v4sf v4sf
cmpeqpd v2df v2df v2df
xorps v4sf v4sf v4sf
xorpd v2df v2df v2df
ucomisdneq i v2df v2df
ucomisdlt i v2df v2df
ucomisdle i v2df v2df
ucomisdgt i v2df v2df
ucomisdge i v2df v2df
ucomisdeq i v2df v2df
ucomineq i v4sf v4sf
ucomilt i v4sf v4sf
ucomile i v4sf v4sf
ucomigt i v4sf v4sf
ucomige i v4sf v4sf
ucomieq i v4sf v4sf
subss v4sf v4sf v4sf
subsd v2df v2df v2df
storelps v p v4sf
storehps v p v4sf
stmxcsr u
ldmxcsr v u
sqrtss v4sf v4sf
sqrtsd v2df v2df
sqrtps v4sf v4sf
sqrtpd v2df v2df
rsqrtss v4sf v4sf
rsqrtps v4sf v4sf
rcpss v4sf v4sf
rcpps v4sf v4sf
pxor v1di v1di v1di
punpcklwd128 v8hi v8hi v8hi
punpcklqdq128 v2di v2di v2di
punpckldq128 v4si v4si v4si
punpcklbw128 v16qi v16qi v16qi
punpckhwd128 v8hi v8hi v8hi
punpckhqdq128 v2di v2di v2di
punpckhdq128 v4si v4si v4si
punpckhbw128 v16qi v16qi v16qi
psubw v4hi v4hi v4hi
psubusw128 v8hi v8hi v8hi
psubusw v4hi v4hi v4hi
psubusb128 v16qi v16qi v16qi
psubusb v8qi v8qi v8qi
psubsw128 v8hi v8hi v8hi
psubsw v4hi v4hi v4hi
psubsb128 v16qi v16qi v16qi
psubsb v8qi v8qi v8qi
psubq v1di v1di v1di
psubd v2si v2si v2si
psubb v8qi v8qi v8qi
psrlwi128 v8hi v8hi i
psrlwi v4hi v4hi i
psrlw128 v8hi v8hi v2di
psrlw v4hi v4hi v1di
psrlqi128 v2di v2di i
psrlqi v1di v1di i
psrlq128 v2di v2di v2di
psrlq v1di v1di v1di
psrldi128 v4si v4si i
psrldi v2si v2si i
psrld128 v4si v4si v2di
psrld v2si v2si v1di
psrawi128 v8hi v8hi i
psrawi v4hi v4hi i
psraw128 v8hi v8hi v2di
psraw v4hi v4hi v1di
psradi128 v4si v4si i
psradi v2si v2si i
psrad128 v4si v4si v2di
psrad v2si v2si v1di
psllwi128 v8hi v8hi i
psllwi v4hi v4hi i
psllw128 v8hi v8hi v2di
psllw v4hi v4hi v1di
psllqi128 v2di v2di i
psllqi v1di v1di i
psllq128 v2di v2di v2di
psllq v1di v1di v1di
pslldi128 v4si v4si i
pslldi v2si v2si i
pslld128 v4si v4si v2di
pslld v2si v2si v1di
psadbw128 v2di v16qi v16qi
psadbw v1di v8qi v8qi
por v1di v1di v1di
pmuludq128 v2di v4si v4si
pmuludq v1di v2si v2si
pmullw v4hi v4hi v4hi
pmulhw128 v8hi v8hi v8hi
pmulhw v4hi v4hi v4hi
pmulhuw128 v8hi v8hi v8hi
pmulhuw v4hi v4hi v4hi
pmovmskb128 i v16qi
pmovmskb i v8qi
pminub128 v16qi v16qi v16qi
pminub v8qi v8qi v8qi
pminsw128 v8hi v8hi v8hi
pminsw v4hi v4hi v4hi
pmaxub128 v16qi v16qi v16qi
pmaxub v8qi v8qi v8qi
pmaxsw128 v8hi v8hi v8hi
pmaxsw v4hi v4hi v4hi
pmaddwd128 v4si v8hi v8hi
pmaddwd v2si v4hi v4hi
pavgw128 v8hi v8hi v8hi
pavgw v4hi v4hi v4hi
pavgb128 v16qi v16qi v16qi
pavgb v8qi v8qi v8qi
pause v
pandn128 v2di v2di v2di
pandn v1di v1di v1di
pand v1di v1di v1di
paddw v4hi v4hi v4hi
paddusw128 v8hi v8hi v8hi
paddusw v4hi v4hi v4hi
paddusb128 v16qi v16qi v16qi
paddusb v8qi v8qi v8qi
paddsw128 v8hi v8hi v8hi
paddsw v4hi v4hi v4hi
paddsb128 v16qi v16qi v16qi
paddsb v8qi v8qi v8qi
paddq v1di v1di v1di
paddd v2si v2si v2si
paddb v8qi v8qi v8qi
orps v4sf v4sf v4sf
orpd v2df v2df v2df
mulss v4sf v4sf v4sf
mulsd v2df v2df v2df
movq128 v2di v2di
movntq v p ull
movntps v p v4sf
movntpd v p v2df
movntdq v p v2di
movnti64 v p ll
movnti v p i
movmskps i v4sf
movmskpd i v2df
minss v4sf v4sf v4sf
minsd v2df v2df v2df
minps v4sf v4sf v4sf
minpd v2df v2df v2df
mfence v
lfence v
sfence v
emms v
maxss v4sf v4sf v4sf
maxsd v2df v2df v2df
maxps v4sf v4sf v4sf
maxpd v2df v2df v2df
loadlps v4sf v4sf p
loadlpd v2df v2df p
loadhps v4sf v4sf p
loadhpd v2df v2df p
divss v4sf v4sf v4sf
divsd v2df v2df v2df
divps v4sf v4sf v4sf
divpd v2df v2df v2df
cvttss2si i v4sf
cvttsd2si i v2df
cvtss2si i v4sf
cvtsd2si i v2df
cvtss2sd v2df v2df v4sf
cvtsd2ss v4sf v4sf v2df
cvtsi2ss v4sf v4sf i
cvtsi2sd v2df v2df i
cvttps2dq v4si v4sf
cvttpd2dq v4si v2df
cvtps2pd v2df v4sf
cvtps2dq v4si v4sf
cvtpd2ps v4sf v2df
cvtpd2dq v4si v2df
cvtdq2ps v4sf v4si
cvtdq2pd v2df v4si
comisdneq i v2df v2df
comisdlt i v2df v2df
comisdle i v2df v2df
comisdgt i v2df v2df
comisdge i v2df v2df
comisdeq i v2df v2df
comineq i v4sf v4sf
comilt i v4sf v4sf
comile i v4sf v4sf
comigt i v4sf v4sf
comige i v4sf v4sf
comieq i v4sf v4sf
andps v4sf v4sf v4sf
andpd v2df v2df v2df
andnps v4sf v4sf v4sf
andnpd v2df v2df v2df
addss v4sf v4sf v4sf
addsd v2df v2df v2df
addps v4sf v4sf v4sf
addpd v2df v2df v2df
clflush v p
"""

_VEC = {
    "v16qi": (T.TY_CHAR, 16), "v8hi": (T.TY_SHORT, 16), "v4si": (T.TY_INT, 16),
    "v2di": (T.TY_LLONG, 16), "v4sf": (T.TY_FLOAT, 16), "v2df": (T.TY_DOUBLE, 16),
    "v8qi": (T.TY_CHAR, 8), "v4hi": (T.TY_SHORT, 8), "v2si": (T.TY_INT, 8),
    "v1di": (T.TY_LLONG, 8), "v2sf": (T.TY_FLOAT, 8),
}
_SCALAR = {"i": T.TY_INT, "ll": T.TY_LLONG, "u": T.TY_UINT, "ull": T.TY_ULLONG, "v": T.TY_VOID}


def _type_of(code: str) -> T.Type:
    if code in _VEC:
        base, nbytes = _VEC[code]
        return T.vector_of(base, nbytes)
    if code == "p":
        return T.pointer_to(T.TY_VOID)
    return _SCALAR[code]


def _parse_sigs(text: str) -> dict:
    sigs = {}
    for line in text.strip().splitlines():
        name, ret, *params = line.split()
        sigs[name] = (ret, tuple(params))
    return sigs


# name (without __builtin_ia32_) -> (return code, tuple of parameter codes)
SIGS = _parse_sigs(SIG_TEXT)
X86_BUILTINS = frozenset("__builtin_ia32_" + n for n in SIGS)


def x86_call(parser, tok):
    """Parse the argument list of an x86 intrinsic (after its '(') and build a
    typed BuiltinCall. The closing ')' is left to the caller."""
    name = tok.text[len("__builtin_"):]
    ret, params = SIGS[name[len("ia32_"):]]
    args = parser._args(len(params))
    out = []
    for i, (a, code) in enumerate(zip(args, params)):
        pty = _type_of(code)
        if pty.is_vector:
            if not a.ty.is_vector or a.ty.size != pty.size:
                parser.error(f"incompatible argument {i + 1} of '{tok.text}'", a)
            out.append(a)
        elif code == "i":
            v = try_eval_const(a)     # immediates: fold constant expressions (`(int)(N)`)
            out.append(N.IntLit(a.loc, v, ty=T.TY_INT) if isinstance(v, int) else new_cast(a, pty))
        else:
            out.append(new_cast(a, pty))
    rty = _type_of(ret)
    node = N.BuiltinCall(tok.loc, name, out, None, ty=rty)
    if rty.is_vector:
        node.result_buf = parser.new_temp(rty, tok.loc)
    return node
