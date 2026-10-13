/* luaV_execute (lvm.c, Lua 5.4.7, computed-goto dispatch) */

#include <string.h>

#define VM_NUM_OPCODES 83
#include <math.h>

#define VM_T_NIL 0
#define VM_T_FALSE 0x01
#define VM_T_TRUE 0x11
#define VM_T_INT 0x03
#define VM_T_FLT 0x13
#define VM_T_TABLE 0x45
#define VM_T_LCL 0x46
#define VM_COLLECT 0x40

#define VM_OP_GET(i) ((i) & 0x7f)
#define VM_GETARG_A(i) (((i) >> 7) & 0xff)
#define VM_GETARG_k(i) (((i) >> 15) & 1)
#define VM_GETARG_B(i) (((i) >> 16) & 0xff)
#define VM_GETARG_C(i) (((i) >> 24) & 0xff)
#define VM_GETARG_Bx(i) ((i) >> 15)
#define VM_GETARG_Ax(i) ((i) >> 7)
#define VM_GETARG_sBx(i) (VM_GETARG_Bx(i) - 65535)
#define VM_GETARG_sB(i) (VM_GETARG_B(i) - 127)
#define VM_GETARG_sC(i) (VM_GETARG_C(i) - 127)
#define VM_GETARG_sJ(i) ((int)((i) >> 7) - 16777215)

#define VM_isblack(x) ((x)->marked & (1 << 5))
#define VM_iswhite(x) ((x)->marked & (3 << 3))
#define VM_objbarrier(L, p, o) ((VM_isblack(p) && VM_iswhite(o)) ? luaC_barrier_(L, (GCObject *)(p), (GCObject *)(o)) : (void)0)
#define VM_objbarrierback(L, p, o) ((VM_isblack(p) && VM_iswhite(o)) ? luaC_barrierback_(L, (GCObject *)(p)) : (void)0)
#define VM_barrier(L, p, v) (((v)->tt_ & BIT_ISCOLLECTABLE) ? VM_objbarrier(L, p, gcvalue(v)) : (void)0)
#define VM_barrierback(L, p, v) (((v)->tt_ & BIT_ISCOLLECTABLE) ? VM_objbarrierback(L, p, gcvalue(v)) : (void)0)
#define VM_tonumberns(o, n) \
  (((o)->tt_ == VM_T_FLT) ? ((n) = fltvalue(o), 1) : \
  (((o)->tt_ == VM_T_INT) ? ((n) = (lua_Number)(ivalue(o)), 1) : 0))
#define VM_tointegerns(o, i) \
  (((o)->tt_ == VM_T_INT) ? (*(i) = ivalue(o), 1) \
                          : luaV_tointegerns(o, i, 0))
#define VM_s2v(o) (o)
#define VM_vRB(i) (base + VM_GETARG_B(i))
#define VM_vRC(i) (base + VM_GETARG_C(i))
#define VM_RA(i) (base + VM_GETARG_A(i))
#define VM_KB(i) (k + VM_GETARG_B(i))
#define VM_KC(i) (k + VM_GETARG_C(i))
#define VM_RKC(i) ((VM_GETARG_k(i)) ? k + VM_GETARG_C(i) : base + VM_GETARG_C(i))

#define VM_vmbreak { VM_vmfetch(); goto *disptab[VM_OP_GET(i)]; }
#define VM_intop_add(L, a, b) ((lua_Integer)((lua_Unsigned)(a) + (lua_Unsigned)(b)))
#define VM_intop_sub(L, a, b) ((lua_Integer)((lua_Unsigned)(a) - (lua_Unsigned)(b)))
#define VM_intop_mul(L, a, b) ((lua_Integer)((lua_Unsigned)(a) * (lua_Unsigned)(b)))
#define VM_intop_bnot(a) ((lua_Integer)(~(lua_Unsigned)0 ^ (lua_Unsigned)(a)))
#define VM_fop_add(L, a, b) ((a) + (b))
#define VM_fop_sub(L, a, b) ((a) - (b))
#define VM_fop_mul(L, a, b) ((a) * (b))
#define VM_fop_div(L, a, b) ((a) / (b))
#define VM_fop_pow(L, a, b) (((b) == 2) ? (a) * (a) : pow(a, b))
#define VM_fop_idiv(L, a, b) floor((a) / (b))
#define VM_band(a, b) ((a) & (b))
#define VM_bor(a, b) ((a) | (b))
#define VM_bxor(a, b) ((a) ^ (b))
#define VM_shr(x, y) luaV_shiftl((x), VM_intop_sub(L, 0, (y)))
#define VM_l_lti_f(a, b) ((a) < (b))
#define VM_l_lei_f(a, b) ((a) <= (b))
#define VM_l_gti_f(a, b) ((a) > (b))
#define VM_l_gei_f(a, b) ((a) >= (b))

#define VM_savepc(L) (ci->u.l.savedpc = pc)
#define VM_savestate(L, ci) (VM_savepc(L), L->top = ci->top)
#define VM_updatetrap(ci) (trap = ci->u.l.trap)
#define VM_updatebase(ci) (base = ci->func + 1)
#define VM_updatestack(ci) { if (__builtin_expect(trap != 0, 0)) { VM_updatebase(ci); ra = VM_RA(i); } }
#define VM_Protect(exp) (VM_savestate(L, ci), (exp), VM_updatetrap(ci))
#define VM_ProtectNT(exp) (VM_savepc(L), (exp), VM_updatetrap(ci))
#define VM_halfProtect(exp) (VM_savestate(L, ci), (exp))
#define VM_checkGC(L, c) { if (G(L)->GCdebt > 0) { VM_savepc(L); L->top = (c); luaC_step(L); VM_updatetrap(ci); } }

#define VM_vmfetch() { \
  if (__builtin_expect(trap != 0, 0)) { \
    trap = luaG_traceexec(L, pc); \
    VM_updatebase(ci); \
  } \
  i = *(pc++); \
}

#define VM_dojump(ci, i, e) { pc += VM_GETARG_sJ(i) + (e); VM_updatetrap(ci); }
#define VM_donextjump(ci) { Instruction ni = *pc; VM_dojump(ci, ni, 1); }
#define VM_docondjump() if (cond != VM_GETARG_k(i)) pc++; else VM_donextjump(ci);

#define VM_fastget(L, t, k, slot, f) \
  (!((t)->tt_ == VM_T_TABLE) ? (slot = NULL, 0) : (slot = f(hvalue(t), k), !ttisnil(slot)))
#define VM_fastgeti(L, t, k, slot) \
  (!((t)->tt_ == VM_T_TABLE) ? (slot = NULL, 0) \
   : (slot = ((lua_Unsigned)(k) - 1u < hvalue(t)->alimit) \
              ? &hvalue(t)->array[k - 1] : luaH_getint(hvalue(t), k), \
      !ttisnil(slot)))
#define VM_finishfastset(L, t, slot, v) \
  { TValue *io1_ = (TValue *)(slot); const TValue *io2_ = (v); \
    io1_->value_ = io2_->value_; io1_->tt_ = io2_->tt_; \
    VM_barrierback(L, gcvalue(t), v); }

#define VM_arithI_aux(L, ra, v1, imm, iop, fop) \
  if ((v1)->tt_ == VM_T_INT) { \
    lua_Integer iv1 = ivalue(v1); \
    pc++; ra->value_.i = iop(L, iv1, imm); ra->tt_ = VM_T_INT; \
  } \
  else if ((v1)->tt_ == VM_T_FLT) { \
    lua_Number nb = fltvalue(v1); \
    lua_Number fimm = (lua_Number)(imm); \
    pc++; ra->value_.n = fop(L, nb, fimm); ra->tt_ = VM_T_FLT; \
  }

#define VM_arithf_aux(L, ra, v1, v2, fop) { \
  lua_Number n1; lua_Number n2; \
  if (VM_tonumberns(v1, n1) && VM_tonumberns(v2, n2)) { \
    pc++; ra->value_.n = fop(L, n1, n2); ra->tt_ = VM_T_FLT; \
  } }

#define VM_arith_aux(L, ra, v1, v2, iop, fop) { \
  if ((v1)->tt_ == VM_T_INT && (v2)->tt_ == VM_T_INT) { \
    lua_Integer i1 = ivalue(v1); lua_Integer i2 = ivalue(v2); \
    pc++; ra->value_.i = iop(L, i1, i2); ra->tt_ = VM_T_INT; \
  } \
  else VM_arithf_aux(L, ra, v1, v2, fop); }

#define VM_op_arithI(L, iop, fop) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  int imm = VM_GETARG_sC(i); \
  VM_arithI_aux(L, ra, v1, imm, iop, fop) }

#define VM_op_arithf(L, fop) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  TValue *v2 = VM_vRC(i); \
  VM_arithf_aux(L, ra, v1, v2, fop); }

#define VM_op_arithfK(L, fop) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  TValue *v2 = VM_KC(i); \
  VM_arithf_aux(L, ra, v1, v2, fop); }

#define VM_op_arith(L, iop, fop) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  TValue *v2 = VM_vRC(i); \
  VM_arith_aux(L, ra, v1, v2, iop, fop); }

#define VM_op_arithK(L, iop, fop) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  TValue *v2 = VM_KC(i); \
  VM_arith_aux(L, ra, v1, v2, iop, fop); }

#define VM_op_bitwiseK(L, op) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  TValue *v2 = VM_KC(i); \
  lua_Integer i1; \
  lua_Integer i2 = ivalue(v2); \
  if (VM_tointegerns(v1, &i1)) { \
    pc++; ra->value_.i = op(i1, i2); ra->tt_ = VM_T_INT; \
  } }

#define VM_op_bitwise(L, op) { \
  StkId ra = VM_RA(i); \
  TValue *v1 = VM_vRB(i); \
  TValue *v2 = VM_vRC(i); \
  lua_Integer i1; lua_Integer i2; \
  if (VM_tointegerns(v1, &i1) && VM_tointegerns(v2, &i2)) { \
    pc++; ra->value_.i = op(i1, i2); ra->tt_ = VM_T_INT; \
  } }

#define VM_op_order(L, opi, opn, other) { \
  StkId ra = VM_RA(i); \
  int cond; \
  TValue *rb = VM_vRB(i); \
  if (ttisinteger(ra) && ttisinteger(rb)) { \
    lua_Integer ia = ivalue(ra); \
    lua_Integer ib = ivalue(rb); \
    cond = opi(ia, ib); \
  } \
  else if (ttisnumber(ra) && ttisnumber(rb)) \
    cond = opn(ra, rb); \
  else \
    VM_Protect(cond = other(L, ra, rb)); \
  VM_docondjump(); }

#define VM_op_orderI(L, opi, opf, inv, tm) { \
  StkId ra = VM_RA(i); \
  int cond; \
  int im = VM_GETARG_sB(i); \
  if (ttisinteger(ra)) \
    cond = opi(ivalue(ra), im); \
  else if (ttisfloat(ra)) { \
    lua_Number fa = fltvalue(ra); \
    lua_Number fim = (lua_Number)(im); \
    cond = opf(fa, fim); \
  } \
  else { \
    int isf = VM_GETARG_C(i); \
    VM_Protect(cond = luaT_callorderiTM(L, ra, im, inv, isf, tm)); \
  } \
  VM_docondjump(); }

#define VM_l_lti(a, b) ((a) < (b))
#define VM_l_lei(a, b) ((a) <= (b))
#define VM_l_gti(a, b) ((a) > (b))
#define VM_l_gei(a, b) ((a) >= (b))


void luaV_execute (lua_State *L, CallInfo *ci) {
  LClosure *cl;
  TValue *k;
  StkId base;
  const Instruction *pc;
  int trap;
  static const void *const disptab[VM_NUM_OPCODES] = {
    &&L_OP_MOVE, &&L_OP_LOADI, &&L_OP_LOADF, &&L_OP_LOADK,
    &&L_OP_LOADKX, &&L_OP_LOADFALSE, &&L_OP_LFALSESKIP, &&L_OP_LOADTRUE,
    &&L_OP_LOADNIL, &&L_OP_GETUPVAL, &&L_OP_SETUPVAL, &&L_OP_GETTABUP,
    &&L_OP_GETTABLE, &&L_OP_GETI, &&L_OP_GETFIELD, &&L_OP_SETTABUP,
    &&L_OP_SETTABLE, &&L_OP_SETI, &&L_OP_SETFIELD, &&L_OP_NEWTABLE,
    &&L_OP_SELF, &&L_OP_ADDI, &&L_OP_ADDK, &&L_OP_SUBK,
    &&L_OP_MULK, &&L_OP_MODK, &&L_OP_POWK, &&L_OP_DIVK,
    &&L_OP_IDIVK, &&L_OP_BANDK, &&L_OP_BORK, &&L_OP_BXORK,
    &&L_OP_SHRI, &&L_OP_SHLI, &&L_OP_ADD, &&L_OP_SUB,
    &&L_OP_MUL, &&L_OP_MOD, &&L_OP_POW, &&L_OP_DIV,
    &&L_OP_IDIV, &&L_OP_BAND, &&L_OP_BOR, &&L_OP_BXOR,
    &&L_OP_SHL, &&L_OP_SHR, &&L_OP_MMBIN, &&L_OP_MMBINI,
    &&L_OP_MMBINK, &&L_OP_UNM, &&L_OP_BNOT, &&L_OP_NOT,
    &&L_OP_LEN, &&L_OP_CONCAT, &&L_OP_CLOSE, &&L_OP_TBC,
    &&L_OP_JMP, &&L_OP_EQ, &&L_OP_LT, &&L_OP_LE,
    &&L_OP_EQK, &&L_OP_EQI, &&L_OP_LTI, &&L_OP_LEI,
    &&L_OP_GTI, &&L_OP_GEI, &&L_OP_TEST, &&L_OP_TESTSET,
    &&L_OP_CALL, &&L_OP_TAILCALL, &&L_OP_RETURN, &&L_OP_RETURN0,
    &&L_OP_RETURN1, &&L_OP_FORLOOP, &&L_OP_FORPREP, &&L_OP_TFORPREP,
    &&L_OP_TFORCALL, &&L_OP_TFORLOOP, &&L_OP_SETLIST, &&L_OP_CLOSURE,
    &&L_OP_VARARG, &&L_OP_VARARGPREP, &&L_OP_EXTRAARG
  };

 startfunc:
  trap = L->hookmask;
 returning:
  cl = (LClosure *)(ci->func->value_.gc);
  k = cl->p->k;
  pc = ci->u.l.savedpc;
  if (__builtin_expect(trap != 0, 0)) {
    trap = luaG_tracecall(L);
  }
  base = ci->func + 1;
  for (;;) {
    Instruction i;
    VM_vmfetch();
    goto *disptab[VM_OP_GET(i)];

    L_OP_MOVE: {
      StkId ra = VM_RA(i);
      *ra = *VM_vRB(i);
      VM_vmbreak;
    }
    L_OP_LOADI: {
      StkId ra = VM_RA(i);
      lua_Integer b = VM_GETARG_sBx(i);
      ra->value_.i = b; ra->tt_ = VM_T_INT;
      VM_vmbreak;
    }
    L_OP_LOADF: {
      StkId ra = VM_RA(i);
      int b = VM_GETARG_sBx(i);
      ra->value_.n = (lua_Number)b; ra->tt_ = VM_T_FLT;
      VM_vmbreak;
    }
    L_OP_LOADK: {
      StkId ra = VM_RA(i);
      TValue *rb = k + VM_GETARG_Bx(i);
      *ra = *rb;
      VM_vmbreak;
    }
    L_OP_LOADKX: {
      StkId ra = VM_RA(i);
      TValue *rb;
      rb = k + VM_GETARG_Ax(*pc);
      pc++;
      *ra = *rb;
      VM_vmbreak;
    }
    L_OP_LOADFALSE: {
      StkId ra = VM_RA(i);
      ra->tt_ = VM_T_FALSE;
      VM_vmbreak;
    }
    L_OP_LFALSESKIP: {
      StkId ra = VM_RA(i);
      ra->tt_ = VM_T_FALSE;
      pc++;
      VM_vmbreak;
    }
    L_OP_LOADTRUE: {
      StkId ra = VM_RA(i);
      ra->tt_ = VM_T_TRUE;
      VM_vmbreak;
    }
    L_OP_LOADNIL: {
      StkId ra = VM_RA(i);
      int b = VM_GETARG_B(i);
      do {
        (ra++)->tt_ = VM_T_NIL;
      } while (b--);
      VM_vmbreak;
    }
    L_OP_GETUPVAL: {
      StkId ra = VM_RA(i);
      int b = VM_GETARG_B(i);
      *ra = *cl->upvals[b]->v.p;
      VM_vmbreak;
    }
    L_OP_SETUPVAL: {
      StkId ra = VM_RA(i);
      UpVal *uv = cl->upvals[VM_GETARG_B(i)];
      *uv->v.p = *ra;
      VM_barrier(L, uv, ra);
      VM_vmbreak;
    }
    L_OP_GETTABUP: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *upval = cl->upvals[VM_GETARG_B(i)]->v.p;
      TValue *rc = VM_KC(i);
      TString *key = tsvalue(rc);
      if (VM_fastget(L, upval, key, slot, luaH_getshortstr)) {
        *ra = *slot;
      }
      else
        VM_Protect(luaV_finishget(L, upval, rc, ra, slot));
      VM_vmbreak;
    }
    L_OP_GETTABLE: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *rb = VM_vRB(i);
      TValue *rc = VM_vRC(i);
      lua_Unsigned n;
      if (rc->tt_ == VM_T_INT
          ? (((void)(n = ivalue(rc))), VM_fastgeti(L, rb, n, slot))
          : VM_fastget(L, rb, rc, slot, luaH_get)) {
        *ra = *slot;
      }
      else
        VM_Protect(luaV_finishget(L, rb, rc, ra, slot));
      VM_vmbreak;
    }
    L_OP_GETI: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *rb = VM_vRB(i);
      int c = VM_GETARG_C(i);
      if (VM_fastgeti(L, rb, c, slot)) {
        *ra = *slot;
      }
      else {
        TValue key;
        key.value_.i = c; key.tt_ = VM_T_INT;
        VM_Protect(luaV_finishget(L, rb, &key, ra, slot));
      }
      VM_vmbreak;
    }
    L_OP_GETFIELD: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *rb = VM_vRB(i);
      TValue *rc = VM_KC(i);
      TString *key = tsvalue(rc);
      if (VM_fastget(L, rb, key, slot, luaH_getshortstr)) {
        *ra = *slot;
      }
      else
        VM_Protect(luaV_finishget(L, rb, rc, ra, slot));
      VM_vmbreak;
    }
    L_OP_SETTABUP: {
      const TValue *slot;
      TValue *upval = cl->upvals[VM_GETARG_A(i)]->v.p;
      TValue *rb = VM_KB(i);
      TValue *rc = VM_RKC(i);
      TString *key = tsvalue(rb);
      if (VM_fastget(L, upval, key, slot, luaH_getshortstr)) {
        VM_finishfastset(L, upval, slot, rc);
      }
      else
        VM_Protect(luaV_finishset(L, upval, rb, rc, slot));
      VM_vmbreak;
    }
    L_OP_SETTABLE: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *rb = VM_vRB(i);
      TValue *rc = VM_RKC(i);
      lua_Unsigned n;
      if (rb->tt_ == VM_T_INT
          ? (((void)(n = ivalue(rb))), VM_fastgeti(L, ra, n, slot))
          : VM_fastget(L, ra, rb, slot, luaH_get)) {
        VM_finishfastset(L, ra, slot, rc);
      }
      else
        VM_Protect(luaV_finishset(L, ra, rb, rc, slot));
      VM_vmbreak;
    }
    L_OP_SETI: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      int c = VM_GETARG_B(i);
      TValue *rc = VM_RKC(i);
      if (VM_fastgeti(L, ra, c, slot)) {
        VM_finishfastset(L, ra, slot, rc);
      }
      else {
        TValue key;
        key.value_.i = c; key.tt_ = VM_T_INT;
        VM_Protect(luaV_finishset(L, ra, &key, rc, slot));
      }
      VM_vmbreak;
    }
    L_OP_SETFIELD: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *rb = VM_KB(i);
      TValue *rc = VM_RKC(i);
      TString *key = tsvalue(rb);
      if (VM_fastget(L, ra, key, slot, luaH_getshortstr)) {
        VM_finishfastset(L, ra, slot, rc);
      }
      else
        VM_Protect(luaV_finishset(L, ra, rb, rc, slot));
      VM_vmbreak;
    }
    L_OP_NEWTABLE: {
      StkId ra = VM_RA(i);
      int b = VM_GETARG_B(i);
      int c = VM_GETARG_C(i);
      Table *t;
      if (b > 0)
        b = 1 << (b - 1);
      if (VM_GETARG_k(i))
        c += VM_GETARG_Ax(*pc) * 256;
      pc++;
      L->top = ra + 1;
      t = luaH_new(L);
      ra->value_.gc = (GCObject *)t;
      ra->tt_ = VM_T_TABLE | VM_COLLECT;
      if (b != 0 || c != 0)
        luaH_resize(L, t, c, b);
      VM_checkGC(L, ra + 1);
      VM_vmbreak;
    }
    L_OP_SELF: {
      StkId ra = VM_RA(i);
      const TValue *slot;
      TValue *rb = VM_vRB(i);
      TValue *rc = VM_RKC(i);
      TString *key = tsvalue(rc);
      *(ra + 1) = *rb;
      if (VM_fastget(L, rb, key, slot, luaH_getstr)) {
        *ra = *slot;
      }
      else
        VM_Protect(luaV_finishget(L, rb, rc, ra, slot));
      VM_vmbreak;
    }
    L_OP_ADDI: {
      VM_op_arithI(L, VM_intop_add, VM_fop_add);
      VM_vmbreak;
    }
    L_OP_ADDK: {
      VM_op_arithK(L, VM_intop_add, VM_fop_add);
      VM_vmbreak;
    }
    L_OP_SUBK: {
      VM_op_arithK(L, VM_intop_sub, VM_fop_sub);
      VM_vmbreak;
    }
    L_OP_MULK: {
      VM_op_arithK(L, VM_intop_mul, VM_fop_mul);
      VM_vmbreak;
    }
    L_OP_MODK: {
      VM_savestate(L, ci);
      VM_op_arithK(L, luaV_mod, luaV_modf);
      VM_vmbreak;
    }
    L_OP_POWK: {
      VM_op_arithfK(L, VM_fop_pow);
      VM_vmbreak;
    }
    L_OP_DIVK: {
      VM_op_arithfK(L, VM_fop_div);
      VM_vmbreak;
    }
    L_OP_IDIVK: {
      VM_savestate(L, ci);
      VM_op_arithK(L, luaV_idiv, VM_fop_idiv);
      VM_vmbreak;
    }
    L_OP_BANDK: {
      VM_op_bitwiseK(L, VM_band);
      VM_vmbreak;
    }
    L_OP_BORK: {
      VM_op_bitwiseK(L, VM_bor);
      VM_vmbreak;
    }
    L_OP_BXORK: {
      VM_op_bitwiseK(L, VM_bxor);
      VM_vmbreak;
    }
    L_OP_SHRI: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_vRB(i);
      int ic = VM_GETARG_sC(i);
      lua_Integer ib;
      if (VM_tointegerns(rb, &ib)) {
        pc++; ra->value_.i = luaV_shiftl(ib, -ic); ra->tt_ = VM_T_INT;
      }
      VM_vmbreak;
    }
    L_OP_SHLI: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_vRB(i);
      int ic = VM_GETARG_sC(i);
      lua_Integer ib;
      if (VM_tointegerns(rb, &ib)) {
        pc++; ra->value_.i = luaV_shiftl(ic, ib); ra->tt_ = VM_T_INT;
      }
      VM_vmbreak;
    }
    L_OP_ADD: {
      VM_op_arith(L, VM_intop_add, VM_fop_add);
      VM_vmbreak;
    }
    L_OP_SUB: {
      VM_op_arith(L, VM_intop_sub, VM_fop_sub);
      VM_vmbreak;
    }
    L_OP_MUL: {
      VM_op_arith(L, VM_intop_mul, VM_fop_mul);
      VM_vmbreak;
    }
    L_OP_MOD: {
      VM_savestate(L, ci);
      VM_op_arith(L, luaV_mod, luaV_modf);
      VM_vmbreak;
    }
    L_OP_POW: {
      VM_op_arithf(L, VM_fop_pow);
      VM_vmbreak;
    }
    L_OP_DIV: {
      VM_op_arithf(L, VM_fop_div);
      VM_vmbreak;
    }
    L_OP_IDIV: {
      VM_savestate(L, ci);
      VM_op_arith(L, luaV_idiv, VM_fop_idiv);
      VM_vmbreak;
    }
    L_OP_BAND: {
      VM_op_bitwise(L, VM_band);
      VM_vmbreak;
    }
    L_OP_BOR: {
      VM_op_bitwise(L, VM_bor);
      VM_vmbreak;
    }
    L_OP_BXOR: {
      VM_op_bitwise(L, VM_bxor);
      VM_vmbreak;
    }
    L_OP_SHL: {
      VM_op_bitwise(L, luaV_shiftl);
      VM_vmbreak;
    }
    L_OP_SHR: {
      VM_op_bitwise(L, VM_shr);
      VM_vmbreak;
    }
    L_OP_MMBIN: {
      StkId ra = VM_RA(i);
      Instruction pi = *(pc - 2);
      TValue *rb = VM_vRB(i);
      TMS tm = (TMS)VM_GETARG_C(i);
      StkId result = VM_RA(pi);
      VM_Protect(luaT_trybinTM(L, ra, rb, result, tm));
      VM_vmbreak;
    }
    L_OP_MMBINI: {
      StkId ra = VM_RA(i);
      Instruction pi = *(pc - 2);
      int imm = VM_GETARG_sB(i);
      TMS tm = (TMS)VM_GETARG_C(i);
      int flip = VM_GETARG_k(i);
      StkId result = VM_RA(pi);
      VM_Protect(luaT_trybiniTM(L, ra, imm, flip, result, tm));
      VM_vmbreak;
    }
    L_OP_MMBINK: {
      StkId ra = VM_RA(i);
      Instruction pi = *(pc - 2);
      TValue *imm = VM_KB(i);
      TMS tm = (TMS)VM_GETARG_C(i);
      int flip = VM_GETARG_k(i);
      StkId result = VM_RA(pi);
      VM_Protect(luaT_trybinassocTM(L, ra, imm, flip, result, tm));
      VM_vmbreak;
    }
    L_OP_UNM: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_vRB(i);
      lua_Number nb;
      if (ttisinteger(rb)) {
        lua_Integer ib = ivalue(rb);
        ra->value_.i = VM_intop_sub(L, 0, ib); ra->tt_ = VM_T_INT;
      }
      else if (VM_tonumberns(rb, nb)) {
        ra->value_.n = -nb; ra->tt_ = VM_T_FLT;
      }
      else
        VM_Protect(luaT_trybinTM(L, rb, rb, ra, TM_UNM));
      VM_vmbreak;
    }
    L_OP_BNOT: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_vRB(i);
      lua_Integer ib;
      if (VM_tointegerns(rb, &ib)) {
        ra->value_.i = VM_intop_bnot(ib); ra->tt_ = VM_T_INT;
      }
      else
        VM_Protect(luaT_trybinTM(L, rb, rb, ra, TM_BNOT));
      VM_vmbreak;
    }
    L_OP_NOT: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_vRB(i);
      if (l_isfalse(rb))
        ra->tt_ = VM_T_TRUE;
      else
        ra->tt_ = VM_T_FALSE;
      VM_vmbreak;
    }
    L_OP_LEN: {
      StkId ra = VM_RA(i);
      VM_Protect(luaV_objlen(L, ra, VM_vRB(i)));
      VM_vmbreak;
    }
    L_OP_CONCAT: {
      StkId ra = VM_RA(i);
      int n = VM_GETARG_B(i);
      L->top = ra + n;
      VM_ProtectNT(luaV_concat(L, n));
      VM_checkGC(L, L->top);
      VM_vmbreak;
    }
    L_OP_CLOSE: {
      StkId ra = VM_RA(i);
      VM_Protect(luaF_close(L, ra, 0, 1));
      VM_vmbreak;
    }
    L_OP_TBC: {
      StkId ra = VM_RA(i);
      VM_halfProtect(luaF_newtbcupval(L, ra));
      VM_vmbreak;
    }
    L_OP_JMP: {
      VM_dojump(ci, i, 0);
      VM_vmbreak;
    }
    L_OP_EQ: {
      StkId ra = VM_RA(i);
      int cond;
      TValue *rb = VM_vRB(i);
      VM_Protect(cond = luaV_equalobj(L, ra, rb));
      VM_docondjump();
      VM_vmbreak;
    }
    L_OP_LT: {
      VM_op_order(L, VM_l_lti, LTnum, lessthanothers);
      VM_vmbreak;
    }
    L_OP_LE: {
      VM_op_order(L, VM_l_lei, LEnum, lessequalothers);
      VM_vmbreak;
    }
    L_OP_EQK: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_KB(i);
      int cond = luaV_equalobj(NULL, ra, rb);
      VM_docondjump();
      VM_vmbreak;
    }
    L_OP_EQI: {
      StkId ra = VM_RA(i);
      int cond;
      int im = VM_GETARG_sB(i);
      if (ttisinteger(ra))
        cond = (ivalue(ra) == im);
      else if (ttisfloat(ra))
        cond = (fltvalue(ra) == (lua_Number)(im));
      else
        cond = 0;
      VM_docondjump();
      VM_vmbreak;
    }
    L_OP_LTI: {
      VM_op_orderI(L, VM_l_lti, VM_l_lti_f, 0, TM_LT);
      VM_vmbreak;
    }
    L_OP_LEI: {
      VM_op_orderI(L, VM_l_lei, VM_l_lei_f, 0, TM_LE);
      VM_vmbreak;
    }
    L_OP_GTI: {
      VM_op_orderI(L, VM_l_gti, VM_l_gti_f, 1, TM_LT);
      VM_vmbreak;
    }
    L_OP_GEI: {
      VM_op_orderI(L, VM_l_gei, VM_l_gei_f, 1, TM_LE);
      VM_vmbreak;
    }
    L_OP_TEST: {
      StkId ra = VM_RA(i);
      int cond = !l_isfalse(ra);
      VM_docondjump();
      VM_vmbreak;
    }
    L_OP_TESTSET: {
      StkId ra = VM_RA(i);
      TValue *rb = VM_vRB(i);
      if (l_isfalse(rb) == VM_GETARG_k(i))
        pc++;
      else {
        *ra = *rb;
        VM_donextjump(ci);
      }
      VM_vmbreak;
    }
    L_OP_CALL: {
      StkId ra = VM_RA(i);
      CallInfo *newci;
      int b = VM_GETARG_B(i);
      int nresults = VM_GETARG_C(i) - 1;
      if (b != 0)
        L->top = ra + b;
      VM_savepc(L);
      if ((newci = luaD_precall(L, ra, nresults)) == NULL)
        VM_updatetrap(ci);
      else {
        ci = newci;
        goto startfunc;
      }
      VM_vmbreak;
    }
    L_OP_TAILCALL: {
      StkId ra = VM_RA(i);
      int b = VM_GETARG_B(i);
      int n;
      int nparams1 = VM_GETARG_C(i);
      int delta = (nparams1) ? ci->u.l.nextraargs + nparams1 : 0;
      if (b != 0)
        L->top = ra + b;
      else
        b = (int)(L->top - ra);
      VM_savepc(ci);
      if (VM_GETARG_k(i)) {
        luaF_closeupval(L, base);
      }
      if ((n = luaD_pretailcall(L, ci, ra, b, delta)) < 0)
        goto startfunc;
      else {
        ci->func -= delta;
        luaD_poscall(L, ci, n);
        VM_updatetrap(ci);
        goto ret;
      }
    }
    L_OP_RETURN: {
      StkId ra = VM_RA(i);
      int n = VM_GETARG_B(i) - 1;
      int nparams1 = VM_GETARG_C(i);
      if (n < 0)
        n = (int)(L->top - ra);
      VM_savepc(ci);
      if (VM_GETARG_k(i)) {
        ci->u2.nres = n;
        if (L->top < ci->top)
          L->top = ci->top;
        luaF_close(L, base, -1, 1);
        VM_updatetrap(ci);
        VM_updatestack(ci);
      }
      if (nparams1)
        ci->func -= ci->u.l.nextraargs + nparams1;
      L->top = ra + n;
      luaD_poscall(L, ci, n);
      VM_updatetrap(ci);
      goto ret;
    }
    L_OP_RETURN0: {
      if (__builtin_expect(L->hookmask != 0, 0)) {
        StkId ra = VM_RA(i);
        L->top = ra;
        VM_savepc(ci);
        luaD_poscall(L, ci, 0);
        trap = 1;
      }
      else {
        int nres;
        L->ci = ci->previous;
        L->top = base - 1;
        for (nres = ci->nresults; __builtin_expect(nres > 0, 0); nres--)
          (L->top++)->tt_ = VM_T_NIL;
      }
      goto ret;
    }
    L_OP_RETURN1: {
      if (__builtin_expect(L->hookmask != 0, 0)) {
        StkId ra = VM_RA(i);
        L->top = ra + 1;
        VM_savepc(ci);
        luaD_poscall(L, ci, 1);
        trap = 1;
      }
      else {
        int nres = ci->nresults;
        L->ci = ci->previous;
        if (nres == 0)
          L->top = base - 1;
        else {
          StkId ra = VM_RA(i);
          *(base - 1) = *ra;
          L->top = base;
          for (; __builtin_expect(nres > 1, 0); nres--)
            (L->top++)->tt_ = VM_T_NIL;
        }
      }
     ret:
      if (ci->callstatus & (1 << 2))
        return;
      else {
        ci = ci->previous;
        goto returning;
      }
    }
    L_OP_FORLOOP: {
      StkId ra = VM_RA(i);
      if (ttisinteger(ra + 2)) {
        lua_Unsigned count = (lua_Unsigned)(ivalue(ra + 1));
        if (count > 0) {
          lua_Integer step = ivalue(ra + 2);
          lua_Integer idx = ivalue(ra);
          (ra + 1)->value_.i = (lua_Integer)(count - 1);
          idx = VM_intop_add(L, idx, step);
          ra->value_.i = idx;
          (ra + 3)->value_.i = idx; (ra + 3)->tt_ = VM_T_INT;
          pc -= VM_GETARG_Bx(i);
        }
      }
      else if (floatforloop(L, ra))
        pc -= VM_GETARG_Bx(i);
      VM_updatetrap(ci);
      VM_vmbreak;
    }
    L_OP_FORPREP: {
      StkId ra = VM_RA(i);
      VM_savestate(L, ci);
      if (forprep(L, ra))
        pc += VM_GETARG_Bx(i) + 1;
      VM_vmbreak;
    }
    L_OP_TFORPREP: {
      StkId ra = VM_RA(i);
      VM_halfProtect(luaF_newtbcupval(L, ra + 3));
      pc += VM_GETARG_Bx(i);
      i = *(pc++);
      goto l_tforcall;
    }
    L_OP_TFORCALL: {
     l_tforcall: {
      StkId ra = VM_RA(i);
      memcpy(ra + 4, ra, 3 * sizeof(*ra));
      L->top = ra + 4 + 3;
      VM_ProtectNT(luaD_call(L, ra + 4, VM_GETARG_C(i)));
      VM_updatestack(ci);
      i = *(pc++);
      goto l_tforloop;
     }}
    L_OP_TFORLOOP: {
     l_tforloop: {
      StkId ra = VM_RA(i);
      if (ra[4].tt_ != VM_T_NIL) {
        *(ra + 2) = *(ra + 4);
        pc -= VM_GETARG_Bx(i);
      }
      VM_vmbreak;
     }}
    L_OP_SETLIST: {
      StkId ra = VM_RA(i);
      int n = VM_GETARG_B(i);
      unsigned int last = VM_GETARG_C(i);
      Table *h = hvalue(ra);
      if (n == 0)
        n = (int)(L->top - ra) - 1;
      else
        L->top = ci->top;
      last += n;
      if (VM_GETARG_k(i)) {
        last += VM_GETARG_Ax(*pc) * 256;
        pc++;
      }
      if (last > luaH_realasize(h))
        luaH_resizearray(L, h, last);
      for (; n > 0; n--) {
        TValue *val = ra + n;
        h->array[last - 1] = *val;
        last--;
        VM_barrierback(L, gcvalue(ra), val);
      }
      VM_vmbreak;
    }
    L_OP_CLOSURE: {
      StkId ra = VM_RA(i);
      Proto *p = cl->p->p[VM_GETARG_Bx(i)];
      VM_halfProtect(pushclosure(L, p, cl->upvals, base, ra));
      VM_checkGC(L, ra + 1);
      VM_vmbreak;
    }
    L_OP_VARARG: {
      StkId ra = VM_RA(i);
      int n = VM_GETARG_C(i) - 1;
      VM_Protect(luaT_getvarargs(L, ci, ra, n));
      VM_vmbreak;
    }
    L_OP_VARARGPREP: {
      VM_ProtectNT(luaT_adjustvarargs(L, VM_GETARG_A(i), ci, cl->p));
      if (__builtin_expect(trap != 0, 0)) {
        luaD_hookcall(L, ci);
        L->oldpc = 1;
      }
      VM_updatebase(ci);
      VM_vmbreak;
    }
    L_OP_EXTRAARG: {
      VM_vmbreak;
    }
  }
}
