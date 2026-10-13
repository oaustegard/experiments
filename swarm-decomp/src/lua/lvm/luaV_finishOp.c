#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaV_concat.h"

/* Lua 5.4 opcode numbers (lopcodes.h) */
#define OP_MMBIN_ 46
#define OP_MMBINI_ 47
#define OP_MMBINK_ 48
#define OP_CONCAT_ 53
#define OP_CLOSE_ 54
#define OP_EQ_ 57
#define OP_LT_ 58
#define OP_LE_ 59
#define OP_LTI_ 62
#define OP_LEI_ 63
#define OP_GTI_ 64
#define OP_GEI_ 65
#define OP_RETURN_ 70

#define CIST_LEQ 0x2000
#define GET_OPCODE_(i) ((i) & 0x7f)
#define GETARG_A_(i) (((i) >> 7) & 0xff)
#define GETARG_k_(i) (((i) >> 15) & 1)

void luaV_finishOp (lua_State *L) {
  CallInfo *ci = L->ci;
  StkId base = ci->func + 1;
  Instruction inst = *(ci->u.l.savedpc - 1);
  int op = GET_OPCODE_(inst);
  switch (op) {
    case OP_MMBIN_: case OP_MMBINI_: case OP_MMBINK_: {
      setobjs2s(L, base + GETARG_A_(*(ci->u.l.savedpc - 2)), --L->top);
      break;
    }
    case 11: case 12: case 13: case 14: case 20: case 49: case 50: case 52: {
      setobjs2s(L, base + GETARG_A_(inst), --L->top);
      break;
    }
    case OP_LT_: case OP_LE_: case OP_LTI_: case OP_LEI_: case OP_GTI_: case OP_GEI_: case OP_EQ_: {
      int res = !l_isfalse(L->top - 1);
      L->top--;
#if defined(LUA_COMPAT_LT_LE)
      if (ci->callstatus & CIST_LEQ) {
        ci->callstatus ^= CIST_LEQ;
        res = !res;
      }
#endif
      if (res != GETARG_k_(inst))
        ci->u.l.savedpc++;
      break;
    }
    case OP_CONCAT_: {
      StkId top = L->top - 1;
      int a = GETARG_A_(inst);
      int total = (int)(top - 1 - (base + a));
      setobjs2s(L, top - 2, top);
      L->top = top - 1;
      luaV_concat(L, total);
      break;
    }
    case OP_CLOSE_: {
      ci->u.l.savedpc--;
      break;
    }
    case OP_RETURN_: {
      L->top = base + GETARG_A_(inst) + ci->u2.nres;
      ci->u.l.savedpc--;
      break;
    }
    default: {
      break;
    }
  }
}
