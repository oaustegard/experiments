#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "UnOpr.h"
#include "BinOpr.h"
#include "fn_luaE_incCstack.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_prefix.h"
#include "fn_luaK_infix.h"
#include "fn_luaK_posfix.h"
#include "fn_luaX_next.h"
#include "fn_luaX_syntaxerror.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_setoneret.h"
#include "fn_luaS_newlstr.h"
#include "subexpr.h"
#include "suffixedexp.h"
#include "constructor.h"
#include "init_exp.h"
#include "codestring.h"
#include "body.h"

#define UNARY_PRIORITY 12

static UnOpr getunopr (int op) {
  switch (op) {
    case 0x10e: return OPR_NOT;
    case '-': return OPR_MINUS;
    case '~': return OPR_BNOT;
    case '#': return OPR_LEN;
    default: return OPR_NOUNOPR;
  }
}

static BinOpr getbinopr (int op) {
  switch (op) {
    case '+': return OPR_ADD;
    case '-': return OPR_SUB;
    case '*': return OPR_MUL;
    case '%': return OPR_MOD;
    case '^': return OPR_POW;
    case '/': return OPR_DIV;
    case 0x116: return OPR_IDIV;
    case '&': return OPR_BAND;
    case '|': return OPR_BOR;
    case '~': return OPR_BXOR;
    case 0x11d: return OPR_SHL;
    case 0x11e: return OPR_SHR;
    case 0x117: return OPR_CONCAT;
    case 0x11c: return OPR_NE;
    case 0x119: return OPR_EQ;
    case '<': return OPR_LT;
    case 0x11b: return OPR_LE;
    case '>': return OPR_GT;
    case 0x11a: return OPR_GE;
    case 0x100: return OPR_AND;
    case 0x10f: return OPR_OR;
    default: return OPR_NOBINOPR;
  }
}

static void simpleexp (LexState *ls, expdesc *v) {
  switch (ls->t.token) {
    case 0x121: {
      init_exp(v, VKFLT, 0);
      v->u.nval = ls->t.seminfo.r;
      break;
    }
    case 0x122: {
      init_exp(v, VKINT, 0);
      v->u.ival = ls->t.seminfo.i;
      break;
    }
    case 0x124: {
      codestring(v, ls->t.seminfo.ts);
      break;
    }
    case 0x10d: {
      init_exp(v, VNIL, 0);
      break;
    }
    case 0x113: {
      init_exp(v, VTRUE, 0);
      break;
    }
    case 0x106: {
      init_exp(v, VFALSE, 0);
      break;
    }
    case 0x118: {
      FuncState *fs = ls->fs;
      if (!fs->f->is_vararg)
        luaX_syntaxerror(ls, "cannot use '...' outside a vararg function");
      init_exp(v, VVARARG, luaK_codeABCk(fs, 0x50, 0, 0, 1, 0));
      break;
    }
    case '{':
      constructor(ls, v);
      return;
    case 0x108:
      luaX_next(ls);
      body(ls, v, 0, ls->linenumber);
      return;
    default:
      suffixedexp(ls, v);
      return;
  }
  luaX_next(ls);
}

static BinOpr subexpr (LexState *ls, expdesc *v, int limit) {
  BinOpr op;
  UnOpr uop;
  luaE_incCstack(ls->L);
  uop = getunopr(ls->t.token);
  if (uop != OPR_NOUNOPR) {
    int line = ls->linenumber;
    luaX_next(ls);
    subexpr(ls, v, UNARY_PRIORITY);
    luaK_prefix(ls->fs, uop, v, line);
  }
  else simpleexp(ls, v);
  op = getbinopr(ls->t.token);
  while (op != OPR_NOBINOPR && priority[op].left > limit) {
    expdesc v2;
    BinOpr nextop;
    int line = ls->linenumber;
    luaX_next(ls);
    luaK_infix(ls->fs, op, v);
    nextop = subexpr(ls, &v2, priority[op].right);
    luaK_posfix(ls->fs, op, v, &v2, line);
    op = nextop;
  }
  ls->L->nCcalls--;
  return op;
}
