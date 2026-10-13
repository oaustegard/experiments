#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_setreturns.h"
#include "fn_luaK_fixline.h"
#include "fn_luaX_next.h"
#include "fn_luaX_syntaxerror.h"
#include "check_match.h"
#include "subexpr.h"
#include "constructor.h"
#include "testnext.h"
#include "codestring.h"
#include "expkind.h"

static void funcargs (LexState *ls, expdesc *f) {
  FuncState *fs = ls->fs;
  expdesc args;
  int base, nparams, pc;
  int line = ls->linenumber;
  switch (ls->t.token) {
    case '(': {
      luaX_next(ls);
      if (ls->t.token == ')') {
        args.k = VVOID;
      }
      else {
        subexpr(ls, &args, 0);
        while (testnext(ls, ',')) {
          luaK_exp2nextreg(ls->fs, &args);
          subexpr(ls, &args, 0);
        }
        if (args.k == VCALL || args.k == VVARARG)
          luaK_setreturns(fs, &args, -1);
      }
      check_match(ls, ')', '(', line);
      break;
    }
    case 0x124: {
      codestring(&args, ls->t.seminfo.ts);
      luaX_next(ls);
      break;
    }
    case '{': {
      constructor(ls, &args);
      break;
    }
    default: {
      luaX_syntaxerror(ls, "function arguments expected");
    }
  }
  base = f->u.info;
  if (args.k == VCALL || args.k == VVARARG) {
    nparams = -1;
  }
  else {
    if (args.k != VVOID)
      luaK_exp2nextreg(fs, &args);
    nparams = fs->freereg - (base + 1);
  }
  pc = luaK_codeABCk(fs, 0x44, base, nparams + 1, 2, 0);
  f->f = f->t = -1;
  f->k = VCALL;
  f->u.info = pc;
  luaK_fixline(fs, line);
  fs->freereg = base + 1;
}
