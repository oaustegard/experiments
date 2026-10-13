#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "expdesc.h"
#include "fn_luaK_ret.h"
#include "fn_luaK_setreturns.h"
#include "fn_luaK_exp2anyreg.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaX_next.h"
#include "fn_luaY_nvarstack.h"
#include "block_follow.h"
#include "explist.h"
#include "testnext.h"

static void retstat (LexState *ls) {
  FuncState *fs = ls->fs;
  expdesc e;
  int nret;
  int first = luaY_nvarstack(fs);
  if (block_follow(ls, 1) || ls->t.token == ';')
    nret = 0;
  else {
    nret = explist(ls, &e);
    if (e.k == 18 || e.k == 19) {
      luaK_setreturns(fs, &e, -1);
      if (e.k == 18 && nret == 1 && !fs->bl->insidetbc) {
        Instruction *pc = &fs->f->code[e.u.info];
        *pc = (*pc & ~0x7fu) | 69u;
      }
      nret = -1;
    }
    else {
      if (nret == 1)
        first = luaK_exp2anyreg(fs, &e);
      else
        luaK_exp2nextreg(fs, &e);
    }
  }
  luaK_ret(fs, first, nret);
  testnext(ls, ';');
}
