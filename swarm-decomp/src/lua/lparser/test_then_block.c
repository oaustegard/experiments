#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "fn_luaX_next.h"
#include "fn_luaK_goiffalse.h"
#include "fn_luaK_goiftrue.h"
#include "fn_luaK_jump.h"
#include "fn_luaK_concat.h"
#include "fn_luaK_patchtohere.h"
#include "fn_luaS_newlstr.h"
#include "subexpr.h"
#include "leaveblock.h"
#include "newlabelentry.h"
#include "error_expected.h"
#include "statement.h"

static void test_then_block (LexState *ls, int *escapelist) {
  BlockCnt bl;
  FuncState *fs = ls->fs;
  expdesc v;
  int jf;
  luaX_next(ls);
  subexpr(ls, &v, 0);
  if (ls->t.token != 0x112)
    error_expected(ls, 0x112);
  luaX_next(ls);
  if (ls->t.token == 0x101) {  /* if x then break */
    int line = ls->linenumber;
    luaK_goiffalse(fs, &v);
    luaX_next(ls);
    enterblock(fs, &bl, 0);
    newlabelentry(ls, &ls->dyd->gt, luaS_newlstr(ls->L, "break", 5), line, v.t);
    while (ls->t.token == ';')
      luaX_next(ls);
    if (block_follow(ls, 0)) {
      leaveblock(fs);
      return;
    }
    else
      jf = luaK_jump(fs);
  }
  else {
    luaK_goiftrue(fs, &v);
    enterblock(fs, &bl, 0);
    jf = v.f;
  }
  statlist(ls);
  leaveblock(fs);
  if (ls->t.token == 0x103 || ls->t.token == 0x104)
    luaK_concat(fs, escapelist, luaK_jump(fs));
  luaK_patchtohere(fs, jf);
}
