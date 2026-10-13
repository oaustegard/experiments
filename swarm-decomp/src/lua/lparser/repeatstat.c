#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "luaK_getlabel.h"
#include "fn_luaK_jump.h"
#include "fn_luaK_patchlist.h"
#include "fn_luaK_patchtohere.h"
#include "fn_luaK_codeABCk.h"
#include "check_match.h"
#include "enterblock.h"
#include "leaveblock.h"
#include "statlist.h"
#include "cond.h"
#include "reglevel.h"

static void repeatstat (LexState *ls, int line) {
  int condexit;
  FuncState *fs = ls->fs;
  int repeat_init = luaK_getlabel(fs);
  BlockCnt bl1, bl2;
  enterblock(fs, &bl1, 1);
  enterblock(fs, &bl2, 0);
  luaX_next(ls);
  statlist(ls);
  check_match(ls, 0x114, 0x110, line);
  condexit = cond(ls);
  leaveblock(fs);
  if (bl2.upval) {
    int exit = luaK_jump(fs);
    luaK_patchtohere(fs, condexit);
    luaK_codeABCk(fs, 54, reglevel(fs, bl2.nactvar), 0, 0, 0);
    condexit = luaK_jump(fs);
    luaK_patchtohere(fs, exit);
  }
  luaK_patchlist(fs, condexit, repeat_init);
  leaveblock(fs);
}
