#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "luaK_getlabel.h"
#include "fn_luaK_jump.h"
#include "fn_luaK_patchlist.h"
#include "fn_luaK_patchtohere.h"
#include "fn_luaX_next.h"
#include "check_match.h"
#include "checknext.h"
#include "enterblock.h"
#include "leaveblock.h"
#include "block.h"
#include "cond.h"

static void whilestat (LexState *ls, int line) {
  FuncState *fs = ls->fs;
  int whileinit;
  int condexit;
  BlockCnt bl;
  luaX_next(ls);
  whileinit = luaK_getlabel(fs);
  condexit = cond(ls);
  enterblock(fs, &bl, 1);
  checknext(ls, 0x102);
  block(ls);
  luaK_patchlist(fs, luaK_jump(fs), whileinit);
  check_match(ls, 0x105, 0x115, line);
  leaveblock(fs);
  luaK_patchtohere(fs, condexit);
}
