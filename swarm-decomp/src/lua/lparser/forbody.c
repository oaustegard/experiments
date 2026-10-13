#include "FuncState.h"
#include "BlockCnt.h"
#include "fn_luaK_codeABx.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaK_fixline.h"
#include "fn_luaX_syntaxerror.h"
#include "checknext.h"
#include "enterblock.h"
#include "leaveblock.h"
#include "block.h"
#include "luaK_getlabel.h"
#include "adjustlocalvars.h"
#include "fixforjump.h"
#include "forbody.h"

static void forbody (LexState *ls, int base, int line, int nvars, int isgen) {
  static const OpCode forprep[2] = {0x4a, 0x4b};
  static const OpCode forloop[2] = {0x49, 0x4d};
  BlockCnt bl;
  FuncState *fs = ls->fs;
  int prep, endfor;
  checknext(ls, 0x102);
  prep = luaK_codeABx(fs, forprep[isgen], base, 0);
  enterblock(fs, &bl, 0);
  adjustlocalvars(ls, nvars);
  luaK_reserveregs(fs, nvars);
  block(ls);
  leaveblock(fs);
  fixforjump(fs, prep, luaK_getlabel(fs), 0);
  if (isgen) {
    luaK_codeABCk(fs, 0x4c, base, 0, nvars, 0);
    luaK_fixline(fs, line);
  }
  endfor = luaK_codeABx(fs, forloop[isgen], base, 0);
  fixforjump(fs, endfor, prep + 1, 1);
  luaK_fixline(fs, line);
}
