#include "FuncState.h"
#include "BlockCnt.h"
#include "fn_luaK_codeABCk.h"
#include "reglevel.h"

static void checktoclose (FuncState *fs, int level) {
  if (level != -1) {
    fs->bl->upval = 1;
    fs->bl->insidetbc = 1;
    fs->needclose = 1;
    luaK_codeABCk(fs, 55, reglevel(fs, level), 0, 0, 0);
  }
}
