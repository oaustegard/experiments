#include "LexState.h"
#include "FuncState.h"
#include "expkind.h"
#include "fn_luaK_exp2nextreg.h"
#include "fn_luaK_nil.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaK_setreturns.h"
#include "adjust_assign.h"

static void adjust_assign (LexState *ls, int nvars, int nexps, expdesc *e) {
  FuncState *fs = ls->fs;
  int needed = nvars - nexps;
  if (e->k == VCALL || e->k == VVARARG) {
    int extra = needed + 1;
    if (extra < 0)
      extra = 0;
    luaK_setreturns(fs, e, extra);
  }
  else {
    if (e->k != VVOID)
      luaK_exp2nextreg(fs, e);
    if (needed > 0)
      luaK_nil(fs, fs->freereg, needed);
  }
  if (needed > 0)
    luaK_reserveregs(fs, needed);
  else
    fs->freereg += needed;
}
