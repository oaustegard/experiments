#include "FuncState.h"
#include "fn_luaK_goiftrue.h"
#include "subexpr.h"

static int cond (LexState *ls) {
  expdesc v;
  subexpr(ls, &v, 0);
  if (v.k == VNIL)
    v.k = VFALSE;
  luaK_goiftrue(ls->fs, &v);
  return v.f;
}
