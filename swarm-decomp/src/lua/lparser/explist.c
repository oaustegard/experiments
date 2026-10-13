#include "FuncState.h"
#include "fn_luaX_next.h"
#include "fn_luaK_exp2nextreg.h"
#include "subexpr.h"

static int explist (LexState *ls, expdesc *v) {
  int n = 1;
  subexpr(ls, v, 0);
  while (ls->t.token == ',') {
    luaK_exp2nextreg(ls->fs, v);
    luaX_next(ls);
    subexpr(ls, v, 0);
    n++;
  }
  return n;
}
