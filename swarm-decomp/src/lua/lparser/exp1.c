#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "fn_luaK_exp2nextreg.h"
#include "subexpr.h"

static void exp1 (LexState *ls) {
  expdesc e;
  subexpr(ls, &e, 0);
  luaK_exp2nextreg(ls->fs, &e);
}
