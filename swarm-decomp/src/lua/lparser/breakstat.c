#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "fn_luaS_newlstr.h"
#include "fn_luaK_jump.h"
#include "fn_luaX_next.h"
#include "newlabelentry.h"

static void breakstat (LexState *ls) {
  int line = ls->linenumber;
  luaX_next(ls);
  newlabelentry(ls, &ls->dyd->gt, luaS_newlstr(ls->L, "break", 5), line, luaK_jump(ls->fs));
}
