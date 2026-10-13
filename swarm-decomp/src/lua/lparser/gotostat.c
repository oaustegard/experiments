#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "Labeldesc.h"
#include "fn_luaK_jump.h"
#include "fn_luaK_patchlist.h"
#include "fn_luaK_codeABCk.h"
#include "str_checkname.h"
#include "findlabel.h"
#include "newlabelentry.h"
#include "reglevel.h"
#include "fn_luaY_nvarstack.h"

static void gotostat (LexState *ls) {
  FuncState *fs = ls->fs;
  int line = ls->linenumber;
  TString *name = str_checkname(ls);
  Labeldesc *lb = findlabel(ls, name);
  if (lb == NULL)
    newlabelentry(ls, &ls->dyd->gt, name, line, luaK_jump(fs));
  else {
    int lblevel = reglevel(fs, lb->nactvar);
    if (luaY_nvarstack(fs) > lblevel)
      luaK_codeABCk(fs, 54, lblevel, 0, 0, 0);
    luaK_patchlist(fs, luaK_jump(fs), lb->pc);
  }
}
