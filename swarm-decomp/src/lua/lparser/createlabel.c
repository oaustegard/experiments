#include "LexState.h"
#include "FuncState.h"
#include "BlockCnt.h"
#include "Dyndata.h"
#include "luaK_getlabel.h"
#include "newlabelentry.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaY_nvarstack.h"

static int createlabel (LexState *ls, TString *name, int line, int last) {
  FuncState *fs = ls->fs;
  Labellist *ll = &ls->dyd->label;
  int l = newlabelentry(ls, ll, name, line, luaK_getlabel(fs));
  if (last)
    ll->arr[l].nactvar = fs->bl->nactvar;
  if (solvegotos(ls, &ll->arr[l])) {
    luaK_codeABCk(fs, 0x36, luaY_nvarstack(fs), 0, 0, 0);
    return 1;
  }
  return 0;
}
