#include "LexState.h"
#include "FuncState.h"
#include "expdesc.h"
#include "ConsControl.h"
#include "fn_luaX_next.h"
#include "fn_luaK_exp2val.h"
#include "fn_luaK_indexed.h"
#include "fn_luaK_storevar.h"
#include "error_expected.h"
#include "subexpr.h"
#include "expkind.h"
#include "TString.h"

static void recfield (LexState *ls, ConsControl *cc) {
  FuncState *fs = ls->fs;
  int reg = ls->fs->freereg;
  expdesc tab, key, val;
  if (ls->t.token == 0x123) {
    TString *name = ls->t.seminfo.ts;
    luaX_next(ls);
    key.f = key.t = -1;
    key.k = VKSTR;
    key.u.strval = name;
  }
  else {
    luaX_next(ls);
    subexpr(ls, &key, 0);
    luaK_exp2val(ls->fs, &key);
    if (ls->t.token != ']')
      error_expected(ls, ']');
    luaX_next(ls);
  }
  cc->nh++;
  if (ls->t.token != '=')
    error_expected(ls, '=');
  luaX_next(ls);
  tab = *cc->t;
  luaK_indexed(fs, &tab, &key);
  subexpr(ls, &val, 0);
  luaK_storevar(fs, &tab, &val);
  fs->freereg = reg;
}
