#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "fn_luaK_exp2anyregup.h"
#include "fn_luaK_indexed.h"
#include "singlevaraux.h"
#include "str_checkname.h"

static void singlevar (LexState *ls, expdesc *var) {
  TString *varname = str_checkname(ls);
  FuncState *fs = ls->fs;
  singlevaraux(fs, varname, var, 1);
  if (var->k == VVOID) {
    expdesc key;
    singlevaraux(fs, ls->envn, var, 1);
    luaK_exp2anyregup(fs, var);
    key.f = key.t = -1;
    key.k = VKSTR;
    key.u.strval = varname;
    luaK_indexed(fs, var, &key);
  }
}
