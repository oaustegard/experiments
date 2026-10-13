#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "expdesc.h"
#include "fn_luaK_exp2const.h"
#include "fn_luaK_codeABCk.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaK_semerror.h"
#include "fn_luaX_next.h"
#include "checknext.h"
#include "str_checkname.h"
#include "new_localvar.h"
#include "adjustlocalvars.h"
#include "adjust_assign.h"
#include "explist.h"
#include "getlocalattribute.h"
#include "checktoclose.h"
#include "subexpr.h"
#include "reglevel.h"

static void localstat (LexState *ls) {
  FuncState *fs = ls->fs;
  int toclose = -1;
  Vardesc *var;
  int vidx, kind;
  int nvars = 0;
  int nexps;
  expdesc e;
  do {
    vidx = new_localvar(ls, str_checkname(ls));
    kind = getlocalattribute(ls);
    fs->ls->dyd->actvar.arr[fs->firstlocal + vidx].vd.kind = kind;
    if (kind == 2) {
      if (toclose != -1)
        luaK_semerror(ls, "multiple to-be-closed variables in local list");
      toclose = fs->nactvar + nvars;
    }
    nvars++;
  } while (ls->t.token == ',' && (luaX_next(ls), 1));
  if (ls->t.token == '=') {
    luaX_next(ls);
    nexps = explist(ls, &e);
  }
  else {
    e.k = VVOID;
    nexps = 0;
  }
  var = &fs->ls->dyd->actvar.arr[fs->firstlocal + vidx];
  if (nvars == nexps && var->vd.kind == 1 &&
      luaK_exp2const(fs, &e, &var->k)) {
    var->vd.kind = 3;
    adjustlocalvars(ls, nvars - 1);
    fs->nactvar++;
  }
  else {
    adjust_assign(ls, nvars, nexps, &e);
    adjustlocalvars(ls, nvars);
  }
  checktoclose(fs, toclose);
}
