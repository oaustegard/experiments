#include "LexState.h"
#include "FuncState.h"
#include "fn_luaK_int.h"
#include "fn_luaK_reserveregs.h"
#include "fn_luaX_newstring.h"
#include "fn_luaX_next.h"
#include "checknext.h"
#include "exp1.h"
#include "forbody.h"
#include "new_localvar.h"
#include "adjustlocalvars.h"

static void fornum (LexState *ls, TString *varname, int line) {
  FuncState *fs = ls->fs;
  int base = fs->freereg;
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, luaX_newstring(ls, "(for state)", 11));
  new_localvar(ls, varname);
  checknext(ls, '=');
  exp1(ls);
  checknext(ls, ',');
  exp1(ls);
  if (ls->t.token == ',') {
    luaX_next(ls);
    exp1(ls);
  }
  else {
    luaK_int(fs, fs->freereg, 1);
    luaK_reserveregs(fs, 1);
  }
  adjustlocalvars(ls, 3);
  forbody(ls, base, line, 1, 0);
}
