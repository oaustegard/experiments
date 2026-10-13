#include "FuncState.h"
#include "fn_luaM_growaux_.h"
#include "errorlimit.h"

static int new_localvar (LexState *ls, TString *name) {
  lua_State *L = ls->L;
  FuncState *fs = ls->fs;
  Dyndata *dyd = ls->dyd;
  Vardesc *var;
  if (dyd->actvar.n + 1 - fs->firstlocal > 200)
    errorlimit(fs, 200, "local variables");
  dyd->actvar.arr = (Vardesc *)luaM_growaux_(L, dyd->actvar.arr, dyd->actvar.n + 1,
                       &dyd->actvar.size, sizeof(Vardesc), 0xffff, "local variables");
  var = &dyd->actvar.arr[dyd->actvar.n++];
  var->vd.kind = 0;
  var->vd.name = name;
  return dyd->actvar.n - 1 - fs->firstlocal;
}
