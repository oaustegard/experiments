#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "Upvaldesc.h"
#include "Vardesc.h"
#include "expkind.h"
#include "fn_luaO_pushfstring.h"
#include "fn_luaK_semerror.h"

static void check_readonly (LexState *ls, expdesc *e) {
  FuncState *fs = ls->fs;
  TString *varname = NULL;
  switch (e->k) {
    case VCONST: {
      varname = ls->dyd->actvar.arr[e->u.info].vd.name;
      break;
    }
    case VLOCAL: {
      Vardesc *vardesc = getlocalvardesc(fs, e->u.var.vidx);
      if (vardesc->vd.kind != 0)
        varname = vardesc->vd.name;
      break;
    }
    case VUPVAL: {
      Upvaldesc *up = &fs->f->upvalues[e->u.info];
      if (up->kind != 0)
        varname = up->name;
      break;
    }
    default:
      return;
  }
  if (varname) {
    const char *msg = luaO_pushfstring(ls->L,
       "attempt to assign to const variable '%s'", varname->contents);
    luaK_semerror(ls, msg);
  }
}
