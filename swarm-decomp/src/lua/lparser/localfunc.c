#include "LexState.h"
#include "FuncState.h"
#include "Dyndata.h"
#include "expdesc.h"
#include "str_checkname.h"
#include "new_localvar.h"
#include "adjustlocalvars.h"
#include "body.h"

static void localfunc (LexState *ls) {
  expdesc b;
  FuncState *fs = ls->fs;
  int fvar = fs->nactvar;
  new_localvar(ls, str_checkname(ls));
  adjustlocalvars(ls, 1);
  body(ls, &b, 0, ls->linenumber);
  fs->f->locvars[fs->ls->dyd->actvar.arr[fs->firstlocal + fvar].vd.pidx].startpc = fs->pc;
}
