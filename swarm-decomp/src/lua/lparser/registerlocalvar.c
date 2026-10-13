#include "FuncState.h"
#include "LocVar.h"
#include "fn_luaM_growaux_.h"
#include "luaC_objbarrier.h"
#include "registerlocalvar.h"

static int registerlocalvar (LexState *ls, FuncState *fs, TString *varname) {
  Proto *f = fs->f;
  int oldsize = f->sizelocvars;
  f->locvars = (LocVar *)luaM_growaux_(ls->L, f->locvars, fs->ndebugvars,
                 &f->sizelocvars, sizeof(LocVar), 0x7fff, "local variables");
  while (oldsize < f->sizelocvars)
    f->locvars[oldsize++].varname = NULL;
  f->locvars[fs->ndebugvars].varname = varname;
  f->locvars[fs->ndebugvars].startpc = fs->pc;
  luaC_objbarrier(ls->L, f, varname);
  return fs->ndebugvars++;
}
