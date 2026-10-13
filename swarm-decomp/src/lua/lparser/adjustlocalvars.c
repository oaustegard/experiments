#include "FuncState.h"
#include "Dyndata.h"
#include "Vardesc.h"
#include "reglevel.h"
#include "registerlocalvar.h"
#include "adjustlocalvars.h"

static void adjustlocalvars (LexState *ls, int nvars) {
  FuncState *fs = ls->fs;
  int reglevel_ = reglevel(fs, fs->nactvar);
  int i;
  for (i = 0; i < nvars; i++) {
    int vidx = fs->nactvar++;
    Vardesc *var = getlocalvardesc(fs, vidx);
    var->vd.ridx = reglevel_++;
    var->vd.pidx = registerlocalvar(ls, fs, var->vd.name);
  }
}
