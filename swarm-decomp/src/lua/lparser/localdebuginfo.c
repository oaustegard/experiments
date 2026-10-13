#include "FuncState.h"
#include "Proto.h"
#include "LocVar.h"
#include "Vardesc.h"

static LocVar *localdebuginfo (FuncState *fs, int vidx) {
  Vardesc *vd = getlocalvardesc(fs, vidx);
  if (vd->vd.kind != 3)
    return &fs->f->locvars[vd->vd.pidx];
  else
    return NULL;
}
